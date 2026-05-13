#!/usr/bin/env python3
"""Fetch or unpack a Fantasy Map Generator atlas bundle for SnowWeave.

This script is intentionally data-prep only. It extracts the bundle and writes
an index that SnowWeave's map pipeline can consume.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


DEFAULT_FMG_BACKEND_URL = "http://127.0.0.1:8765"
DEFAULT_CHUNK_SIZE = 4096


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_bundle(*, backend_url: str, seed: str | None, chunk_size: int, output_zip: Path) -> Path:
    query: dict[str, str] = {"chunk_size": str(chunk_size)}
    if seed:
        query["seed"] = seed
    url = f"{backend_url.rstrip('/')}/exports/atlas.bundle.zip?{urlencode(query)}"
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=600) as response:
        output_zip.write_bytes(response.read())
    return output_zip


def extract_bundle(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)


def rect_from_chunk(chunk: dict[str, Any]) -> dict[str, float]:
    origin = chunk.get("sourceOrigin") or chunk.get("origin") or {}
    size = chunk.get("size") or {}
    x = float(origin.get("x", 0))
    y = float(origin.get("y", 0))
    width = float(size.get("width", DEFAULT_CHUNK_SIZE))
    height = float(size.get("height", DEFAULT_CHUNK_SIZE))
    return {"xMin": x, "yMin": y, "xMax": x + width, "yMax": y + height}


def has_items(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0


def build_legend_context(chunk_manifest: dict[str, Any]) -> str:
    lines = [
        "FANTASY ATLAS LEGEND CONTEXT:",
        "The attached reference image is an abstract Fantasy Map Generator atlas chunk. Interpret its legend semantically as game-world terrain, but do not draw labels, symbols, UI, or legend text into the output image.",
        "You may beautify and repaint the abstract atlas into a polished clean HD RPG basemap. Keep the terrain topology, coastline, water network, route/path network, elevation distribution, and major terrain boundaries unchanged; only improve the visual treatment.",
    ]

    ocean = chunk_manifest.get("oceanLayers") if isinstance(chunk_manifest.get("oceanLayers"), dict) else {}
    heights = chunk_manifest.get("heightLayers") if isinstance(chunk_manifest.get("heightLayers"), dict) else {}
    entries: list[str] = []

    if has_items(ocean.get("offshore")):
        entries.append("- offshore ocean regions: deep open sea or distant navigable water background.")
    if has_items(ocean.get("nearshore")):
        entries.append("- nearshore ocean regions: shallow coastal water and readable coastline transitions.")
    if has_items(heights.get("0")):
        entries.append("- height layer 1: lowlands such as grass, beaches, plains, and soft walkable terrain.")
    if has_items(heights.get("1")):
        entries.append("- height layer 2: raised hills, rocky slopes, and secondary elevation bands.")
    if has_items(heights.get("2")):
        entries.append("- height layer 3: high mountains, cliffs, ridges, and blocked or difficult terrain.")
    if has_items(chunk_manifest.get("connectors")):
        entries.append("- connector regions: natural passes, ramps, saddles, or traversable elevation transitions.")
    if has_items(chunk_manifest.get("lakes")):
        entries.append("- lakes: inland water bodies with readable shores.")
    if has_items(chunk_manifest.get("rivers")):
        entries.append("- rivers: flowing water corridors, narrow water obstacles, or route guides.")
    if has_items(chunk_manifest.get("routes")):
        entries.append("- routes: roads, dirt tracks, trails, or intentional traversal paths; preserve their topology.")

    if not entries:
        entries.append("- visible atlas shapes: preserve the terrain topology and reinterpret it as clean playable ground.")

    lines.extend(entries)
    lines.append(
        "Preserve the chunk's coastline, water paths, lake positions, route topology, elevation distribution, and major terrain boundaries while rendering them as a clean playable top-down RPG basemap."
    )
    return "\n".join(lines)


def build_index(extract_dir: Path, output_index: Path, *, seed: str | None, source_zip: Path) -> dict[str, Any]:
    manifest_path = extract_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest.json in {extract_dir}")

    manifest = read_json(manifest_path)
    chunks: dict[str, Any] = {}
    for chunk in manifest.get("chunks", []):
        if not isinstance(chunk, dict) or not chunk.get("id"):
            continue
        chunk_id = str(chunk["id"])
        chunk_manifest_path = extract_dir / str(chunk.get("manifest", f"chunks/{chunk_id}.json"))
        chunk_png_path = extract_dir / str(chunk.get("png", f"chunks/{chunk_id}.png"))
        if not chunk_manifest_path.exists():
            raise SystemExit(f"Missing chunk manifest: {chunk_manifest_path}")
        if not chunk_png_path.exists():
            raise SystemExit(f"Missing chunk PNG: {chunk_png_path}")

        chunk_manifest = read_json(chunk_manifest_path)
        chunks[chunk_id] = {
            "id": chunk_id,
            "column": chunk.get("column"),
            "row": chunk.get("row"),
            "png": str(chunk_png_path.resolve()),
            "manifest": str(chunk_manifest_path.resolve()),
            "sourceOrigin": chunk.get("sourceOrigin") or chunk.get("origin"),
            "origin": chunk.get("origin"),
            "size": chunk.get("size"),
            "bounds": chunk.get("bounds") or rect_from_chunk(chunk),
            "overlap": chunk.get("overlap"),
            "legend_context": build_legend_context(chunk_manifest),
        }

    index = {
        "version": 1,
        "kind": "snowweave-fmg-reference-index",
        "seed": seed or manifest.get("seed"),
        "source_zip": str(source_zip.resolve()),
        "extract_dir": str(extract_dir.resolve()),
        "atlas_png": str((extract_dir / "atlas.png").resolve()) if (extract_dir / "atlas.png").exists() else "",
        "manifest": str(manifest_path.resolve()),
        "chunk_size": manifest.get("chunkSize", DEFAULT_CHUNK_SIZE),
        "map_size": manifest.get("mapSize"),
        "grid": manifest.get("grid"),
        "chunks": chunks,
        "generation_order": ["chunk_0_0", "chunk_1_0", "chunk_0_1", "chunk_1_1"],
    }
    write_json(output_index, index)
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, help="Existing atlas bundle zip. If omitted, the script fetches one.")
    parser.add_argument("--backend-url", default=DEFAULT_FMG_BACKEND_URL)
    parser.add_argument("--seed")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-name", default="fmg_reference_index.json")
    parser.add_argument("--force", action="store_true", help="Remove output dir before extracting.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.chunk_size != DEFAULT_CHUNK_SIZE:
        raise SystemExit("SnowWeave FMG basemap integration currently requires fixed 4096px chunks.")

    output_dir = args.output_dir.resolve()
    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = args.zip
    if zip_path:
        zip_path = zip_path.resolve()
        if not zip_path.exists():
            raise SystemExit(f"Bundle zip not found: {zip_path}")
    else:
        zip_path = fetch_bundle(
            backend_url=args.backend_url,
            seed=args.seed,
            chunk_size=args.chunk_size,
            output_zip=output_dir / "atlas-bundle.zip",
        )

    extract_bundle(zip_path, output_dir)
    index = build_index(output_dir, output_dir / args.index_name, seed=args.seed, source_zip=zip_path)
    print(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
