#!/usr/bin/env python3
"""Create a stitched reference image for an FMG chunk.

The current chunk remains 4096x4096. Any overlap with already generated
basemaps is replaced by pixels from those generated basemaps so later chunks
inherit the established SnowWeave visual style.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_CHUNK_SIZE = 4096


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_generated(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected CHUNK_ID=BASEMAP_PATH")
    chunk_id, path = value.split("=", 1)
    chunk_id = chunk_id.strip()
    if not chunk_id:
        raise argparse.ArgumentTypeError("Missing CHUNK_ID")
    return chunk_id, Path(path.strip())


def rect_from_chunk(chunk: dict[str, Any]) -> dict[str, float]:
    origin = chunk.get("sourceOrigin") or chunk.get("origin") or {}
    size = chunk.get("size") or {}
    x = float(origin.get("x", 0))
    y = float(origin.get("y", 0))
    width = float(size.get("width", DEFAULT_CHUNK_SIZE))
    height = float(size.get("height", DEFAULT_CHUNK_SIZE))
    return {"xMin": x, "yMin": y, "xMax": x + width, "yMax": y + height}


def intersect(a: dict[str, float], b: dict[str, float]) -> dict[str, int] | None:
    x_min = max(a["xMin"], b["xMin"])
    y_min = max(a["yMin"], b["yMin"])
    x_max = min(a["xMax"], b["xMax"])
    y_max = min(a["yMax"], b["yMax"])
    if x_max <= x_min or y_max <= y_min:
        return None
    return {
        "xMin": int(round(x_min)),
        "yMin": int(round(y_min)),
        "xMax": int(round(x_max)),
        "yMax": int(round(y_max)),
    }


def local_box(global_rect: dict[str, int], chunk_rect: dict[str, float]) -> tuple[int, int, int, int]:
    x0 = int(round(global_rect["xMin"] - chunk_rect["xMin"]))
    y0 = int(round(global_rect["yMin"] - chunk_rect["yMin"]))
    x1 = int(round(global_rect["xMax"] - chunk_rect["xMin"]))
    y1 = int(round(global_rect["yMax"] - chunk_rect["yMin"]))
    return x0, y0, x1, y1


def load_chunk(index: dict[str, Any], chunk_id: str) -> dict[str, Any]:
    chunks = index.get("chunks")
    if not isinstance(chunks, dict) or chunk_id not in chunks:
        raise SystemExit(f"Chunk not found in index: {chunk_id}")
    chunk = chunks[chunk_id]
    if not isinstance(chunk, dict):
        raise SystemExit(f"Invalid chunk entry: {chunk_id}")
    return chunk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--chunk-id", required=True)
    parser.add_argument("--generated", action="append", type=parse_generated, default=[], help="Already generated basemap as CHUNK_ID=PATH. Repeatable.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    index = read_json(args.index)
    current = load_chunk(index, args.chunk_id)
    current_rect = rect_from_chunk(current)
    current_png = Path(str(current["png"]))
    if not current_png.exists():
        raise SystemExit(f"Current chunk PNG not found: {current_png}")

    canvas = Image.open(current_png).convert("RGBA")
    replacements: list[dict[str, Any]] = []

    for source_chunk_id, basemap_path in args.generated:
        source = load_chunk(index, source_chunk_id)
        source_rect = rect_from_chunk(source)
        overlap = intersect(current_rect, source_rect)
        if not overlap:
            continue
        if not basemap_path.exists():
            raise SystemExit(f"Generated basemap not found: {basemap_path}")

        src_box = local_box(overlap, source_rect)
        dst_box = local_box(overlap, current_rect)
        patch = Image.open(basemap_path).convert("RGBA").crop(src_box)
        expected_size = (dst_box[2] - dst_box[0], dst_box[3] - dst_box[1])
        if patch.size != expected_size:
            patch = patch.resize(expected_size, Image.Resampling.LANCZOS)
        canvas.paste(patch, (dst_box[0], dst_box[1]))
        replacements.append(
            {
                "from_chunk": source_chunk_id,
                "from_basemap": str(basemap_path.resolve()),
                "global_overlap": overlap,
                "source_crop_box": src_box,
                "target_paste_box": dst_box,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)

    report_path = args.report or args.output.with_suffix(".stitch_report.json")
    report = {
        "version": 1,
        "chunk_id": args.chunk_id,
        "input_png": str(current_png.resolve()),
        "output": str(args.output.resolve()),
        "current_rect": current_rect,
        "replacements": replacements,
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
