#!/usr/bin/env python3
"""Run the SnowWeave full map pipeline from one FMG 4K chunk reference."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SNOWWEAVE_ROOT = Path(__file__).resolve().parents[1]
ASF_SCRIPTS = SNOWWEAVE_ROOT / "dependencies" / "agent-sprite-forge" / "scripts"
DEFAULT_FMG_BACKEND_URL = "http://127.0.0.1:8765"
DEFAULT_OUTPUT_ROOT = SNOWWEAVE_ROOT / "out" / "maps"
DEFAULT_CHUNK_SIZE = 4096

sys.path.insert(0, str(ASF_SCRIPTS))
import asf  # type: ignore  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def api_key_from_env() -> str | None:
    for name in ("OPENROUTER_API_KEY", "NAGA_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def run_unpack(args: argparse.Namespace, output_dir: Path) -> Path:
    ref_dir = output_dir / "fmg-reference"
    command = [
        sys.executable,
        str(SNOWWEAVE_ROOT / "scripts" / "fmg_unpack_atlas_bundle.py"),
        "--output-dir",
        str(ref_dir),
        "--backend-url",
        args.fmg_backend_url,
        "--chunk-size",
        str(args.fmg_chunk_size),
        "--force",
    ]
    if args.fmg_seed:
        command.extend(["--seed", args.fmg_seed])
    if args.fmg_bundle_zip:
        command.extend(["--zip", str(args.fmg_bundle_zip)])

    print("[FMG] unpack command:", " ".join(command), flush=True)
    subprocess.check_call(command)
    return ref_dir / "fmg_reference_index.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-name")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="gemini3.1flash")
    parser.add_argument("--map-mode", default="scene_mode")
    parser.add_argument("--image", type=Path, action="append", default=[], help="Extra reference image. Repeatable.")
    parser.add_argument("--prompt-context", default="")
    parser.add_argument("--fmg-seed")
    parser.add_argument("--fmg-chunk-id", default="chunk_0_0")
    parser.add_argument("--fmg-chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--fmg-backend-url", default=DEFAULT_FMG_BACKEND_URL)
    parser.add_argument("--fmg-bundle-zip", type=Path)
    parser.add_argument("--sub-diff-threshold", type=float, default=30.0)
    parser.add_argument("--sub-min-component-area", type=int, default=100)
    parser.add_argument("--sub-no-shadow-suppression", action="store_true")
    parser.add_argument("--sub-no-edge-delta", action="store_true", default=True)
    parser.add_argument("--sub-edge-threshold", type=float, default=18.0)
    parser.add_argument("--sub-edge-grow-radius", type=int, default=2)
    parser.add_argument("--sub-edge-support-radius", type=int, default=10)
    parser.add_argument("--sub-no-fill-holes", action="store_true")
    parser.add_argument("--sub-matting-backend", choices=["auto", "none", "rembg"], default="auto")
    parser.add_argument("--sub-no-rembg-alpha-matting", action="store_true")
    parser.add_argument("--sub-no-constrain-rembg-to-diff-mask", action="store_true")
    parser.add_argument("--no-catalog", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.fmg_chunk_size != DEFAULT_CHUNK_SIZE:
        raise SystemExit("FMG SnowWeave pipeline currently requires fixed 4096px chunks.")

    output_dir = args.output_dir
    if output_dir is None:
        output_name = args.output_name or f"fmg-{args.fmg_chunk_id}-{time.strftime('%Y%m%d-%H%M%S')}"
        output_dir = DEFAULT_OUTPUT_ROOT / output_name
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SnowWeave] output_dir={output_dir}", flush=True)
    print(f"[SnowWeave] chunk={args.fmg_chunk_id} size={args.fmg_chunk_size}", flush=True)

    index_path = run_unpack(args, output_dir)
    index = read_json(index_path)
    chunks = index.get("chunks") if isinstance(index.get("chunks"), dict) else {}
    chunk = chunks.get(args.fmg_chunk_id)
    if not isinstance(chunk, dict):
        raise SystemExit(f"Chunk not found in FMG index: {args.fmg_chunk_id}")

    fmg_image = Path(str(chunk["png"]))
    image_paths = [fmg_image, *args.image]
    prompt_context = "\n\n".join(
        text for text in [args.prompt_context.strip(), str(chunk.get("legend_context") or "").strip()] if text
    )

    print(f"[SnowWeave] fmg_reference={fmg_image}", flush=True)
    print("[SnowWeave] starting ASF full pipeline: base -> dressed -> subtract props -> preview", flush=True)

    result = asf.generate_map_pipeline(
        prompt=args.prompt,
        output_dir=output_dir,
        image_paths=image_paths,
        prompt_context=prompt_context,
        reference_metadata={
            "fmg_reference": {
                "index_path": str(index_path.resolve()),
                "chunk_id": args.fmg_chunk_id,
                "chunk": chunk,
            }
        },
        model=args.model,
        map_mode=args.map_mode,
        api_key=api_key_from_env(),
        no_catalog=args.no_catalog,
        timeout=args.timeout,
        sub_diff_threshold=args.sub_diff_threshold,
        sub_min_component_area=args.sub_min_component_area,
        sub_no_shadow_suppression=args.sub_no_shadow_suppression,
        sub_no_edge_delta=args.sub_no_edge_delta,
        sub_edge_threshold=args.sub_edge_threshold,
        sub_edge_grow_radius=args.sub_edge_grow_radius,
        sub_edge_support_radius=args.sub_edge_support_radius,
        sub_no_fill_holes=args.sub_no_fill_holes,
        sub_matting_backend=args.sub_matting_backend,
        sub_no_rembg_alpha_matting=args.sub_no_rembg_alpha_matting,
        sub_no_constrain_rembg_to_diff_mask=args.sub_no_constrain_rembg_to_diff_mask,
    )
    print("[SnowWeave] completed", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
