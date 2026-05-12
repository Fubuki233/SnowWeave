"""
Split an ASF-generated sprite sheet without VLM.

The ASF response/manifest already records action row numbers under
`asf.actions[*].row` or `actions[*].row`. This script reads those rows, detects
the visible grid lines in the image, maps action rows to physical row bands, and
then delegates per-action frame extraction to agent-sprite-forge's
generate2dsprite.py when it is available.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

DIRECTION_IDS = ["front", "back", "right"]

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


class FrameSplitter:
    def __init__(
        self,
        dark_threshold: int = 110,
        coverage_threshold: float = 0.25,
        expected_cols: int = 8,
        line_rgb: tuple[int, int, int] = (0, 0, 0),
        line_tolerance: int = 24,
        workers: int = 0,
    ):
        self.dark_threshold = dark_threshold
        self.coverage_threshold = coverage_threshold
        self.expected_cols = expected_cols
        self.line_rgb = line_rgb
        self.line_tolerance = line_tolerance
        self.workers = workers

    def _read_metadata(self, metadata_path: str | Path) -> dict:
        return json.loads(Path(metadata_path).read_text(encoding="utf-8"))

    def _extract_actions(self, metadata: dict) -> list[dict]:
        actions = metadata.get("actions")
        if actions is None:
            actions = metadata.get("asf", {}).get("actions", [])

        clean: list[dict] = []
        for index, item in enumerate(actions):
            if not isinstance(item, dict):
                continue
            action_id = item.get("id") or item.get("action")
            row_start = item.get("row_start", item.get("row"))
            row_end = item.get("row_end", row_start)
            if not action_id or row_start is None or row_end is None:
                continue
            row_start = int(row_start)
            row_end = int(row_end)
            clean.append(
                {
                    "id": str(action_id),
                    "label": str(item.get("label") or action_id),
                    "index": int(item.get("index", index)),
                    "row": row_start,
                    "row_start": row_start,
                    "row_end": row_end,
                    "rows": max(1, row_end - row_start + 1),
                    "directions": item.get("directions", []),
                    "source": item.get("source", "metadata"),
                }
            )
        clean.sort(key=lambda item: (item["index"], item["row"], item["id"]))
        return clean

    def _auto_metadata_path(self, image_path: str | Path) -> Path | None:
        image_path = Path(image_path)
        for name in ("raw.response.json", "raw.manifest.json"):
            candidate = image_path.with_name(name)
            if candidate.exists():
                return candidate
        return None

    def _is_line_pixel(self, r: int, g: int, b: int) -> bool:
        tr, tg, tb = self.line_rgb
        return (
            abs(r - tr) <= self.line_tolerance
            and abs(g - tg) <= self.line_tolerance
            and abs(b - tb) <= self.line_tolerance
        )

    def _line_runs(
        self, img: Image.Image, axis: str, coverage_threshold: float | None = None
    ) -> list[tuple[int, int]]:
        """Detect separator lines by strict RGB channel matching."""
        rgb = img.convert("RGB")
        width, height = rgb.size
        pixels = rgb.load()
        limit = height if axis == "horizontal" else width
        span = width if axis == "horizontal" else height
        coverage = self.coverage_threshold if coverage_threshold is None else coverage_threshold
        hits: list[int] = []

        for i in range(limit):
            matched = 0
            for j in range(span):
                x, y = (j, i) if axis == "horizontal" else (i, j)
                r, g, b = pixels[x, y]
                if self._is_line_pixel(r, g, b):
                    matched += 1
            if matched / max(1, span) >= coverage:
                hits.append(i)

        if not hits:
            return []

        runs: list[tuple[int, int]] = []
        start = prev = hits[0]
        for value in hits[1:]:
            if value <= prev + 1:
                prev = value
                continue
            runs.append((start, prev))
            start = prev = value
        runs.append((start, prev))
        return runs

    def _bands_from_line_runs(
        self, size: int, runs: list[tuple[int, int]], min_size: int = 8
    ) -> list[dict]:
        """Convert separator runs to content bands between those separators."""
        bands: list[dict] = []
        cursor = 0

        for start, end in runs:
            if start <= min_size:
                cursor = max(cursor, end + 1)
                continue
            if end >= size - min_size:
                break
            if start - cursor >= min_size:
                bands.append({"index": len(bands), "start": cursor, "end": start})
            cursor = end + 1

        if size - cursor >= min_size:
            bands.append({"index": len(bands), "start": cursor, "end": size})
        return bands

    def _detect_physical_grid(self, image_path: str | Path) -> dict:
        img = Image.open(image_path).convert("RGB")
        h_runs = self._line_runs(img, "horizontal", coverage_threshold=0.72)
        v_runs = self._line_runs(img, "vertical", coverage_threshold=0.72)
        physical_rows = self._bands_from_line_runs(img.height, h_runs, min_size=12)
        physical_cols = self._bands_from_line_runs(img.width, v_runs, min_size=12)
        print(f"   📏 检测到 {len(physical_rows)} 个物理行: {physical_rows}")
        print(f"   📐 检测到 {len(physical_cols)} 个物理列: {physical_cols}")
        return {
            "image_size": [img.width, img.height],
            "horizontal_line_runs": [list(run) for run in h_runs],
            "vertical_line_runs": [list(run) for run in v_runs],
            "physical_rows": physical_rows,
            "physical_cols": physical_cols,
        }

    def _nearest_guided_boundaries(
        self, size: int, runs: list[tuple[int, int]], expected_count: int
    ) -> list[int]:
        """Pick separator positions nearest an expected equal grid."""
        if expected_count <= 0:
            return [0, size]

        centers = [((start + end) / 2, start, end) for start, end in runs]
        tolerance = max(12, size / max(1, expected_count) * 0.32)
        boundaries = [0]

        for index in range(1, expected_count):
            expected = size * index / expected_count
            candidates = [
                (abs(center - expected), start, end)
                for center, start, end in centers
                if abs(center - expected) <= tolerance
            ]
            if candidates:
                _, start, end = min(candidates, key=lambda item: item[0])
                boundaries.append(int(round((start + end + 1) / 2)))
            else:
                boundaries.append(int(round(expected)))

        boundaries.append(size)
        # Keep monotonic boundaries even if the image model drew noisy double lines.
        clean = [boundaries[0]]
        for value in boundaries[1:]:
            clean.append(max(clean[-1] + 1, min(size, value)))
        clean[-1] = size
        return clean

    def _bands_from_boundaries(self, boundaries: list[int]) -> list[dict]:
        bands: list[dict] = []
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            bands.append({"index": index, "start": int(start), "end": int(end)})
        return bands

    def _action_bands(
        self, physical_rows: list[dict], action_row_count: int, image_height: int
    ) -> list[dict]:
        """Group physical rows into the conceptual action rows from ASF metadata."""
        if action_row_count <= 0:
            return []
        if not physical_rows:
            step = image_height / action_row_count
            return [
                {
                    "index": i,
                    "start": round(i * step),
                    "end": round((i + 1) * step),
                    "physical_row_start": None,
                    "physical_row_end": None,
                }
                for i in range(action_row_count)
            ]

        row_count = len(physical_rows)
        groups: list[tuple[int, int]] = []
        if row_count == action_row_count:
            groups = [(i, i) for i in range(row_count)]
        elif action_row_count == 1:
            groups = [(0, row_count - 1)]
        elif row_count > action_row_count:
            groups.append((0, 0))
            middle_rows = row_count - 2
            middle_actions = action_row_count - 2
            for i in range(middle_actions):
                start = 1 + round(i * middle_rows / middle_actions)
                end = 1 + round((i + 1) * middle_rows / middle_actions) - 1
                groups.append((start, max(start, end)))
            groups.append((row_count - 1, row_count - 1))
        else:
            step = image_height / action_row_count
            return [
                {
                    "index": i,
                    "start": round(i * step),
                    "end": round((i + 1) * step),
                    "physical_row_start": None,
                    "physical_row_end": None,
                }
                for i in range(action_row_count)
            ]

        bands: list[dict] = []
        for index, (start_idx, end_idx) in enumerate(groups):
            start_row = physical_rows[start_idx]
            end_row = physical_rows[end_idx]
            bands.append(
                {
                    "index": index,
                    "start": int(start_row["start"]),
                    "end": int(end_row["end"]),
                    "physical_row_start": start_idx,
                    "physical_row_end": end_idx,
                }
            )
        return bands

    def _best_action_bands(self, grid: dict, action_row_count: int) -> list[dict]:
        width, height = grid["image_size"]
        physical_rows = grid.get("physical_rows", [])
        if len(physical_rows) == action_row_count:
            return self._action_bands(physical_rows, action_row_count, height)
        return self._guided_action_bands(grid, action_row_count)

    def _guided_action_bands(self, grid: dict, action_row_count: int) -> list[dict]:
        width, height = grid["image_size"]
        boundaries = self._nearest_guided_boundaries(
            height,
            [tuple(run) for run in grid.get("horizontal_line_runs", [])],
            action_row_count,
        )
        bands = self._bands_from_boundaries(boundaries)
        for band in bands:
            band["physical_row_start"] = band["index"]
            band["physical_row_end"] = band["index"]
        return bands

    def _detect_cols_for_crop(self, crop: Image.Image) -> list[dict]:
        # Vertical grid lines should run through most of the action region. Use a
        # stricter threshold than horizontal detection so dark sprite pixels are
        # not treated as separators.
        v_runs = self._line_runs(crop, "vertical", coverage_threshold=0.82)
        physical_cols = self._bands_from_line_runs(crop.width, v_runs, min_size=18)
        if physical_cols:
            return physical_cols
        boundaries = self._nearest_guided_boundaries(crop.width, v_runs, self.expected_cols)
        return self._bands_from_boundaries(boundaries)

    def _sprite_processor_path(self) -> Path | None:
        candidates = [
            Path(__file__).resolve().parent
            / "agent-sprite-forge"
            / "skills"
            / "generate2dsprite"
            / "scripts"
            / "generate2dsprite.py",
            Path(__file__).resolve().parent
            / "dependencies"
            / "agent-sprite-forge"
            / "skills"
            / "generate2dsprite"
            / "scripts"
            / "generate2dsprite.py",
            Path(__file__).resolve().parents[1]
            / "agent-sprite-forge"
            / "skills"
            / "generate2dsprite"
            / "scripts"
            / "generate2dsprite.py",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _build_assignments(self, actions: list[dict], action_bands: list[dict]) -> list[dict]:
        assignments: list[dict] = []
        max_row = len(action_bands)
        for item in actions:
            row_start = int(item.get("row_start", item["row"]))
            row_end = int(item.get("row_end", row_start))
            start_index = row_start - 1
            end_index = row_end - 1
            if start_index < 0 or end_index >= max_row or start_index > end_index:
                band = None
            else:
                start_band = action_bands[start_index]
                end_band = action_bands[end_index]
                band = {
                    "index": start_band["index"],
                    "start": int(start_band["start"]),
                    "end": int(end_band["end"]),
                    "physical_row_start": start_band.get("physical_row_start"),
                    "physical_row_end": end_band.get("physical_row_end"),
                    "row_start": row_start,
                    "row_end": row_end,
                }
            assignments.append(
                {
                    "action": item["id"],
                    "label": item["label"],
                    "order": item["index"],
                    "metadata_row": row_start,
                    "metadata_row_start": row_start,
                    "metadata_row_end": row_end,
                    "action_row_index": start_index,
                    "action_row_start_index": start_index,
                    "action_row_end_index": end_index,
                    "rows": max(1, row_end - row_start + 1),
                    "directions": item.get("directions", []),
                    "band": band,
                    "source": item.get("source", "metadata"),
                }
            )
        return assignments

    def split_frames(
        self,
        image_path: str,
        output_json: str,
        output_vis: str,
        metadata_json: str | None = None,
        actions_filter: list[str] | None = None,
        no_process: bool = False,
    ) -> dict:
        metadata_path = Path(metadata_json) if metadata_json else self._auto_metadata_path(image_path)
        if not metadata_path:
            raise FileNotFoundError("未找到 metadata JSON，请传入 --metadata-json")

        print(f"📂 处理图片: {image_path}")
        print(f"   🧾 metadata: {metadata_path}")
        metadata = self._read_metadata(metadata_path)
        actions = self._extract_actions(metadata)
        if actions_filter:
            wanted = set(actions_filter)
            actions = [item for item in actions if item["id"] in wanted]
        if not actions:
            raise ValueError("metadata 中没有可用的 actions[*].row / asf.actions[*].row")
        print(f"   🎬 动作行号: {json.dumps(actions, ensure_ascii=False)}")

        grid = self._detect_physical_grid(image_path)
        width, height = grid["image_size"]
        action_row_count = max(int(item.get("row_end", item["row"])) for item in actions)
        action_bands = self._best_action_bands(grid, action_row_count)
        assignments = self._build_assignments(actions, action_bands)

        result = {
            "version": "5.0",
            "image": str(image_path),
            "image_size": [width, height],
            "metadata_json": str(metadata_path),
            "grid": {
                **grid,
                "action_row_bands": action_bands,
            },
            "actions": actions,
            "assignments": assignments,
            "outputs": [],
        }
        result["outputs"] = self._crop_and_process_actions(
            image_path, result, output_json, no_process=no_process
        )
        self.export_json(result, output_json)
        self.visualize(image_path, result, output_vis)
        return result

    def _crop_and_process_actions(
        self, image_path: str, result: dict, output_json: str, no_process: bool = False
    ) -> list[dict]:
        output_root = Path(output_json).with_suffix("")
        output_root.mkdir(parents=True, exist_ok=True)
        processor = None if no_process else self._sprite_processor_path()
        assignments = list(result["assignments"])

        def handle_assignment(index: int, assignment: dict) -> tuple[int, dict]:
            band = assignment.get("band")
            if not band:
                return index, {**assignment, "status": "missing_band"}

            y1, y2 = int(band["start"]), int(band["end"])
            with Image.open(image_path) as opened:
                image = opened.convert("RGBA")
            rows = max(1, int(assignment.get("rows", 1)))

            action = str(assignment["action"])
            action_dir = output_root / action
            if action_dir.exists():
                shutil.rmtree(action_dir)
            action_dir.mkdir(parents=True, exist_ok=True)

            if rows > 1:
                full_crop = image.crop((0, y1, image.width, y2))
                full_crop_path = action_dir / "region.png"
                full_crop.save(full_crop_path)
                direction_outputs: list[dict] = []
                row_bands = result["grid"]["action_row_bands"]
                start_index = int(assignment.get("action_row_start_index", assignment.get("action_row_index", 0)))
                directions = assignment.get("directions") or [
                    {"id": DIRECTION_IDS[offset] if offset < len(DIRECTION_IDS) else f"direction_{offset + 1}"}
                    for offset in range(rows)
                ]

                for offset in range(rows):
                    direction = directions[offset] if offset < len(directions) else {"id": f"direction_{offset + 1}"}
                    direction_id = str(direction.get("id") or f"direction_{offset + 1}")
                    row_band = row_bands[start_index + offset]
                    dy1, dy2 = int(row_band["start"]), int(row_band["end"])
                    direction_crop = image.crop((0, dy1, image.width, dy2))
                    col_bands = self._detect_cols_for_crop(direction_crop)
                    cols = max(1, len(col_bands))
                    direction_dir = action_dir / direction_id
                    direction_dir.mkdir(parents=True, exist_ok=True)
                    crop_path = direction_dir / "region.png"
                    direction_crop.save(crop_path)

                    direction_item = {
                        "direction": direction_id,
                        "status": "cropped",
                        "crop": str(crop_path),
                        "crop_box": [0, dy1, image.width, dy2],
                        "rows": 1,
                        "cols": cols,
                        "col_bands": col_bands,
                        "processed_dir": None,
                    }

                    if processor:
                        processed_dir = direction_dir / "processed"
                        command = [
                            sys.executable,
                            str(processor),
                            "process",
                            "--input",
                            str(crop_path),
                            "--target",
                            "asset",
                            "--action",
                            f"{action}_{direction_id}",
                            "--rows",
                            "1",
                            "--cols",
                            str(cols),
                            "--label-prefix",
                            f"{action}_{direction_id}",
                            "--output-dir",
                            str(processed_dir),
                            "--shared-scale",
                            "--preserve-cell-coords",
                            "--preserve-frame-size",
                        ]
                        subprocess.run(command, check=True)
                        direction_item["status"] = "processed"
                        direction_item["processed_dir"] = str(processed_dir)

                    direction_outputs.append(direction_item)

                return index, {
                    **assignment,
                    "status": "processed" if processor else "cropped",
                    "crop": str(full_crop_path),
                    "crop_box": [0, y1, image.width, y2],
                    "rows": rows,
                    "cols": max((item["cols"] for item in direction_outputs), default=0),
                    "directions": directions,
                    "direction_outputs": direction_outputs,
                    "processed_dir": str(action_dir),
                }

            crop = image.crop((0, y1, image.width, y2))
            crop_box = [0, y1, image.width, y2]
            col_bands = self._detect_cols_for_crop(crop)
            cols = max(1, len(col_bands))
            crop_path = action_dir / "region.png"
            crop.save(crop_path)

            item = {
                **assignment,
                "status": "cropped",
                "crop": str(crop_path),
                "crop_box": crop_box,
                "rows": rows,
                "cols": cols,
                "col_bands": col_bands,
                "processed_dir": None,
            }

            if processor:
                processed_dir = action_dir / "processed"
                command = [
                    sys.executable,
                    str(processor),
                    "process",
                    "--input",
                    str(crop_path),
                    "--target",
                    "asset",
                    "--action",
                    action,
                    "--rows",
                    str(rows),
                    "--cols",
                    str(cols),
                    "--label-prefix",
                    action,
                    "--output-dir",
                    str(processed_dir),
                    "--shared-scale",
                    "--preserve-cell-coords",
                    "--preserve-frame-size",
                ]
                subprocess.run(command, check=True)
                item["status"] = "processed"
                item["processed_dir"] = str(processed_dir)

            return index, item

        if not assignments:
            return []

        max_workers = self.workers or min(len(assignments), max(1, min(4, os.cpu_count() or 1)))
        max_workers = max(1, min(max_workers, len(assignments)))
        if max_workers == 1:
            return [handle_assignment(index, assignment)[1] for index, assignment in enumerate(assignments)]

        print(f"   ⚙️ 异步拆图处理: {max_workers} workers")
        outputs: list[dict | None] = [None] * len(assignments)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(handle_assignment, index, assignment)
                for index, assignment in enumerate(assignments)
            ]
            for future in concurrent.futures.as_completed(futures):
                index, item = future.result()
                outputs[index] = item
        return [item for item in outputs if item is not None]

    def export_json(self, result: dict, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"💾 JSON: {output_path}")

    def visualize(self, image_path: str, result: dict, output_path: str) -> None:
        img = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        colors = [
            (255, 80, 80, 96),
            (80, 200, 255, 96),
            (120, 255, 120, 96),
            (255, 220, 80, 96),
            (220, 120, 255, 96),
        ]

        for index, obj in enumerate(result.get("assignments", [])):
            band = obj.get("band")
            if not band:
                continue
            x1, y1, x2, y2 = 0, int(band["start"]), img.width, int(band["end"])
            color = colors[index % len(colors)]
            draw.rectangle((x1, y1, x2, y2), fill=color, outline=(255, 255, 255, 220), width=3)
            tag = f"{obj.get('action')} row {obj.get('metadata_row')}"
            draw.text((12, max(8, y1 + 10)), tag, fill=(255, 255, 0, 255))

        result_img = Image.alpha_composite(img, overlay).convert("RGB")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result_img.save(output_path)
        print(f"🎨 可视化: {output_path}")


def _parse_actions(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_rgb(value: str) -> tuple[int, int, int]:
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
        if len(text) != 6:
            raise argparse.ArgumentTypeError("RGB hex color must be #RRGGBB")
        return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("RGB color must be #RRGGBB or r,g,b")
    rgb = tuple(int(part) for part in parts)
    if any(value < 0 or value > 255 for value in rgb):
        raise argparse.ArgumentTypeError("RGB channel must be in 0..255")
    return rgb  # type: ignore[return-value]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="读取 ASF action row metadata，并按图片网格线拆分动作帧")
    parser.add_argument("--image", help="输入的动作帧序列图路径", required=True)
    parser.add_argument("--metadata-json", help="raw.response.json 或 raw.manifest.json 路径")
    parser.add_argument("--actions", help="可选：只处理这些逗号分隔动作；默认读取 metadata 全部动作")
    parser.add_argument("--output-json", help="输出 JSON 路径", default="output/frames.json")
    parser.add_argument("--output-vis", help="输出可视化图片路径", default="output/frames_vis.png")
    parser.add_argument("--dark-threshold", type=int, default=110, help="网格线深色阈值")
    parser.add_argument("--coverage-threshold", type=float, default=0.25, help="判定整行/整列网格线的覆盖率")
    parser.add_argument("--line-rgb", type=_parse_rgb, default=(0, 0, 0), help="网格线 RGB，格式 #RRGGBB 或 r,g,b")
    parser.add_argument("--line-tolerance", type=int, default=24, help="RGB 三通道各自允许的误差")
    parser.add_argument("--expected-cols", type=int, default=8, help="每个动作行的期望列数")
    parser.add_argument("--workers", type=int, default=0, help="异步处理 worker 数；0 表示自动，1 表示顺序处理")
    parser.add_argument("--no-process", action="store_true", help="只裁剪动作区域，不调用 generate2dsprite.py")
    parser.add_argument("--api_key", help="兼容旧命令，当前不会使用", required=False)
    args = parser.parse_args()

    splitter = FrameSplitter(
        dark_threshold=args.dark_threshold,
        coverage_threshold=args.coverage_threshold,
        expected_cols=args.expected_cols,
        line_rgb=args.line_rgb,
        line_tolerance=args.line_tolerance,
        workers=args.workers,
    )
    splitter.split_frames(
        args.image,
        args.output_json,
        args.output_vis,
        metadata_json=args.metadata_json,
        actions_filter=_parse_actions(args.actions),
        no_process=args.no_process,
    )
