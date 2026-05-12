"""
motion_sheet.py
---------------
将一个动作目录（包含 up/left/down/right 四个方向子目录）拼合为
一张 "方向 × 帧序列" 的 Sprite Sheet。

布局规则
--------
- 行数 = 4（上 → 左 → 下 → 右）
- 列数 = 该动作在各方向中帧数的最大值 N
- 每列宽 = 单帧宽度 M，每行高 = 单帧高度 H
- 画布底色 = 白色，对齐方式 = 左对齐，不足的帧留空白

作为模块使用
-------------
    from motion_sheet import build_motion_sheet, process_all

    # 处理单个动作，返回 (PIL.Image, action_name)
    img, name = build_motion_sheet("motions/standard/1h_slash")

    # 处理整个 standard 目录下的所有动作并保存图片
    results = process_all("motions/standard", "output/sheets")

命令行使用
----------
    # 处理单个动作目录
    python motion_sheet.py motions/standard/1h_slash output/sheets

    # 处理整个标准动作库（自动遍历所有子动作）
    python motion_sheet.py motions/standard output/sheets
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image

# 方向的固定顺序：上 → 左 → 下 → 右
# 注意：源文件中 left 与 down 文件夹内容互换，读取时对调以修正
DIRECTION_ORDER = ["up", "down", "left", "right"]


# ---------------------------------------------------------------------------
# 核心：处理单个动作目录
# ---------------------------------------------------------------------------

def _load_frames(direction_dir: Path) -> List[Image.Image]:
    """按帧编号升序加载一个方向下的所有帧（支持任意数量）。"""
    files = [
        f for f in direction_dir.iterdir()
        if f.suffix.lower() == ".png" and f.stem.isdigit()
    ]
    files.sort(key=lambda f: int(f.stem))
    return [Image.open(f).convert("RGBA") for f in files]


def build_motion_sheet(action_dir: str | Path) -> Tuple[Image.Image, str]:
    """
    将一个动作目录拼合成 Sprite Sheet。

    Parameters
    ----------
    action_dir : str | Path
        动作根目录，其下应包含 up / left / down / right 子文件夹。

    Returns
    -------
    (PIL.Image.Image, str)
        拼合后的图像 和 动作名称（目录名）。
    """
    action_path = Path(action_dir)
    action_name = action_path.name

    # 按固定方向顺序加载帧列表（目录不存在或为空则跳过，不占行）
    rows_frames: List[List[Image.Image]] = []
    for direction in DIRECTION_ORDER:
        dir_path = action_path / direction
        if dir_path.is_dir():
            frames = _load_frames(dir_path)
            if frames:
                rows_frames.append(frames)

    # 推断单帧尺寸（取第一张有效帧）
    frame_w, frame_h = 64, 64
    for frames in rows_frames:
        if frames:
            frame_w, frame_h = frames[0].size
            break

    # 列数 = 所有方向中帧数的最大值；行数 = 实际存在的方向数
    max_frames = max((len(f) for f in rows_frames), default=1)
    actual_rows = len(rows_frames)

    # 创建白色 RGBA 画布
    canvas_w = max_frames * frame_w
    canvas_h = actual_rows * frame_h
    sheet = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    # 将各帧粘贴到对应位置
    for row_idx, frames in enumerate(rows_frames):
        for col_idx, frame in enumerate(frames):
            x = col_idx * frame_w
            y = row_idx * frame_h
            # 若帧带透明通道，使用自身 alpha 作为蒙版
            sheet.paste(frame, (x, y), frame if frame.mode == "RGBA" else None)

    return sheet, action_name


# ---------------------------------------------------------------------------
# 批量处理整个 standard 目录
# ---------------------------------------------------------------------------

def get_all_actions(standard_dir: str | Path) -> List[str]:
    """返回 standard_dir 下所有动作目录的名称列表（按字母排序）。"""
    standard_path = Path(standard_dir)
    actions = sorted(
        d.name for d in standard_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    return actions


def process_all(
    standard_dir: str | Path,
    output_dir: str | Path,
) -> List[Tuple[Image.Image, str]]:
    """
    遍历 standard_dir 下的所有动作，逐一生成 Sprite Sheet 并保存。

    Parameters
    ----------
    standard_dir : str | Path
        包含多个动作子目录的标准动作库目录。
    output_dir : str | Path
        输出目录，保存 {action_name}.png。

    Returns
    -------
    list of (PIL.Image.Image, str)
        所有动作的 (图像, 动作名) 元组列表。
    """
    standard_path = Path(standard_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    actions = get_all_actions(standard_path)
    print(f"[motion_sheet] 发现 {len(actions)} 个动作：{', '.join(actions)}")

    results = []
    for action_name in actions:
        action_dir = standard_path / action_name
        sheet, name = build_motion_sheet(action_dir)
        save_path = output_path / f"{name}.png"
        sheet.save(save_path)
        print(f"  ✓ {name} → {save_path}  ({sheet.width}×{sheet.height})")
        results.append((sheet, name))

    print(f"[motion_sheet] 全部完成，已保存到 {output_path}")
    return results


# ---------------------------------------------------------------------------
# 判断目录是"单个动作"还是"标准动作库"
# ---------------------------------------------------------------------------

def _is_action_dir(path: Path) -> bool:
    """若该目录直接包含 up/left/down/right 中至少一个子目录，视为单动作目录。"""
    return any((path / d).is_dir() for d in DIRECTION_ORDER)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 3:
        print("用法:")
        print("  python motion_sheet.py <动作目录或标准动作库目录> <输出目录>")
        print()
        print("示例:")
        print("  python motion_sheet.py motions/standard/1h_slash output/sheets")
        print("  python motion_sheet.py motions/standard output/sheets")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"[错误] 目录不存在：{input_path}")
        sys.exit(1)

    if _is_action_dir(input_path):
        # 单个动作目录
        output_path.mkdir(parents=True, exist_ok=True)
        sheet, name = build_motion_sheet(input_path)
        save_path = output_path / f"{name}.png"
        sheet.save(save_path)
        print(f"✓ {name} → {save_path}  ({sheet.width}×{sheet.height})")
    else:
        # 标准动作库目录，批量处理
        process_all(input_path, output_path)


if __name__ == "__main__":
    main()
