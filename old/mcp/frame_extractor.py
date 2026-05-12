"""
Video Frame Extractor (MCP Standalone)
视频帧提取工具（MCP 独立版）

从视频中提取帧并保存，支持背景移除（抠图）
"""

import os
import cv2
import numpy as np
from typing import List, Optional, Tuple
from PIL import Image
from scipy import ndimage


def detect_background_color(image: Image.Image, border_skip: int = 20) -> Tuple[int, int, int]:
    """
    智能检测背景色，跳过可能的边框区域
    
    Args:
        image: 输入图片
        border_skip: 跳过边框的像素数
    
    Returns:
        检测到的背景色 RGB
    """
    arr = np.array(image.convert('RGB'))
    h, w = arr.shape[:2]
    
    # 跳过边框，在内部区域采样
    inner_start = border_skip
    inner_end_h = h - border_skip
    inner_end_w = w - border_skip
    
    if inner_end_h <= inner_start or inner_end_w <= inner_start:
        inner_start = 5
        inner_end_h = h - 5
        inner_end_w = w - 5
    
    # 采样四个角的内部区域
    sample_size = 15
    corners = [
        arr[inner_start:inner_start+sample_size, inner_start:inner_start+sample_size],  # 左上内
        arr[inner_start:inner_start+sample_size, inner_end_w-sample_size:inner_end_w],  # 右上内
        arr[inner_end_h-sample_size:inner_end_h, inner_start:inner_start+sample_size],  # 左下内
        arr[inner_end_h-sample_size:inner_end_h, inner_end_w-sample_size:inner_end_w],  # 右下内
    ]
    
    # 取四角的平均色
    corner_colors = [c.mean(axis=(0, 1)) for c in corners]
    bg_color = tuple(int(x) for x in np.mean(corner_colors, axis=0))
    
    return bg_color


def remove_background_smart(
    image: Image.Image,
    tolerance: int = 50,
    edge_shrink: int = 3,
    border_skip: int = 20
) -> Image.Image:
    """
    智能背景移除：自动检测背景色，处理边框和渐变背景
    
    Args:
        image: 输入图片
        tolerance: 颜色容差（建议 40-60 用于渐变背景）
        edge_shrink: 边缘内缩像素数
        border_skip: 跳过边框像素数（用于检测背景色）
    
    Returns:
        带透明通道的 RGBA 图片
    """
    # 转换为 RGBA
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # 智能检测背景色
    bg_color = detect_background_color(image, border_skip)
    print(f"[SmartBG] Detected background color: RGB{bg_color}")
    
    # 提取 RGB 通道
    rgb = img_array[:, :, :3].astype(np.int32)
    bg = np.array(bg_color, dtype=np.int32)
    gray = rgb.mean(axis=2)
    
    # 计算颜色差异
    diff = np.sqrt(np.sum((rgb - bg) ** 2, axis=2))
    
    # 创建背景掩码 - 使用基础容差
    bg_mask = (diff < tolerance).astype(np.uint8)
    
    # 判断背景是否为浅色（白色/灰色）
    # 只有浅色背景才需要检测暗边框
    bg_brightness = sum(bg_color) / 3
    is_light_background = bg_brightness > 200  # 背景亮度 > 200 才算浅色
    
    if is_light_background:
        # 浅色背景：同时检测暗边框（视频边缘可能有黑色边框）
        dark_mask = (gray < 40).astype(np.uint8)  # 非常暗的区域
        combined_bg_mask = np.maximum(bg_mask, dark_mask)
        print(f"[SmartBG] Light background detected, enabling dark border detection")
    else:
        # 彩色背景（如绿幕）：只用颜色匹配，不检测暗色
        combined_bg_mask = bg_mask
        print(f"[SmartBG] Colored background detected, using color match only")
    
    # 使用形态学膨胀连接断开的背景区域
    kernel = np.ones((3, 3), np.uint8)  # 减小kernel避免过度膨胀
    dilated_mask = cv2.dilate(combined_bg_mask, kernel, iterations=1)
    
    # 使用连通区域分析
    labeled, num_features = ndimage.label(dilated_mask)
    
    # 找到与边缘连通的标签（包括外边缘和内边缘）
    edge_labels = set()
    # 外边缘
    edge_labels.update(labeled[0, :].flatten())
    edge_labels.update(labeled[h-1, :].flatten())
    edge_labels.update(labeled[:, 0].flatten())
    edge_labels.update(labeled[:, w-1].flatten())
    # 内边缘（跳过暗边框）
    inner = border_skip
    if inner < h - inner and inner < w - inner:
        edge_labels.update(labeled[inner, inner:w-inner].flatten())
        edge_labels.update(labeled[h-1-inner, inner:w-inner].flatten())
        edge_labels.update(labeled[inner:h-inner, inner].flatten())
        edge_labels.update(labeled[inner:h-inner, w-1-inner].flatten())
    edge_labels.discard(0)
    
    # 创建膨胀后的连通区域掩码
    dilated_edge_mask = np.zeros((h, w), dtype=np.uint8)
    for label in edge_labels:
        dilated_edge_mask[labeled == label] = 1
    
    # 只保留原始掩码中与边缘连通的部分
    final_bg_mask = combined_bg_mask & dilated_edge_mask
    
    # 边缘内缩（腐蚀前景）
    if edge_shrink > 0:
        foreground = (1 - final_bg_mask).astype(np.uint8)
        kernel = np.ones((edge_shrink * 2 + 1, edge_shrink * 2 + 1), np.uint8)
        foreground_eroded = cv2.erode(foreground, kernel, iterations=1)
        final_bg_mask = 1 - foreground_eroded
    
    # 边缘平滑
    foreground = (1 - final_bg_mask).astype(np.uint8) * 255
    foreground = cv2.GaussianBlur(foreground, (3, 3), 0)
    
    # 创建 alpha 通道
    alpha = foreground.astype(np.uint8)
    img_array[:, :, 3] = alpha
    
    # 统计
    transparent_pixels = np.sum(alpha == 0)
    total_pixels = alpha.size
    print(f"[SmartBG] Removed {transparent_pixels} pixels ({transparent_pixels/total_pixels*100:.1f}% of image)")
    
    return Image.fromarray(img_array, 'RGBA')


def remove_background_floodfill(
    image: Image.Image,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    tolerance: int = 30,
    edge_shrink: int = 5,
    edge_smooth: int = 1
) -> Image.Image:
    """
    使用 flood fill 从边缘移除背景，保留角色内部的白色区域
    
    Args:
        image: 输入图片
        bg_color: 背景色 RGB 值，默认白色
        tolerance: 颜色容差
        edge_shrink: 边缘内缩像素数，用于去除边框
        edge_smooth: 边缘平滑程度
    
    Returns:
        带透明通道的 RGBA 图片
    """
    # 转换为 RGBA
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # 提取 RGB 通道
    rgb = img_array[:, :, :3].astype(np.int32)
    
    # 计算每个像素与背景色的差异
    bg = np.array(bg_color, dtype=np.int32)
    diff = np.sqrt(np.sum((rgb - bg) ** 2, axis=2))
    
    # 创建背景掩码（接近背景色的像素）
    bg_mask = (diff < tolerance).astype(np.uint8)
    
    # 创建边缘种子掩码（从图像边缘开始 flood fill）
    seed_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 设置四条边为种子点
    seed_mask[0, :] = 1      # 上边
    seed_mask[h-1, :] = 1    # 下边
    seed_mask[:, 0] = 1      # 左边
    seed_mask[:, w-1] = 1    # 右边
    
    # 只保留边缘上属于背景的种子点
    seed_mask = seed_mask & bg_mask
    
    # 使用 flood fill 找到从边缘连通的所有背景区域
    # 这样角色内部的白色不会被移除
    labeled, num_features = ndimage.label(bg_mask)
    
    # 找到与边缘连通的标签
    edge_labels = set()
    edge_labels.update(labeled[0, :].flatten())      # 上边
    edge_labels.update(labeled[h-1, :].flatten())    # 下边
    edge_labels.update(labeled[:, 0].flatten())      # 左边
    edge_labels.update(labeled[:, w-1].flatten())    # 右边
    edge_labels.discard(0)  # 移除非背景标签
    
    # 创建最终的背景掩码（只包含与边缘连通的背景区域）
    final_bg_mask = np.zeros((h, w), dtype=np.uint8)
    for label in edge_labels:
        final_bg_mask[labeled == label] = 1
    
    # 边缘内缩处理（腐蚀操作去除边框）
    if edge_shrink > 0:
        # 对前景区域（非背景）进行腐蚀，相当于内缩
        foreground = (1 - final_bg_mask).astype(np.uint8)
        kernel = np.ones((edge_shrink * 2 + 1, edge_shrink * 2 + 1), np.uint8)
        foreground_eroded = cv2.erode(foreground, kernel, iterations=1)
        
        # 更新背景掩码
        final_bg_mask = 1 - foreground_eroded
    
    # 边缘平滑处理
    if edge_smooth > 0:
        kernel_size = edge_smooth * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # 对前景进行形态学闭操作，平滑边缘
        foreground = (1 - final_bg_mask).astype(np.uint8) * 255
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
        foreground = cv2.GaussianBlur(foreground, (3, 3), 0)
        final_bg_mask = (foreground < 128).astype(np.uint8)
    
    # 创建 alpha 通道
    alpha = ((1 - final_bg_mask) * 255).astype(np.uint8)
    
    # 合并 alpha 通道
    img_array[:, :, 3] = alpha
    
    return Image.fromarray(img_array, 'RGBA')


def remove_background(
    image: Image.Image,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    tolerance: int = 30,
    edge_smooth: int = 1
) -> Image.Image:
    """
    移除背景色，将其转为透明（抠图）- 简单版本，会移除所有匹配颜色
    
    Args:
        image: 输入图片
        bg_color: 背景色 RGB 值，默认白色
        tolerance: 颜色容差，用于处理接近背景色的像素
        edge_smooth: 边缘平滑程度（0-3）
    
    Returns:
        带透明通道的 RGBA 图片
    """
    # 转换为 RGBA
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # 转为 numpy 数组
    img_array = np.array(image)
    
    # 提取 RGB 通道
    rgb = img_array[:, :, :3].astype(np.int32)
    
    # 计算与背景色的差异
    bg = np.array(bg_color, dtype=np.int32)
    diff = np.sqrt(np.sum((rgb - bg) ** 2, axis=2))
    
    # 创建 alpha 通道
    # 差异小于容差的像素设为透明
    alpha = np.where(diff < tolerance, 0, 255).astype(np.uint8)
    
    # 边缘平滑处理
    if edge_smooth > 0:
        # 使用形态学操作平滑边缘
        kernel_size = edge_smooth * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 先膨胀再腐蚀，去除锯齿
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
        
        # 轻微模糊边缘
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    
    # 合并 alpha 通道
    img_array[:, :, 3] = alpha
    
    return Image.fromarray(img_array, 'RGBA')


def remove_background_advanced(
    image: Image.Image,
    method: str = "auto",
    tolerance: int = 50,
    edge_shrink: int = 3
) -> Image.Image:
    """
    高级背景移除，使用智能检测和 flood fill
    
    Args:
        image: 输入图片
        method: 移除方法
            - "white": 移除白色背景（自动检测实际背景色）
            - "green": 移除绿幕背景（自动检测实际背景色）
            - "auto": 自动检测背景色（推荐，处理渐变和边框）
            - "smart": 智能模式（同 auto，更强的边框处理）
        tolerance: 颜色容差（auto/smart 模式建议 50）
        edge_shrink: 边缘内缩像素数（去除边框）
    
    Returns:
        带透明通道的 RGBA 图片
    
    Note:
        所有方法现在都使用智能检测，method 参数主要用于日志记录
        实际背景色会从图片四角自动检测
    """
    # 所有方法都使用智能背景移除，自动检测实际背景色
    # 这样无论生成的是白色还是绿色背景，都能正确处理
    return remove_background_smart(image, tolerance, edge_shrink)


def extract_frames_from_video(
    video_path: str,
    start_time: float = 0.0,
    end_time: float = 0.0,
    max_frames: int = 24,
    target_fps: int = 16
) -> List[Image.Image]:
    """
    从视频中提取帧
    
    Args:
        video_path: 视频文件路径
        start_time: 开始时间（秒），0表示开头
        end_time: 结束时间（秒），0或-1表示结尾
        max_frames: 最大提取帧数
        target_fps: 目标帧率
    
    Returns:
        PIL Image 列表
    """
    print(f"[FrameExtractor] Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"[FrameExtractor] Video info:")
    print(f"  - FPS: {fps}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Duration: {duration:.2f}s")
    
    # 处理时间范围
    if end_time <= 0 or end_time == -1:
        end_time = duration
    if start_time < 0:
        start_time = 0
    end_time = min(end_time, duration)
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    end_frame = min(end_frame, total_frames)
    
    available_frames = end_frame - start_frame
    
    # 计算帧间隔
    frame_interval = max(1, int(fps / target_fps))
    actual_max_frames = available_frames // frame_interval
    
    if max_frames > 0 and max_frames < actual_max_frames:
        actual_max_frames = max_frames
        frame_interval = available_frames // actual_max_frames if actual_max_frames > 0 else 1
    
    print(f"[FrameExtractor] Extraction settings:")
    print(f"  - Time range: {start_time:.2f}s - {end_time:.2f}s")
    print(f"  - Frame range: {start_frame} - {end_frame}")
    print(f"  - Frame interval: {frame_interval}")
    print(f"  - Target frames: {actual_max_frames}")
    
    frame_indices = list(range(start_frame, end_frame, frame_interval))[:actual_max_frames]
    
    if not frame_indices:
        cap.release()
        print("[FrameExtractor] No frames to extract!")
        return []
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(pil_frame)
        else:
            print(f"[FrameExtractor] Failed to read frame {idx}")
    
    cap.release()
    print(f"[FrameExtractor] Extracted {len(frames)} frames")
    
    return frames


def save_frames(
    frames: List[Image.Image],
    output_dir: str,
    prefix: str = "frame",
    start_index: int = 1,
    remove_bg: bool = False,
    bg_method: str = "white",
    bg_tolerance: int = 30,
    bg_edge_shrink: int = 5
) -> List[str]:
    """
    保存帧到指定目录，可选抠图
    
    Args:
        frames: PIL Image 列表
        output_dir: 输出目录
        prefix: 文件名前缀
        start_index: 起始索引
        remove_bg: 是否移除背景（抠图）
        bg_method: 背景移除方法 ("white", "green", "auto")
        bg_tolerance: 颜色容差
        bg_edge_shrink: 边缘内缩像素数（去除边框）
    
    Returns:
        保存的文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if remove_bg:
        print(f"[FrameExtractor] Background removal enabled (method: {bg_method}, tolerance: {bg_tolerance}, edge_shrink: {bg_edge_shrink})")
    
    saved_paths = []
    for i, frame in enumerate(frames):
        # 抠图处理
        if remove_bg:
            frame = remove_background_advanced(frame, method=bg_method, tolerance=bg_tolerance, edge_shrink=bg_edge_shrink)
        
        filename = f"{prefix}_{start_index + i:04d}.png"
        filepath = os.path.join(output_dir, filename)
        frame.save(filepath, "PNG")
        saved_paths.append(filepath)
    
    print(f"[FrameExtractor] Saved {len(saved_paths)} frames to {output_dir}")
    return saved_paths


def create_sprite_sheet(
    frames: List[Image.Image],
    columns: int = 0
) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    创建 Sprite Sheet
    
    Args:
        frames: PIL Image 列表
        columns: 列数，0表示自动计算
    
    Returns:
        (sprite_sheet, (frame_width, frame_height))
    """
    if not frames:
        raise ValueError("No frames provided")
    
    frame_width, frame_height = frames[0].size
    n_frames = len(frames)
    
    # 自动计算列数
    if columns <= 0:
        import math
        columns = int(math.ceil(math.sqrt(n_frames)))
    
    rows = int(math.ceil(n_frames / columns))
    
    # 创建画布
    sheet_width = columns * frame_width
    sheet_height = rows * frame_height
    
    # 使用 RGBA 模式支持透明
    sprite_sheet = Image.new('RGBA', (sheet_width, sheet_height), (0, 0, 0, 0))
    
    for i, frame in enumerate(frames):
        row = i // columns
        col = i % columns
        x = col * frame_width
        y = row * frame_height
        
        # 确保帧是 RGBA 模式
        if frame.mode != 'RGBA':
            frame = frame.convert('RGBA')
        
        sprite_sheet.paste(frame, (x, y))
    
    print(f"[SpriteSheet] Created {sheet_width}x{sheet_height} sheet ({columns}x{rows})")
    
    return sprite_sheet, (frame_width, frame_height)
