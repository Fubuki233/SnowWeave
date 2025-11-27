
import os
import sys
import cv2
import numpy as np
from PIL import Image

def detect_black_border_params(image, black_threshold=30):
    img_array = np.array(image.convert('RGB'))
    h, w = img_array.shape[:2]
    if not (1270 <= w <= 1290 and 710 <= h <= 770):
        return None
    col_brightness = np.mean(img_array, axis=(0, 2))
    non_black_cols = np.where(col_brightness > black_threshold)[0]
    
    if len(non_black_cols) == 0:
        return None
    
    left = non_black_cols[0]
    right = non_black_cols[-1] + 1
    if left == 0 and right == w:
        return None
    
    return (left, right)

def apply_crop(image, left, right):
    h, w = image.size[1], image.size[0]
    return image.crop((left, 0, right, h))

def extract_frames_from_video_segment(video_path, start_time=0.0, end_time=1.0, max_frames=0):
    print(f"正在处理视频: {video_path}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"视频信息:")
    print(f"  - 帧率: {fps} FPS")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 时长: {duration:.2f} 秒")
    
    # 判断是否解析整个视频
    parse_full_video = (start_time == 0 and end_time == 0) or start_time == -1 or end_time == -1
    
    if parse_full_video:
        start_time = 0
        end_time = duration
        print(f"  - 模式: 解析整个视频")
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    end_frame = min(end_frame, total_frames)
    
    available_frames = end_frame - start_frame
    
    target_fps = 16  
    frame_interval = max(1, int(fps / target_fps))  
    
    actual_max_frames = available_frames // frame_interval
    
    if max_frames > 0 and max_frames < actual_max_frames:
        actual_max_frames = max_frames
        frame_interval = available_frames // actual_max_frames
    
    print(f"  - 原始帧率: {fps} FPS")
    print(f"  - 提取帧率: {target_fps} FPS (每 {frame_interval} 帧提取一次)")
    print(f"  - 目标帧数: {max_frames if max_frames > 0 else '自动'}, 实际提取: {actual_max_frames}")
    
    print(f"\n提取时间段: {start_time:.2f}s - {end_time:.2f}s")
    print(f"对应帧范围: {start_frame} - {end_frame}")
    
    frame_indices = list(range(start_frame, end_frame, frame_interval))[:actual_max_frames]
    print(f"将提取 {len(frame_indices)} 帧")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("无法读取第一帧")
    
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    first_pil = Image.fromarray(first_frame_rgb)
    crop_params = detect_black_border_params(first_pil)
    
    if crop_params:
        left, right = crop_params
        print(f"  - 检测到左右黑边: 左{left}px, 右{first_pil.width - right}px")
        print(f"  - 裁剪后宽度: {right - left}px (原始: {first_pil.width}px)")
    else:
        print(f"  - 未检测到黑边,保持原始尺寸")
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            
            if crop_params:
                left, right = crop_params
                pil_frame = apply_crop(pil_frame, left, right)
            
            frames.append(pil_frame)
            print(f"   提取帧 {idx} (时间: {idx/fps:.2f}s)")
        else:
            print(f"   无法读取帧 {idx}")
    
    if frames:
        print(f"  - 最终帧尺寸: {frames[0].size}")
    
    cap.release()
    print(f"\n 成功提取了 {len(frames)} 帧")
    return frames

def create_sprite_sheet(frames, frame_size=None):
    """将帧组合成横向 sprite sheet"""
    if frame_size is None:
        # 使用原图大小
        if frames:
            frame_size = frames[0].size
            print(f"\n正在创建 sprite sheet (保持原图大小 {frame_size[0]}x{frame_size[1]})...")
        else:
            raise ValueError("没有帧可以处理")
    else:
        print(f"\n正在创建 sprite sheet (每帧 {frame_size[0]}x{frame_size[1]})...")
        # 调整每一帧的大小
        frames = [frame.resize(frame_size, Image.Resampling.LANCZOS) for frame in frames]
    
    # 创建 sprite sheet (横向排列)
    sheet_width = frame_size[0] * len(frames)
    sheet_height = frame_size[1]
    sprite_sheet = Image.new('RGBA', (sheet_width, sheet_height), (0, 0, 0, 0))
    
    # 粘贴每一帧
    for i, frame in enumerate(frames):
        x_offset = i * frame_size[0]
        # 转换为 RGBA 如果需要
        if frame.mode != 'RGBA':
            frame = frame.convert('RGBA')
        sprite_sheet.paste(frame, (x_offset, 0))
    
    print(" Sprite sheet 创建完成!")
    return sprite_sheet, frames

def save_individual_frames(frames, output_dir="extracted_frames"):
    """保存单独的帧图片"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n正在保存单独帧到 {output_dir}/ ...")
    
    for i, frame in enumerate(frames):
        output_path = os.path.join(output_dir, f"frame_{i:03d}.png")
        frame.save(output_path)
    
    print(f" 保存了 {len(frames)} 个帧")

