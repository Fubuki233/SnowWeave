"""
从视频中提取sprite帧
专门用于处理生成的动画视频，提取第2-3秒的帧

使用方法:
    python extract_sprite_frames.py <视频文件路径>
    
示例:
    python extract_sprite_frames.py temp_animation.mp4
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

def extract_frames_from_video_segment(video_path, start_time=2.0, end_time=3.0, max_frames=8):
    """
    从视频的指定时间段提取帧
    
    参数:
        video_path: 视频文件路径
        start_time: 开始时间（秒），设为0或-1表示从头开始
        end_time: 结束时间（秒），设为0或-1表示到结尾
        max_frames: 最大提取帧数（默认8）
    
    返回:
        提取的帧列表
    """
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
    
    # 计算时间段对应的帧范围
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # 限制不超过总帧数
    end_frame = min(end_frame, total_frames)
    
    # 计算可提取的帧数范围
    available_frames = end_frame - start_frame
    
    # 自动调整提取间隔
    # 如果请求的帧数高于 总帧数/8，则限制为 总帧数/8
    max_possible_frames = max(1, available_frames // 8)
    actual_max_frames = min(max_frames, max_possible_frames)
    
    # 计算提取间隔
    if actual_max_frames >= available_frames:
        frame_interval = 1
        actual_max_frames = available_frames
    else:
        frame_interval = available_frames // actual_max_frames
    
    print(f"  - 提取间隔: 每 {frame_interval} 帧提取一次")
    print(f"  - 目标帧数: {max_frames}, 实际提取: {actual_max_frames}")
    
    print(f"\n提取时间段: {start_time:.2f}s - {end_time:.2f}s")
    print(f"对应帧范围: {start_frame} - {end_frame}")
    
    # 生成要提取的帧索引
    frame_indices = list(range(start_frame, end_frame, frame_interval))[:actual_max_frames]
    print(f"将提取 {len(frame_indices)} 帧")
    
    # 提取帧
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 转换 BGR 到 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            print(f"  ✓ 提取帧 {idx} (时间: {idx/fps:.2f}s)")
        else:
            print(f"  × 无法读取帧 {idx}")
    
    cap.release()
    print(f"\n✓ 成功提取了 {len(frames)} 帧")
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
    
    print("✓ Sprite sheet 创建完成!")
    return sprite_sheet, frames

def save_individual_frames(frames, output_dir="extracted_frames"):
    """保存单独的帧图片"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n正在保存单独帧到 {output_dir}/ ...")
    
    for i, frame in enumerate(frames):
        output_path = os.path.join(output_dir, f"frame_{i:03d}.png")
        frame.save(output_path)
    
    print(f"✓ 保存了 {len(frames)} 个帧")

def main():
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python extract_sprite_frames.py <视频文件路径> [开始时间] [结束时间] [帧大小]")
        print("\n参数说明:")
        print("  视频文件路径: 必需，要处理的视频文件")
        print("  开始时间: 可选，默认 2.0 秒")
        print("  结束时间: 可选，默认 3.0 秒")
        print("  帧大小: 可选，默认 64")
        print("\n示例:")
        print("  python extract_sprite_frames.py temp_animation.mp4")
        print("  python extract_sprite_frames.py temp_animation.mp4 2.0 3.0")
        print("  python extract_sprite_frames.py temp_animation.mp4 2.0 3.0 128")
        sys.exit(1)
    
    video_path = sys.argv[1]
    start_time = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    end_time = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    frame_size = None  # 默认使用原图大小
    
    # 如果提供了帧大小参数，则使用指定大小
    if len(sys.argv) > 4:
        size = int(sys.argv[4])
        frame_size = (size, size)
    
    if not os.path.exists(video_path):
        print(f"× 错误: 找不到视频文件 {video_path}")
        sys.exit(1)
    
    try:
        # 1. 提取帧
        frames = extract_frames_from_video_segment(video_path, start_time, end_time)
        
        if not frames:
            print("× 错误: 没有提取到任何帧")
            sys.exit(1)
        
        # 2. 创建 sprite sheet
        sprite_sheet, output_frames = create_sprite_sheet(frames, frame_size=frame_size)
        
        # 3. 保存结果
        sprite_sheet_path = "extracted_sprite_sheet.png"
        sprite_sheet.save(sprite_sheet_path)
        print(f"\n✓ Sprite sheet 已保存: {sprite_sheet_path}")
        
        # 4. 保存单独的帧
        save_individual_frames(output_frames, output_dir="extracted_frames")
        
        print("\n" + "="*60)
        print("完成! 生成的文件:")
        print(f"  - Sprite sheet: {sprite_sheet_path}")
        print(f"  - 单独帧: extracted_frames/frame_*.png")
        print(f"  - 提取时间: {start_time}s - {end_time}s")
        print(f"  - 提取帧数: {len(frames)}")
        print("="*60)
        
    except Exception as e:
        print(f"\n× 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
