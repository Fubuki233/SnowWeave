"""
SnowWeave Frame Processor Module
帧处理模块 - 处理视频帧提取、背景去除、Sprite Sheet 创建
"""

import os
from datetime import datetime
from typing import List, Optional, Tuple, Union
from PIL import Image

from lib.extract_sprite_frames import (
    extract_frames_from_video_segment,
    create_sprite_sheet,
    save_individual_frames,
    process_frames_to_target_size
)
from lib.remove_background import (
    process_directory,
    process_image
)

from .config import OUTPUT_DIR, t, get_current_language


class FrameProcessor:
    """
    帧处理器
    处理视频帧提取、背景去除、Sprite Sheet 创建等操作
    """
    
    def extract_frames(
        self, 
        video_path: str, 
        start_time: float, 
        end_time: float, 
        max_frames: int
    ) -> Tuple[Optional[str], Optional[List[Image.Image]], str]:
        """
        从视频提取帧
        
        Args:
            video_path: 视频文件路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            max_frames: 最大帧数
        
        Returns:
            (sprite_sheet_path, preview_images, status_message)
        """
        if video_path is None:
            return None, None, t("upload_video")
        
        try:
            # 提取帧
            frames = extract_frames_from_video_segment(
                video_path,
                float(start_time),
                float(end_time),
                int(max_frames)
            )
            
            if not frames:
                return None, None, "[ERROR] " + t("no_frames")
            
            # 保存帧
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(OUTPUT_DIR, f"frames_{timestamp}")
            frames_dir = os.path.join(output_dir, "frames")
            save_individual_frames(frames, output_dir=frames_dir)
            
            # 创建 sprite sheet
            sprite_sheet, _ = create_sprite_sheet(frames, frame_size=None)
            sheet_path = os.path.join(output_dir, "sprite_sheet.png")
            sprite_sheet.save(sheet_path)
            
            # 创建预览
            preview_images = [frame for frame in frames[:8]]
            
            # 生成摘要
            lang = get_current_language()
            if lang == "zh":
                summary = f"[OK] 提取完成!\n共 {len(frames)} 帧\nSprite Sheet: {sheet_path}\n帧目录: {frames_dir}"
            else:
                summary = f"[OK] Extraction complete!\nTotal {len(frames)} frames\nSprite Sheet: {sheet_path}\nFrames directory: {frames_dir}"
            
            return sheet_path, preview_images, summary
            
        except Exception as e:
            return None, None, f"[ERROR] {t('error')}: {str(e)}"
    
    def remove_background(
        self,
        uploaded_files: Union[str, List[str]],
        tolerance: int = 180,
        auto_crop: bool = False,
        crop_padding: int = 0,
        progress_callback=None
    ) -> Tuple[Optional[str], Optional[List[Image.Image]], str]:
        """
        去除图片背景
        
        Args:
            uploaded_files: 上传的文件路径（单个或列表）
            tolerance: 颜色容差
            auto_crop: 是否自动裁剪
            crop_padding: 裁剪边距
            progress_callback: 进度回调函数 (progress_value, description)
        
        Returns:
            (sprite_sheet_path, processed_images, status_message)
        """
        if uploaded_files is None or (isinstance(uploaded_files, list) and len(uploaded_files) == 0):
            return None, None, t("upload_image")
        
        try:
            if progress_callback:
                progress_callback(0, t("processing_images"))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(OUTPUT_DIR, f"nobg_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 处理文件
            if isinstance(uploaded_files, list) and len(uploaded_files) > 1:
                return self._process_multiple_files(
                    uploaded_files, output_dir, tolerance, auto_crop, crop_padding, progress_callback
                )
            else:
                file_path = uploaded_files[0] if isinstance(uploaded_files, list) else uploaded_files
                return self._process_single_file(
                    file_path, output_dir, tolerance, auto_crop, crop_padding, progress_callback
                )
            
        except Exception as e:
            return None, None, f"[ERROR] {t('error')}: {str(e)}"
    
    def _process_multiple_files(
        self,
        files: List[str],
        output_dir: str,
        tolerance: int,
        auto_crop: bool,
        crop_padding: int,
        progress_callback
    ) -> Tuple[Optional[str], Optional[List[Image.Image]], str]:
        """处理多个文件"""
        if progress_callback:
            progress_callback(0.2, t("processing_n_images", len(files)))
        
        nobg_dir = os.path.join(output_dir, "frames")
        os.makedirs(nobg_dir, exist_ok=True)
        
        processed_images = []
        lang = get_current_language()
        
        for i, file_path in enumerate(files):
            progress_desc = f"{i+1}/{len(files)}"
            if progress_callback:
                desc = f"处理 {progress_desc}..." if lang == "zh" else f"Processing {progress_desc}..."
                progress_callback(0.2 + 0.6 * (i / len(files)), desc)
            
            filename = os.path.basename(file_path)
            output_path = os.path.join(nobg_dir, filename)
            
            process_image(
                file_path,
                output_path=output_path,
                tolerance=int(tolerance),
                auto_crop=auto_crop,
                crop_padding=int(crop_padding)
            )
            
            processed_images.append(Image.open(output_path))
        
        if progress_callback:
            progress_callback(0.8, t("creating_sprite"))
        
        # 创建 sprite sheet
        sheet_path = None
        if processed_images:
            final_sheet, _ = create_sprite_sheet(processed_images, frame_size=None)
            sheet_path = os.path.join(output_dir, "sprite_sheet.png")
            final_sheet.save(sheet_path)
        
        preview_images = processed_images[:8]
        
        if progress_callback:
            progress_callback(1.0, t("complete"))
        
        if lang == "zh":
            summary = f"[OK] 背景去除完成!\n共处理 {len(files)} 张图片\nSprite Sheet: {sheet_path}\n帧目录: {nobg_dir}"
        else:
            summary = f"[OK] Background removal complete!\nProcessed {len(files)} images\nSprite Sheet: {sheet_path}\nFrames directory: {nobg_dir}"
        
        return sheet_path, preview_images, summary
    
    def _process_single_file(
        self,
        file_path: str,
        output_dir: str,
        tolerance: int,
        auto_crop: bool,
        crop_padding: int,
        progress_callback
    ) -> Tuple[Optional[str], Optional[List[Image.Image]], str]:
        """处理单个文件"""
        if progress_callback:
            progress_callback(0.3, t("processing_single"))
        
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        
        process_image(
            file_path,
            output_path=output_path,
            tolerance=int(tolerance),
            auto_crop=auto_crop,
            crop_padding=int(crop_padding)
        )
        
        if progress_callback:
            progress_callback(1.0, t("complete"))
        
        result_img = Image.open(output_path)
        lang = get_current_language()
        
        if lang == "zh":
            summary = f"[OK] 背景去除完成!\n保存路径: {output_path}"
        else:
            summary = f"[OK] Background removal complete!\nSave path: {output_path}"
        
        return output_path, [result_img], summary
    
    @staticmethod
    def resize_frames_to_width(frames_dir: str, target_width: int = 128) -> List[Image.Image]:
        """
        将目录中的所有帧等比缩放到指定宽度
        
        Args:
            frames_dir: 帧文件目录
            target_width: 目标宽度
        
        Returns:
            缩放后的图片列表
        """
        resized_frames = []
        
        frame_files = sorted([
            f for f in os.listdir(frames_dir) 
            if f.lower().endswith('.png')
        ])
        
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            img = Image.open(frame_path)
            
            original_width, original_height = img.size
            scale_ratio = target_width / original_width
            target_height = int(original_height * scale_ratio)
            
            resized_img = img.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS
            )
            resized_frames.append(resized_img)
        
        return resized_frames


# 便捷函数
def extract_frames(video_path: str, start_time: float, end_time: float, max_frames: int):
    """便捷函数：从视频提取帧"""
    processor = FrameProcessor()
    return processor.extract_frames(video_path, start_time, end_time, max_frames)


def remove_background(uploaded_files, tolerance: int = 180, auto_crop: bool = False, crop_padding: int = 0, progress_callback=None):
    """便捷函数：去除背景"""
    processor = FrameProcessor()
    return processor.remove_background(uploaded_files, tolerance, auto_crop, crop_padding, progress_callback)
