"""
SnowWeave Full Pipeline Module
完整流水线模块 - 一键完成视频生成、帧提取、背景去除的完整流程
"""

import os
import tempfile
import time
import shutil
from datetime import datetime
from typing import Optional, Tuple, List, Generator
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from lib.generate_sprite_animation import (
    load_reference_image,
    generate_animation_video
)
from lib.extract_sprite_frames import (
    extract_frames_from_video_segment,
    create_sprite_sheet,
    save_individual_frames
)
from lib.remove_background import process_directory

from .config import OUTPUT_DIR, t, get_current_language
from .api_manager import get_api_manager
from .video_generator import build_sprite_animation_prompt


class FullPipeline:
    """
    完整流水线
    一键完成：视频生成 → 帧提取 → 背景去除 → Sprite Sheet 输出
    """
    
    def __init__(self):
        self._api_manager = get_api_manager()
    
    def _clean_old_outputs(self, output_type: str = "full"):
        """清理旧的输出文件"""
        import shutil
        try:
            for item in os.listdir(OUTPUT_DIR):
                item_path = os.path.join(OUTPUT_DIR, item)
                if os.path.isdir(item_path) and item.startswith(output_type):
                    shutil.rmtree(item_path)
                    print(f"Deleted old output: {item_path}")
        except Exception as e:
            print(f"Error cleaning outputs: {e}")
    
    def run(
        self,
        image,
        action: str,
        start_time: float,
        end_time: float,
        max_frames: int,
        tolerance: int,
        auto_crop: bool,
        crop_padding: int,
        model_name: str,
        duration: int,
        backend: str = "gemini",
        resolution: str = "720p",
        max_workers: int = 3,
        progress_callback=None
    ) -> Tuple[Optional[List[str]], Optional[str], Optional[str], Optional[List[Image.Image]], str]:
        """
        执行完整流水线
        
        Args:
            image: 输入图片 (numpy array)
            action: 动作描述 (每行一个动作，支持多动作)
            start_time: 帧提取开始时间
            end_time: 帧提取结束时间
            max_frames: 最大帧数
            tolerance: 背景去除颜色容差
            auto_crop: 是否自动裁剪
            crop_padding: 裁剪边距
            model_name: 模型名称
            duration: 视频时长
            backend: 后端名称 (gemini/seedance)
            resolution: 分辨率 (Seedance 专用)
            max_workers: 最大并行数
            progress_callback: 进度回调函数 (progress_value, description)
        
        Returns:
            (video_paths, sprite_sheet_path, reference_path, preview_images, status_message)
        """
        # 验证 API 状态
        error_msg = self._api_manager.validate_backend(backend)
        if error_msg:
            return None, None, None, None, error_msg
        
        if image is None:
            return None, None, None, None, t("upload_image")
        
        try:
            self._clean_old_outputs("full")
            
            # 解析动作列表
            actions = [line.strip() for line in action.strip().split('\n') if line.strip()]
            if not actions:
                return None, None, None, None, "[ERROR] 请至少输入一个动作"
            
            # 准备参考图片
            if progress_callback:
                progress_callback(0, "准备参考图片...")
            
            temp_img_path = os.path.join(tempfile.gettempdir(), f"temp_{int(time.time())}.png")
            Image.fromarray(image).save(temp_img_path)
            
            reference_image = load_reference_image(temp_img_path)
            img_width, img_height = reference_image.size
            
            # 创建输出目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = os.path.join(OUTPUT_DIR, f"full_{timestamp}")
            os.makedirs(output_base, exist_ok=True)
            videos_dir = os.path.join(output_base, "videos")
            os.makedirs(videos_dir, exist_ok=True)
            
            # 保存参考图片
            reference_path = os.path.join(output_base, "reference_image.png")
            reference_image.save(reference_path)
            
            # 步骤1: 并行生成所有动作的视频
            if progress_callback:
                progress_callback(0.1, f"开始生成 {len(actions)} 个动作视频...")
            
            video_backend = self._api_manager.video_backend
            video_paths = []
            
            if len(actions) == 1:
                # 单个动作，直接生成
                full_prompt = build_sprite_animation_prompt(actions[0], img_width, img_height)
                video_result = video_backend.generate_video(
                    reference_image=reference_image,
                    prompt=full_prompt,
                    model_name=model_name,
                    duration=duration,
                    resolution=resolution if backend == "seedance" else None
                )
                
                if not video_result:
                    return None, None, None, None, "[ERROR] 视频生成失败"
                
                # 保存视频
                action_safe = actions[0].replace(" ", "_").replace("/", "_")[:30]
                video_filename = f"animation_{action_safe}.mp4"
                video_path = os.path.join(videos_dir, video_filename)
                
                if video_result.video_data:
                    with open(video_path, "wb") as f:
                        f.write(video_result.video_data)
                else:
                    shutil.copy(video_result.video_path, video_path)
                
                video_paths.append(video_path)
            else:
                # 多个动作，并行生成
                video_results = []
                with ThreadPoolExecutor(max_workers=min(len(actions), max_workers)) as executor:
                    # 提交所有任务
                    future_to_action = {
                        executor.submit(
                            self._generate_single_action_video,
                            reference_image,
                            actions[i],
                            model_name,
                            duration,
                            backend,
                            resolution,
                            videos_dir,
                            i
                        ): i for i in range(len(actions))
                    }
                    
                    # 等待完成
                    completed = 0
                    for future in as_completed(future_to_action):
                        action_idx = future_to_action[future]
                        result = future.result()
                        video_results.append(result)
                        completed += 1
                        
                        if progress_callback:
                            progress = 0.1 + (completed / len(actions)) * 0.3
                            progress_callback(progress, f"视频生成进度: {completed}/{len(actions)}")
                
                # 排序并收集视频路径
                video_results.sort(key=lambda x: x["action_idx"])
                video_paths = [r["video_path"] for r in video_results if r["success"]]
                
                if not video_paths:
                    return None, None, None, None, "[ERROR] 所有视频生成失败"
            
            # 步骤2: 提取所有视频的帧并合并
            if progress_callback:
                progress_callback(0.4, "提取帧中...")
            
            all_frames = []
            for i, video_path in enumerate(video_paths):
                frames = extract_frames_from_video_segment(
                    video_path,
                    float(start_time),
                    float(end_time),
                    int(max_frames)
                )
                all_frames.extend(frames)
                
                if progress_callback:
                    progress = 0.4 + (i + 1) / len(video_paths) * 0.2
                    progress_callback(progress, f"提取帧: {i + 1}/{len(video_paths)}")
            
            frames_dir = os.path.join(output_base, "1_extracted_frames")
            save_individual_frames(all_frames, output_dir=frames_dir)
            
            original_sheet, _ = create_sprite_sheet(all_frames, frame_size=None)
            original_sheet_path = os.path.join(output_base, "1_original_sprite_sheet.png")
            original_sheet.save(original_sheet_path)
            
            # 步骤3: 去除背景
            if progress_callback:
                progress_callback(0.6, "去除背景中...")

            
            nobg_dir = os.path.join(output_base, "2_nobg_frames")
            process_directory(
                frames_dir,
                output_dir=nobg_dir,
                tolerance=int(tolerance),
                num_workers=None,
                auto_crop=auto_crop,
                crop_padding=int(crop_padding)
            )
            
            # 步骤4: 创建最终 sprite sheet
            if progress_callback:
                progress_callback(0.9, t("step_final_sheet"))
            
            nobg_files = sorted([f for f in os.listdir(nobg_dir) if f.endswith('.png')])
            final_frames = [Image.open(os.path.join(nobg_dir, f)) for f in nobg_files]
            
            final_sheet, _ = create_sprite_sheet(final_frames, frame_size=None)
            final_sheet_path = os.path.join(output_base, "3_final_sprite_sheet.png")
            final_sheet.save(final_sheet_path)
            
            preview_images = final_frames[:12]
            
            # 清理临时文件
            try:
                os.remove(temp_img_path)
            except Exception as e:
                print(f"[Warning] Could not remove temp file: {e}")
            
            # 保存元数据
            self._save_metadata(
                output_base, timestamp, actions, model_name,
                start_time, end_time, max_frames,
                tolerance, auto_crop, crop_padding,
                backend, len(video_paths)
            )
            
            if progress_callback:
                progress_callback(1.0, "完成!")
            
            # 生成摘要
            summary = self._generate_summary(
                output_base, len(all_frames), len(final_frames), 
                len(actions), len(video_paths), get_current_language()
            )
            
            return (
                [os.path.abspath(v) for v in video_paths],
                os.path.abspath(final_sheet_path),
                os.path.abspath(reference_path),
                preview_images,
                summary
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None, None, None, f"[ERROR] {t('error')}: {str(e)}"
    
    def _generate_single_action_video(
        self,
        reference_image,
        action: str,
        model_name: str,
        duration: int,
        backend: str,
        resolution: str,
        videos_dir: str,
        action_idx: int
    ):
        """生成单个动作的视频（用于多线程）"""
        try:
            print(f"[Action {action_idx + 1}] 开始生成: {action}")
            
            img_width, img_height = reference_image.size
            full_prompt = build_sprite_animation_prompt(action, img_width, img_height)
            
            video_backend = self._api_manager.video_backend
            video_result = video_backend.generate_video(
                reference_image=reference_image,
                prompt=full_prompt,
                model_name=model_name,
                duration=duration,
                resolution=resolution if backend == "seedance" else None
            )
            
            # 保存视频
            action_safe = action.replace(" ", "_").replace("/", "_")[:30]
            video_filename = f"animation_{action_idx + 1}_{action_safe}.mp4"
            video_path = os.path.join(videos_dir, video_filename)
            
            if video_result.video_data:
                with open(video_path, "wb") as f:
                    f.write(video_result.video_data)
            else:
                shutil.copy(video_result.video_path, video_path)
            
            print(f"[Action {action_idx + 1}] ✓ 完成: {action}")
            
            return {
                "success": True,
                "action_idx": action_idx,
                "action": action,
                "video_path": video_path
            }
        except Exception as e:
            print(f"[Action {action_idx + 1}] ✗ 失败: {action} - {str(e)}")
            return {
                "success": False,
                "action_idx": action_idx,
                "action": action,
                "error": str(e)
            }
    
    def _save_metadata(
        self, output_dir: str, timestamp: str, actions, model_name: str,
        start_time: float, end_time: float, max_frames: int,
        tolerance: int, auto_crop: bool, crop_padding: int,
        backend: str, video_count: int
    ):
        """保存元数据文件"""
        metadata_path = os.path.join(output_dir, "metadata.txt")
        actions_text = actions if isinstance(actions, str) else "\n".join(f"  {i+1}. {a}" for i, a in enumerate(actions))
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(f"=== SnowWeave Full Pipeline Output / SnowWeave 完整流程输出 ===\n\n")
            f.write(f"Generation Time / 生成时间: {timestamp}\n")
            f.write(f"Backend / 后端: {backend}\n")
            f.write(f"Model Used / 使用模型: {model_name}\n")
            f.write(f"Total Videos / 视频总数: {video_count}\n\n")
            f.write(f"=== Actions / 动作列表 ===\n{actions_text}\n\n")
            f.write(f"=== Video Generation Parameters / 视频生成参数 ===\n")
            f.write(f"Extraction Time Range / 提取时间范围: {start_time}s - {end_time}s\n")
            f.write(f"Max Frames / 最大帧数: {max_frames}\n\n")
            f.write(f"=== Background Removal Parameters / 背景去除参数 ===\n")
            f.write(f"Color Tolerance / 颜色容差: {tolerance}\n")
            f.write(f"Auto Crop / 自动裁剪: {auto_crop}\n")
            f.write(f"Crop Padding / 裁剪边距: {crop_padding}px\n\n")
            f.write(f"=== Output Files / 输出文件 ===\n")
            f.write(f"Videos / 视频: videos/\n")
            f.write(f"Reference Image / 参考图片: reference_image.png\n")
            f.write(f"Original Extracted Frames / 原始提取帧: 1_extracted_frames/\n")
            f.write(f"No-Background Frames / 去背景帧: 2_nobg_frames/\n")
            f.write(f"Original Sprite Sheet / 原始Sprite Sheet: 1_original_sprite_sheet.png\n")
            f.write(f"Final Sprite Sheet / 最终Sprite Sheet: 3_final_sprite_sheet.png\n")
    
    def _generate_summary(self, output_base: str, num_frames: int, num_final_frames: int, num_actions: int, num_videos: int, language: str) -> str:
        """生成摘要文本"""
        if language == "zh":
            return f"""[OK] 完整流程执行完成!

输出目录: {output_base}

生成的文件:
  视频文件: videos/ ({num_videos} 个动作视频)
  参考图片: reference_image.png
  元数据文件: metadata.txt
  1. 原始提取帧: 1_extracted_frames/ ({num_frames} 帧)
  2. 去背景帧: 2_nobg_frames/ ({num_final_frames} 帧)
  3. 原始Sprite Sheet: 1_original_sprite_sheet.png
  4. 最终Sprite Sheet: 3_final_sprite_sheet.png

总共生成了 {num_actions} 个动作，{num_videos} 个视频
可直接在游戏引擎中使用最终Sprite Sheet!
可下载视频和Sprite Sheet
"""
        else:
            return f"""[OK] Full pipeline execution complete!

Output directory: {output_base}

Generated files:
  Videos: videos/ ({num_videos} action videos)
  Reference image: reference_image.png
  Metadata: metadata.txt
  1. Original frames: 1_extracted_frames/ ({num_frames} frames)
  2. No-BG frames: 2_nobg_frames/ ({num_final_frames} frames)
  3. Original Sprite Sheet: 1_original_sprite_sheet.png
  4. Final Sprite Sheet: 3_final_sprite_sheet.png

Generated {num_actions} actions, {num_videos} videos
Ready to use the final Sprite Sheet in game engines!
You can download videos and Sprite Sheets
"""


# 便捷函数
def run_full_pipeline(
    image, action, start_time, end_time, max_frames,
    tolerance, auto_crop, crop_padding, model_name, duration,
    progress_callback=None
):
    """便捷函数：执行完整流水线"""
    pipeline = FullPipeline()
    return pipeline.run(
        image, action, start_time, end_time, max_frames,
        tolerance, auto_crop, crop_padding, model_name, duration,
        progress_callback
    )
