"""
SnowWeave Video Generator Module
视频生成模块 - 处理动画视频的生成
"""

import os
import tempfile
import time
import shutil
from datetime import datetime
from typing import Optional, Tuple, Generator, List, Dict, Any
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from lib.generate_sprite_animation import load_reference_image

from .config import OUTPUT_DIR, t, get_current_language
from .api_manager import get_api_manager


def build_sprite_animation_prompt(action: str, img_width: int, img_height: int) -> str:
    """
    构建 Sprite 动画生成的提示词
    
    Args:
        action: 动作描述
        img_width: 图片宽度
        img_height: 图片高度
    
    Returns:
        完整的提示词
    """
    return f"""
Create a smooth sprite animation of a STYLIZED, NON-REALISTIC game character performing {action} IN PLACE.


CRITICAL REQUIREMENTS:
- Character STAYS IN THE CENTER, does NOT move left or right across the screen
- Only the character's body/limbs animate, position remains FIXED
- Smooth, fluid animation with natural motion
- Keep the exact same character design, colors, and art style
- Loop-able animation cycle

VISUAL STYLE REQUIREMENTS:
- NO physics effects (no particles, debris, dust, etc.)
- NO lighting effects (no shadows, highlights, glows, reflections)
- NO post-processing effects (no blur, bloom, color grading)
- Flat, clean animation with solid colors only
- Simple sprite animation style without any special effects

Camera: Fixed, character stays in center and animates in place
Effects: NONE - no physics, lighting, or post-processing effects
"""


class VideoGenerator:
    """
    视频生成器
    处理动画视频的生成、保存和元数据管理
    支持 Gemini 和 Seedance 后端，支持多动作并行生成
    """
    
    def __init__(self):
        self._api_manager = get_api_manager()
    
    def _clean_old_outputs(self, output_type: str = "video"):
        """清理旧的输出文件"""
        try:
            for item in os.listdir(OUTPUT_DIR):
                item_path = os.path.join(OUTPUT_DIR, item)
                if os.path.isdir(item_path) and item.startswith(output_type):
                    shutil.rmtree(item_path)
                    print(f"Deleted old output: {item_path}")
        except Exception as e:
            print(f"Error cleaning outputs: {e}")
    
    def generate(
        self,
        image,
        action: str,
        model_name: str,
        duration: int = 6,
        backend: str = "seedance",
        resolution: str = "720p"
    ) -> Generator[Tuple[Optional[str], Optional[str], str], None, None]:
        """
        生成单个动作的动画视频
        
        Args:
            image: 输入图片 (numpy array)
            action: 动作描述
            model_name: 模型名称
            duration: 视频时长（秒）
            backend: 后端名称 (gemini/seedance)
            resolution: 分辨率（Seedance专用）
        
        Yields:
            (video_path, reference_path, status_message) 元组
        """
        # 验证 API 状态
        error_msg = self._api_manager.validate_backend(backend)
        if error_msg:
            yield None, None, error_msg
            return
        
        if image is None:
            yield None, None, t("upload_image")
            return
        
        try:
            # 清理旧输出
            self._clean_old_outputs()
            
            yield None, None, t("loading_image")
            
            # 保存临时图片
            temp_img_path = os.path.join(tempfile.gettempdir(), f"temp_{int(time.time())}.png")
            Image.fromarray(image).save(temp_img_path)
            
            # 加载图片
            reference_image = load_reference_image(temp_img_path)
            img_width, img_height = reference_image.size
            
            # 创建输出目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = os.path.join(OUTPUT_DIR, f"video_{timestamp}")
            os.makedirs(output_base, exist_ok=True)
            
            # 保存参考图片
            reference_path = os.path.join(output_base, "reference_image.png")
            reference_image.save(reference_path)
            
            yield None, None, t("generating_video")
            
            # 构建提示词
            full_prompt = build_sprite_animation_prompt(action, img_width, img_height)
            
            # 使用统一后端生成视频
            video_backend = self._api_manager.video_backend
            video_result = video_backend.generate_video(
                reference_image=reference_image,
                prompt=full_prompt,
                model_name=model_name,
                duration=duration,
                resolution=resolution if backend == "seedance" else None
            )
            
            # 保存视频
            video_path = os.path.join(output_base, "animation.mp4")
            if video_result.video_data:
                with open(video_path, "wb") as f:
                    f.write(video_result.video_data)
            else:
                shutil.copy(video_result.video_path, video_path)
            
            # 保存元数据
            self._save_metadata(output_base, timestamp, action, model_name)
            
            # 清理临时文件
            os.remove(temp_img_path)
            
            # 生成摘要
            summary = self._generate_summary(output_base, get_current_language())
            
            yield os.path.abspath(video_path), os.path.abspath(reference_path), summary
            
        except Exception as e:
            yield None, None, f"[ERROR] {t('error')}: {str(e)}"
    
    def _generate_single_video(
        self,
        reference_image: Image.Image,
        action: str,
        model_name: str,
        duration: int,
        backend: str,
        resolution: str,
        output_base: str,
        action_index: int
    ) -> Dict[str, Any]:
        """
        生成单个动作的视频
        
        Returns:
            包含视频路径、状态等信息的字典
        """
        try:
            img_width, img_height = reference_image.size
            
            # 构建提示词
            full_prompt = build_sprite_animation_prompt(action, img_width, img_height)
            
            print(f"[动作 {action_index + 1}] 开始生成: {action}")
            
            # 使用统一后端生成视频
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
            video_filename = f"animation_{action_index + 1}_{action_safe}.mp4"
            video_path = os.path.join(output_base, video_filename)
            
            if video_result.video_data:
                with open(video_path, "wb") as f:
                    f.write(video_result.video_data)
            else:
                shutil.copy(video_result.video_path, video_path)
            
            print(f"[动作 {action_index + 1}] ✓ 完成: {action}")
            
            return {
                "success": True,
                "action": action,
                "action_index": action_index,
                "video_path": video_path,
                "video_filename": video_filename
            }
            
        except Exception as e:
            print(f"[动作 {action_index + 1}] ✗ 失败: {action} - {str(e)}")
            return {
                "success": False,
                "action": action,
                "action_index": action_index,
                "error": str(e)
            }
    
    def generate_multiple(
        self,
        image,
        actions: List[str],
        model_name: str,
        duration: int = 6,
        backend: str = "seedance",
        resolution: str = "720p",
        max_workers: int = 3
    ) -> Generator[Tuple[Optional[List[str]], Optional[str], str], None, None]:
        """
        生成多个动作的视频（并行处理）
        
        Args:
            image: 输入图片 (numpy array)
            actions: 动作描述列表
            model_name: 模型名称
            duration: 视频时长（秒）
            backend: 后端名称
            resolution: 分辨率（Seedance 专用）
            max_workers: 最大并行数
        
        Yields:
            (video_paths, reference_path, status_message) 元组
        """
        # 验证 API 状态
        error_msg = self._api_manager.validate_backend(backend)
        if error_msg:
            yield None, None, error_msg
            return
        
        if image is None:
            yield None, None, t("upload_image")
            return
        
        if not actions or len(actions) == 0:
            yield None, None, "[ERROR] 请至少输入一个动作 / Please enter at least one action"
            return
        
        try:
            # 清理旧输出
            self._clean_old_outputs("video_batch")
            
            yield None, None, t("loading_image")
            
            # 保存临时图片
            temp_img_path = os.path.join(tempfile.gettempdir(), f"temp_{int(time.time())}.png")
            Image.fromarray(image).save(temp_img_path)
            
            # 加载图片
            reference_image = load_reference_image(temp_img_path)
            
            # 创建输出目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = os.path.join(OUTPUT_DIR, f"video_batch_{timestamp}")
            os.makedirs(output_base, exist_ok=True)
            
            # 保存参考图片
            reference_path = os.path.join(output_base, "reference_image.png")
            reference_image.save(reference_path)
            
            lang = get_current_language()
            if lang == "zh":
                yield None, None, f"开始生成 {len(actions)} 个动作视频 (并行: {max_workers})...\n后端: {backend}"
            else:
                yield None, None, f"Generating {len(actions)} action videos (parallel: {max_workers})...\nBackend: {backend}"
            
            # 并行生成视频
            results = []
            completed_count = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_action = {
                    executor.submit(
                        self._generate_single_video,
                        reference_image,
                        action,
                        model_name,
                        duration,
                        backend,
                        resolution,
                        output_base,
                        i
                    ): (i, action) for i, action in enumerate(actions)
                }
                
                # 获取完成的任务
                for future in as_completed(future_to_action):
                    action_index, action_name = future_to_action[future]
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                    # 更新进度
                    if lang == "zh":
                        progress_msg = f"进度: {completed_count}/{len(actions)} - "
                        if result["success"]:
                            progress_msg += f"✓ {action_name}"
                        else:
                            progress_msg += f"✗ {action_name}: {result['error']}"
                    else:
                        progress_msg = f"Progress: {completed_count}/{len(actions)} - "
                        if result["success"]:
                            progress_msg += f"✓ {action_name}"
                        else:
                            progress_msg += f"✗ {action_name}: {result['error']}"
                    
                    yield None, None, progress_msg
            
            # 排序结果（按原始顺序）
            results.sort(key=lambda x: x["action_index"])
            
            # 保存元数据
            self._save_batch_metadata(output_base, timestamp, results, model_name, backend)
            
            # 收集成功的视频路径
            success_videos = [r["video_path"] for r in results if r["success"]]
            failed_count = len(results) - len(success_videos)
            
            # 清理临时文件
            os.remove(temp_img_path)
            
            # 生成摘要
            summary = self._generate_batch_summary(
                output_base, results, get_current_language()
            )
            
            yield success_videos if success_videos else None, os.path.abspath(reference_path), summary
            
        except Exception as e:
            yield None, None, f"[ERROR] {t('error')}: {str(e)}"
    
    def _save_batch_metadata(
        self, 
        output_dir: str, 
        timestamp: str, 
        results: List[Dict], 
        model_name: str,
        backend: str
    ):
        """保存批量生成的元数据文件"""
        metadata_path = os.path.join(output_dir, "metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(f"=== Batch Video Generation / 批量视频生成 ===\n\n")
            f.write(f"Generation Time / 生成时间: {timestamp}\n")
            f.write(f"Backend / 后端: {backend}\n")
            f.write(f"Model Used / 使用模型: {model_name}\n")
            f.write(f"Total Actions / 总动作数: {len(results)}\n")
            f.write(f"Successful / 成功: {sum(1 for r in results if r['success'])}\n")
            f.write(f"Failed / 失败: {sum(1 for r in results if not r['success'])}\n\n")
            
            f.write(f"=== Actions / 动作列表 ===\n")
            for r in results:
                if r["success"]:
                    f.write(f"✓ {r['action']}: {r['video_filename']}\n")
                else:
                    f.write(f"✗ {r['action']}: {r['error']}\n")
    
    def _generate_batch_summary(self, output_dir: str, results: List[Dict], language: str) -> str:
        """生成批量生成的摘要文本"""
        success_count = sum(1 for r in results if r["success"])
        fail_count = len(results) - success_count
        
        if language == "zh":
            summary = f"""[OK] 批量视频生成完成!

输出目录: {output_dir}
总动作数: {len(results)}
成功: {success_count}
失败: {fail_count}

生成的视频:
"""
            for r in results:
                if r["success"]:
                    summary += f"  ✓ {r['action']}: {r['video_filename']}\n"
                else:
                    summary += f"  ✗ {r['action']}: 失败 - {r['error']}\n"
            
            summary += f"\n参考图片: reference_image.png\n元数据: metadata.txt"
        else:
            summary = f"""[OK] Batch video generation complete!

Output directory: {output_dir}
Total actions: {len(results)}
Successful: {success_count}
Failed: {fail_count}

Generated videos:
"""
            for r in results:
                if r["success"]:
                    summary += f"  ✓ {r['action']}: {r['video_filename']}\n"
                else:
                    summary += f"  ✗ {r['action']}: Failed - {r['error']}\n"
            
            summary += f"\nReference image: reference_image.png\nMetadata: metadata.txt"
        
        return summary
    
    def _save_metadata(self, output_dir: str, timestamp: str, action: str, model_name: str):
        """保存元数据文件"""
        metadata_path = os.path.join(output_dir, "metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(f"Generation Time / 生成时间: {timestamp}\n")
            f.write(f"Action Description / 动作描述: {action}\n")
            f.write(f"Model Used / 使用模型: {model_name}\n")
            f.write(f"Video File / 视频文件: animation.mp4\n")
            f.write(f"Reference Image / 参考图片: reference_image.png\n")
    
    def _generate_summary(self, output_dir: str, language: str) -> str:
        """生成摘要文本"""
        if language == "zh":
            return f"""[OK] 视频生成完成!

输出目录: {output_dir}
视频文件: animation.mp4
参考图片: reference_image.png
元数据: metadata.txt

可直接下载视频和图片
"""
        else:
            return f"""[OK] Video generation complete!

Output directory: {output_dir}
Video file: animation.mp4
Reference image: reference_image.png
Metadata: metadata.txt

You can download the video and images directly
"""


# 便捷函数
def generate_video_batch(image, actions: List[str], model_name: str, duration: int = 6, backend: str = "seedance", resolution: str = "720p", max_workers: int = 3):
    """便捷函数：批量生成动画视频"""
    generator = VideoGenerator()
    return generator.generate_multiple(image, actions, model_name, duration, backend, resolution, max_workers)
