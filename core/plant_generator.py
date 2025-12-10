"""
SnowWeave Plant Generator Module
植物生成模块 - 生成植物各生长阶段的动画帧
"""

import os
import json
import shutil
import tempfile
import time
import traceback
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any, Generator
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from lib.generate_sprite_animation import load_reference_image
from lib.extract_sprite_frames import (
    extract_frames_from_video_segment,
    create_sprite_sheet,
    save_individual_frames,
    process_frames_to_target_size
)
from lib.remove_background import process_directory

from .config import OUTPUT_DIR, DEFAULT_DIRT_IMAGE_PATH, t, get_current_language
from .api_manager import get_api_manager
from .frame_processor import FrameProcessor


def generate_stage_prompt(base_prompt: str, stage_index: int, total_stages: int) -> str:
    """
    生成特定阶段的提示词
    
    Args:
        base_prompt: 基础植物描述
        stage_index: 当前阶段索引 (0-based)
        total_stages: 总阶段数
    
    Returns:
        该阶段的完整提示词
    """
    progress = stage_index / (total_stages - 1) if total_stages > 1 else 1.0
    
    # 根据进度确定阶段描述和动画描述
    if progress < 0.2:
        # 萌芽期
        stage_desc = "a TINY seed just cracking open with minimal green showing"
        size_desc = "MICROSCOPIC - the sprout must be SMALLER than the soil mound, only a tiny green dot or crack visible"
        growth_animation = """- Start: bare soil, seed hidden underground
- Animation: soil slightly cracks, a TINY green tip barely peeks out
- End: only a VERY SMALL green dot or thin line visible on soil surface
- Motion: minimal movement, just soil cracking slightly

CRITICAL SIZE CONSTRAINT:
- The sprout must be EXTREMELY SMALL - like a tiny green dot
- Height: maximum 10% of the soil tile height
- The sprout must stay WITHIN the soil tile boundary, NOT extend above it
- Think of it as just the tip of a seed cracking - almost invisible
- NO LEAVES yet, just a tiny green tip emerging"""
    elif progress < 0.4:
        # 幼苗期
        stage_desc = "small seedling with first few true leaves"
        size_desc = "small, about 25-35% of final size, slightly taller than soil tile"
        growth_animation = """- Start: tiny sprout from previous stage (within soil tile height)
- Animation: stem elongates upward, first true leaves unfurl and expand
- End: small but established seedling with several leaves, now extending above tile
- Motion: stem stretches upward, leaves gradually open outward"""
    elif progress < 0.6:
        # 成长期
        stage_desc = "growing plant with developing stems and leaves"
        size_desc = "medium, about 50-60% of final size"
        growth_animation = """- Start: small seedling from previous stage
- Animation: main stem grows taller, side branches develop, more leaves appear
- End: medium-sized plant with fuller foliage
- Motion: upward growth with branching, leaves multiply and spread"""
    elif progress < 0.8:
        # 接近成熟
        stage_desc = "well-developed plant approaching maturity"
        size_desc = "large, about 70-80% of final size"
        growth_animation = """- Start: medium plant from previous stage
- Animation: plant fills out, stems thicken, foliage becomes dense
- End: large, well-developed plant nearly at full size
- Motion: expanding outward and upward, leaves reaching full size"""
    else:
        # 完全成熟
        stage_desc = "fully mature and grown plant at maximum size"
        size_desc = "full size, 100% grown"
        growth_animation = """- Start: large plant from previous stage
- Animation: final growth spurt, plant reaches maximum size, may flower/fruit
- End: fully mature plant at peak condition with gentle idle sway
- Motion: final stretching upward, then settling into gentle breeze animation"""
    
    return f"""
Create a sprite animation of a STYLIZED {base_prompt} showing growth transition on a dirt soil tile.

GROWTH STAGE: Stage {stage_index + 1} of {total_stages} ({int(progress * 100)}% maturity)
- Current stage: {stage_desc}
- Target size: {size_desc}

GROWTH ANIMATION (CRITICAL):
{growth_animation}

ANIMATION REQUIREMENTS:
- Show continuous growth transformation throughout the video
- Plant physically changes size/shape from start to end
- Growth direction: from bottom (soil) upward
- Final frames: plant at this stage's full size with gentle idle sway
- Natural, organic growth motion
- Plant stays rooted in same position

STYLE REQUIREMENTS:
- 2D game sprite, isometric pixel art style
- Clean, flat colors with clear outlines
- No realistic lighting or complex shadows
- NO SHADOWS of any kind (no drop shadow, no cast shadow, no ambient shadow)
- Cartoon/stylized game aesthetic
- Background must be pure white (#FFFFFF), completely clean with no shadows or gradients
"""


def build_full_prompt(stage_prompt: str, base_prompt: str, img_width: int, img_height: int) -> str:
    """
    构建完整的视频生成提示词
    
    Args:
        stage_prompt: 阶段提示词
        base_prompt: 基础植物描述
        img_width: 图片宽度
        img_height: 图片高度
    
    Returns:
        完整的提示词
    """
    return f"""
{stage_prompt}

REFERENCE IMAGE (CRITICAL - THIS IS THE SOIL/GROUND):
- The provided reference image IS the dirt/soil tile base - use it directly
- This image represents the ground where the plant will grow FROM
- DO NOT replace or modify the soil - the {base_prompt} must grow OUT OF this exact soil
- The plant emerges and grows upward from this soil surface
- Maintain the isometric perspective and colors of the reference soil
- Video dimensions: {img_width}x{img_height} pixels

CRITICAL TECHNICAL REQUIREMENTS:
- Plant grows FROM the soil shown in reference image
- START IMMEDIATELY with growth animation - NO fade in effect
- Object STAYS IN THE CENTER, does NOT move across the screen
- Keep the dirt/soil from reference image visible as the base throughout
- Only the plant animates (growing), soil remains static

Resolution: {img_width}x{img_height}
Effects: NONE - flat colors only
"""


class PlantGenerator:
    """
    植物生成器
    生成植物各生长阶段的动画帧
    """
    
    def __init__(self):
        self._api_manager = get_api_manager()
        self._frame_processor = FrameProcessor()
        self._progress_lock = threading.Lock()
        self._stage_status = {}  # 用于跟踪每个阶段的状态
    
    def _clean_old_outputs(self, output_type: str = "plant"):
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
        ref_images: Optional[List[str]],
        prompt: str,
        plant_id: str,
        stages: int,
        target_width: int,
        frames_per_stage: int,
        tolerance: int,
        auto_crop: bool,
        crop_padding: int,
        model_name: str,
        duration: int,
        backend: str = "gemini",
        resolution: str = "720p",
        reference_mode: str = "last_frame"
    ) -> Generator[Tuple[Optional[str], Optional[List], Optional[str], Optional[str], str], None, None]:
        """
        生成植物各生长阶段的帧图片
        
        Args:
            image: 参考图片 (numpy array)
            ref_images: 额外参考图片列表
            prompt: 植物描述
            plant_id: 植物 ID
            stages: 生长阶段数
            target_width: 输出宽度
            frames_per_stage: 每阶段帧数
            tolerance: 背景去除容差
            auto_crop: 是否自动裁剪
            crop_padding: 裁剪边距
            model_name: 模型名称
            duration: 视频时长
            backend: 后端名称
            resolution: 分辨率 (Seedance 专用)
            reference_mode: 参考图片模式
        
        Yields:
            (preview_path, preview_frames, config_path, video_path, status_message)
        """
        # 验证后端
        error_msg = self._api_manager.validate_backend(backend)
        if error_msg:
            yield None, None, None, None, error_msg
            return
        
        # 验证输入
        if image is None and (not prompt or prompt.strip() == ""):
            yield None, None, None, None, "[ERROR] " + t("plant_need_input")
            return
        
        if not plant_id or plant_id.strip() == "":
            yield None, None, None, None, "[ERROR] " + t("plant_need_id")
            return
        
        plant_id = plant_id.strip().replace(" ", "_").lower()
        
        try:
            self._clean_old_outputs("plant")
            
            yield None, None, None, None, f"正在初始化 (后端: {backend})... / Initializing (backend: {backend})..."
            
            # 创建输出目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = os.path.join(OUTPUT_DIR, f"plant_{plant_id}_{timestamp}")
            final_output_dir = os.path.join(output_base, "final_frames")
            videos_dir = os.path.join(output_base, "videos")
            os.makedirs(final_output_dir, exist_ok=True)
            os.makedirs(videos_dir, exist_ok=True)
            
            # 准备参考图片
            reference_image_path, temp_img_path = self._prepare_reference_image(image, output_base)
            if reference_image_path is None:
                yield None, None, None, None, "[ERROR] Failed to prepare reference image"
                return
            
            yield None, None, None, None, "参考图片已准备好 / Reference image ready"
            
            # 基础提示词
            base_prompt = prompt if prompt and prompt.strip() else f"a {plant_id.replace('_', ' ')}"
            
            # 初始化变量
            all_preview_frames = []
            total_frames = 0
            stage_results = []
            all_video_paths = []
            latest_video_path = None
            current_reference_path = reference_image_path
            
            # 记录原始图片尺寸
            original_image = load_reference_image(reference_image_path)
            ORIGINAL_WIDTH, ORIGINAL_HEIGHT = original_image.size
            print(f"[PlantGenerator] Original image size: {ORIGINAL_WIDTH}x{ORIGINAL_HEIGHT}")
            
            # 初始化状态跟踪
            self._stage_status = {i: "pending" for i in range(stages)}
            
            # ===== 第一步: 并行生成所有阶段的视频 =====
            yield None, None, None, None, f"开始并行生成 {stages} 个阶段的视频 (后端: {backend})..."
            
            video_results = []
            with ThreadPoolExecutor(max_workers=min(stages, 3)) as executor:
                # 提交所有视频生成任务
                future_to_stage = {
                    executor.submit(
                        self._generate_single_stage_video,
                        stage_idx,
                        stages,
                        base_prompt,
                        reference_image_path,
                        model_name,
                        duration,
                        resolution,
                        ref_images,
                        ORIGINAL_WIDTH,
                        ORIGINAL_HEIGHT,
                        videos_dir
                    ): stage_idx for stage_idx in range(stages)
                }
                
                # 等待任务完成并更新进度
                completed = 0
                for future in as_completed(future_to_stage):
                    stage_idx = future_to_stage[future]
                    result = future.result()
                    video_results.append(result)
                    completed += 1
                    
                    status_msg = f"视频生成进度: {completed}/{stages} - "
                    if result["success"]:
                        status_msg += f"✓ Stage {stage_idx + 1}"
                        all_video_paths.append(result["video_path"])
                    else:
                        status_msg += f"✗ Stage {stage_idx + 1}: {result['error']}"
                    
                    yield None, None, None, None, status_msg
            
            # 按阶段顺序排序视频结果
            video_results.sort(key=lambda x: x["stage_idx"])
            
            # 检查是否所有视频都成功
            failed_stages = [r for r in video_results if not r["success"]]
            if failed_stages:
                error_msg = f"有 {len(failed_stages)} 个阶段视频生成失败\n"
                for r in failed_stages:
                    error_msg += f"  Stage {r['stage_idx'] + 1}: {r['error']}\n"
                yield None, None, None, None, error_msg
            
            # ===== 第二步: 串行处理每个视频的帧提取 =====
            yield None, None, None, None, "所有视频生成完成，开始处理帧..."
            
            current_reference_path = reference_image_path
            
            for result in video_results:
                if not result["success"]:
                    # 记录失败
                    stage_results.append({
                        "stage": result["stage_idx"] + 1,
                        "success": False,
                        "error": result["error"]
                    })
                    continue
                
                stage_idx = result["stage_idx"]
                video_path = result["video_path"]
                
                yield None, None, None, None, f"[阶段 {stage_idx + 1}/{stages}] 提取并处理帧..."
                
                # 处理帧提取
                stage_dir = os.path.join(output_base, f"stage_{stage_idx + 1}")
                os.makedirs(stage_dir, exist_ok=True)
                
                # 复制视频到阶段目录
                stage_video_path = os.path.join(stage_dir, "animation.mp4")
                shutil.copy(video_path, stage_video_path)
                
                # 提取和处理帧
                frame_result = self._extract_and_process_frames(
                    video_path=video_path,
                    stage_idx=stage_idx,
                    stages=stages,
                    plant_id=plant_id,
                    stage_dir=stage_dir,
                    final_output_dir=final_output_dir,
                    frames_per_stage=frames_per_stage,
                    target_width=target_width,
                    tolerance=tolerance,
                    auto_crop=auto_crop,
                    crop_padding=crop_padding,
                    ORIGINAL_WIDTH=ORIGINAL_WIDTH,
                    ORIGINAL_HEIGHT=ORIGINAL_HEIGHT,
                    duration=duration,
                    all_preview_frames=all_preview_frames,
                    latest_video_path=latest_video_path
                )
                
                if frame_result is None:
                    stage_results.append({
                        "stage": stage_idx + 1,
                        "success": False,
                        "error": "Frame extraction failed"
                    })
                    continue
                
                # 更新参考图片路径
                if stage_idx < stages - 1:
                    if reference_mode == "last_frame" and frame_result.get("original_last_frame_path"):
                        current_reference_path = frame_result["original_last_frame_path"]
                
                # 记录结果
                stage_result = {
                    "stage": stage_idx + 1,
                    "success": True,
                    "transition_frames": frame_result["transition_count"],
                    "idle_frames": frame_result["idle_count"],
                    "total_frames": frame_result["stage_frame_count"],
                    "video_path": video_path,
                    "last_frame_path": frame_result.get("original_last_frame_path")
                }
                
                stage_results.append(stage_result)
                total_frames += frame_result["stage_frame_count"]
                latest_video_path = os.path.abspath(video_path)
                
                yield None, None, None, None, f"[阶段 {stage_idx + 1}/{stages}] ✓ 完成! {frame_result['transition_count']} 过渡帧 + {frame_result['idle_count']} 循环帧"
            
            yield None, None, None, None, "正在生成配置文件... / Generating config file..."
            
            # 生成配置文件和元数据
            config_path = self._generate_config(
                output_base, plant_id, base_prompt, stages, stage_results, final_output_dir
            )
            
            self._generate_metadata(
                output_base, plant_id, timestamp, stages, total_frames, target_width,
                base_prompt, stage_results, all_video_paths
            )
            
            # 创建预览合成图
            preview_path = None
            if all_preview_frames:
                preview_sheet, _ = create_sprite_sheet(all_preview_frames, frame_size=None)
                preview_path = os.path.join(output_base, "preview_sheet.png")
                preview_sheet.save(preview_path)
            
            # 清理临时文件 (使用 try-except 避免权限错误)
            if temp_img_path and os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
                except Exception as e:
                    print(f"[Warning] Could not remove temp file {temp_img_path}: {e}")
            
            # 生成摘要
            summary = self._generate_summary(
                plant_id, stages, stage_results, total_frames, target_width,
                all_video_paths, output_base, videos_dir, final_output_dir, config_path
            )
            
            # 返回所有视频路径用于 Gradio 预览
            video_paths_for_preview = [os.path.abspath(v) for v in all_video_paths] if all_video_paths else None
            
            yield (
                os.path.abspath(preview_path) if preview_path else None,
                all_preview_frames[:12] if all_preview_frames else [],
                os.path.abspath(config_path),
                video_paths_for_preview,  # 改为视频列表
                summary
            )
            
        except Exception as e:
            traceback.print_exc()
            yield None, None, None, None, f"[ERROR] {t('error')}: {str(e)}"
    
    def _prepare_reference_image(self, image, output_base: str) -> Tuple[Optional[str], Optional[str]]:
        """准备参考图片"""
        temp_img_path = None
        
        if image is not None:
            temp_img_path = os.path.join(tempfile.gettempdir(), f"plant_ref_{int(time.time())}.png")
            Image.fromarray(image).save(temp_img_path)
            return temp_img_path, temp_img_path
        else:
            # 使用默认泥土图片
            if os.path.exists(DEFAULT_DIRT_IMAGE_PATH):
                return DEFAULT_DIRT_IMAGE_PATH, None
            else:
                # 创建绿色背景
                temp_img = Image.new('RGB', (512, 512), color=(0, 255, 0))
                temp_img_path = os.path.join(output_base, "temp_reference.png")
                temp_img.save(temp_img_path)
                return temp_img_path, temp_img_path
    
    def _generate_single_stage_video(
        self,
        stage_idx: int,
        stages: int,
        base_prompt: str,
        reference_image_path: str,
        model_name: str,
        duration: int,
        resolution: str,
        ref_images: Optional[List[str]],
        ORIGINAL_WIDTH: int,
        ORIGINAL_HEIGHT: int,
        videos_dir: str
    ) -> Dict[str, Any]:
        """
        单独生成一个阶段的视频(用于多线程)
        
        Returns:
            包含视频路径、状态等信息的字典
        """
        try:
            # 更新状态
            with self._progress_lock:
                self._stage_status[stage_idx] = "generating"
            
            print(f"[Stage {stage_idx + 1}] 开始生成视频...")
            
            # 生成提示词
            stage_prompt = generate_stage_prompt(base_prompt, stage_idx, stages)
            
            # 加载并裁剪参考图片
            reference_image = load_reference_image(reference_image_path)
            reference_image, progress = self._crop_reference_image(
                reference_image, stage_idx, stages, ORIGINAL_WIDTH, ORIGINAL_HEIGHT
            )
            
            img_width, img_height = reference_image.size
            
            # 构建完整提示词
            full_prompt = build_full_prompt(stage_prompt, base_prompt, img_width, img_height)
            
            # 准备额外参考图
            extra_ref_images = None
            if stage_idx == 0 and ref_images and "lite-i2v" in model_name.lower():
                extra_ref_images = [f.name if hasattr(f, 'name') else f for f in ref_images]
            
            # 生成视频
            video_backend = self._api_manager.video_backend
            video_result = video_backend.generate_video(
                reference_image=reference_image,
                prompt=full_prompt,
                model_name=model_name,
                duration=duration,
                resolution=resolution,
                reference_images=extra_ref_images
            )
            
            # 保存视频
            plant_id = base_prompt.replace(" ", "_").lower()[:20]
            video_filename = f"{plant_id}_stage{stage_idx + 1}.mp4"
            video_path = os.path.join(videos_dir, video_filename)
            
            if video_result.video_data:
                with open(video_path, "wb") as f:
                    f.write(video_result.video_data)
            else:
                shutil.copy(video_result.video_path, video_path)
            
            # 更新状态
            with self._progress_lock:
                self._stage_status[stage_idx] = "completed"
            
            print(f"[Stage {stage_idx + 1}] ✓ 视频生成完成: {video_filename}")
            
            return {
                "success": True,
                "stage_idx": stage_idx,
                "video_path": video_path,
                "video_filename": video_filename
            }
            
        except Exception as e:
            with self._progress_lock:
                self._stage_status[stage_idx] = "failed"
            
            print(f"[Stage {stage_idx + 1}] ✗ 视频生成失败: {str(e)}")
            traceback.print_exc()
            
            return {
                "success": False,
                "stage_idx": stage_idx,
                "error": str(e)
            }
    
    def _generate_stage(
        self,
        stage_idx: int,
        stages: int,
        plant_id: str,
        base_prompt: str,
        current_reference_path: str,
        reference_image_path: str,
        ref_images: Optional[List[str]],
        model_name: str,
        duration: int,
        resolution: str,
        reference_mode: str,
        frames_per_stage: int,
        target_width: int,
        tolerance: int,
        auto_crop: bool,
        crop_padding: int,
        output_base: str,
        videos_dir: str,
        final_output_dir: str,
        ORIGINAL_WIDTH: int,
        ORIGINAL_HEIGHT: int,
        all_preview_frames: List,
        all_video_paths: List,
        latest_video_path: Optional[str]
    ) -> Generator:
        """生成单个阶段"""
        
        status_msg = f"[阶段 {stage_idx + 1}/{stages}] 正在生成视频... / [Stage {stage_idx + 1}/{stages}] Generating video..."
        yield None, None, None, latest_video_path, status_msg
        
        # 生成该阶段的提示词
        stage_prompt = generate_stage_prompt(base_prompt, stage_idx, stages)
        
        try:
            # 加载并裁剪参考图片
            reference_image = load_reference_image(current_reference_path)
            reference_image, progress = self._crop_reference_image(
                reference_image, stage_idx, stages, ORIGINAL_WIDTH, ORIGINAL_HEIGHT
            )
            
            img_width, img_height = reference_image.size
            
            # 构建完整提示词
            full_prompt = build_full_prompt(stage_prompt, base_prompt, img_width, img_height)
            
            # 准备额外参考图
            extra_ref_images = None
            if stage_idx == 0 and ref_images and "lite-i2v" in model_name.lower():
                extra_ref_images = [f.name if hasattr(f, 'name') else f for f in ref_images]
            
            # 生成视频
            video_backend = self._api_manager.video_backend
            try:
                video_result = video_backend.generate_video(
                    reference_image=reference_image,
                    prompt=full_prompt,
                    model_name=model_name,
                    duration=duration,
                    resolution=resolution,
                    reference_images=extra_ref_images
                )
            except Exception as e:
                error_msg = f"[阶段 {stage_idx + 1}] 视频生成失败: {str(e)}"
                yield None, None, None, latest_video_path, error_msg
                # 错误情况 yield 结果
                yield {"stage_result": {"stage": stage_idx + 1, "success": False, "error": str(e)}}
                return
            
            yield None, None, None, latest_video_path, f"[阶段 {stage_idx + 1}/{stages}] 保存视频中..."
            
            # 保存视频
            video_filename = f"{plant_id}_stage{stage_idx + 1}.mp4"
            video_path = os.path.join(videos_dir, video_filename)
            
            stage_dir = os.path.join(output_base, f"stage_{stage_idx + 1}")
            os.makedirs(stage_dir, exist_ok=True)
            stage_video_path = os.path.join(stage_dir, "animation.mp4")
            
            if video_result.video_data:
                with open(video_path, "wb") as f:
                    f.write(video_result.video_data)
            else:
                shutil.copy(video_result.video_path, video_path)
            
            shutil.copy(video_path, stage_video_path)
            
            all_video_paths.append(video_path)
            latest_video_path = os.path.abspath(video_path)
            
            yield None, None, None, latest_video_path, f"[阶段 {stage_idx + 1}/{stages}] ✓ 视频已保存: {video_filename}\n提取帧中..."
            
            # 提取帧
            result = self._extract_and_process_frames(
                video_path=video_path,
                stage_idx=stage_idx,
                stages=stages,
                plant_id=plant_id,
                stage_dir=stage_dir,
                final_output_dir=final_output_dir,
                frames_per_stage=frames_per_stage,
                target_width=target_width,
                tolerance=tolerance,
                auto_crop=auto_crop,
                crop_padding=crop_padding,
                ORIGINAL_WIDTH=ORIGINAL_WIDTH,
                ORIGINAL_HEIGHT=ORIGINAL_HEIGHT,
                duration=duration,
                all_preview_frames=all_preview_frames,
                latest_video_path=latest_video_path
            )
            
            if result is None:
                yield {"stage_result": {"stage": stage_idx + 1, "success": False, "error": "Frame extraction failed"}}
                return
            
            # 确定下一阶段的参考图片
            next_reference_path = current_reference_path
            if stage_idx < stages - 1:
                if reference_mode == "last_frame" and result.get("original_last_frame_path"):
                    next_reference_path = result["original_last_frame_path"]
            
            stage_result = {
                "stage": stage_idx + 1,
                "success": True,
                "transition_frames": result["transition_count"],
                "idle_frames": result["idle_count"],
                "total_frames": result["stage_frame_count"],
                "video_path": video_path,
                "last_frame_path": result.get("original_last_frame_path")
            }
            
            yield None, None, None, latest_video_path, f"[阶段 {stage_idx + 1}/{stages}] ✓ 完成! {result['transition_count']} 过渡帧 + {result['idle_count']} 循环帧"
            
            # 最后 yield 结果字典 (不是元组)
            yield {
                "stage_result": stage_result,
                "next_reference_path": next_reference_path,
                "latest_video_path": latest_video_path,
                "stage_frame_count": result["stage_frame_count"]
            }
            
        except Exception as e:
            error_msg = f"[阶段 {stage_idx + 1}] 错误: {str(e)}"
            yield None, None, None, latest_video_path, error_msg
            # 错误情况也 yield 结果
            yield {"stage_result": {"stage": stage_idx + 1, "success": False, "error": str(e)}}
    
    def _crop_reference_image(
        self, 
        reference_image: Image.Image, 
        stage_idx: int, 
        stages: int,
        ORIGINAL_WIDTH: int,
        ORIGINAL_HEIGHT: int
    ) -> Tuple[Image.Image, float]:
        """根据阶段裁剪参考图片"""
        ref_width, ref_height = reference_image.size
        
        min_height = min(300, ORIGINAL_HEIGHT)
        max_height = ORIGINAL_HEIGHT
        
        if stages == 1:
            target_height = max_height
            progress = 1.0
        else:
            progress = stage_idx / (stages - 1)
            target_height = int(min_height + (max_height - min_height) * progress)
        
        if target_height < ref_height:
            crop_top = ref_height - target_height
            crop_box = (0, crop_top, ref_width, ref_height)
            reference_image = reference_image.crop(crop_box)
            print(f"[PlantGenerator] Stage {stage_idx + 1}: Cropped to bottom {target_height}px")
        
        return reference_image, progress
    
    def _extract_and_process_frames(
        self,
        video_path: str,
        stage_idx: int,
        stages: int,
        plant_id: str,
        stage_dir: str,
        final_output_dir: str,
        frames_per_stage: int,
        target_width: int,
        tolerance: int,
        auto_crop: bool,
        crop_padding: int,
        ORIGINAL_WIDTH: int,
        ORIGINAL_HEIGHT: int,
        duration: int,
        all_preview_frames: List,
        latest_video_path: str
    ) -> Optional[Dict]:
        """提取并处理帧"""
        
        # 获取视频信息
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = frame_count_cv / fps if fps > 0 else duration
        cap.release()
        
        # 计算分割点
        transition_end_time = video_duration * (2/3)
        frame_duration = 1.0 / fps if fps > 0 else 0.033
        idle_start_time = transition_end_time + frame_duration
        
        # 提取帧
        transition_frames = extract_frames_from_video_segment(video_path, 0.3, transition_end_time, frames_per_stage)
        idle_frames = extract_frames_from_video_segment(video_path, idle_start_time, 0.0, frames_per_stage)
        
        if not transition_frames and not idle_frames:
            return None
        
        # 处理帧尺寸
        transition_frames = process_frames_to_target_size(transition_frames, ORIGINAL_WIDTH, ORIGINAL_HEIGHT)
        idle_frames = process_frames_to_target_size(idle_frames, ORIGINAL_WIDTH, ORIGINAL_HEIGHT)
        
        # 保存原始帧
        transition_frames_dir = os.path.join(stage_dir, "transition_frames")
        idle_frames_dir = os.path.join(stage_dir, "idle_frames")
        save_individual_frames(transition_frames, output_dir=transition_frames_dir)
        save_individual_frames(idle_frames, output_dir=idle_frames_dir)
        
        # 获取最后一帧路径
        idle_frame_files = sorted([f for f in os.listdir(idle_frames_dir) if f.endswith('.png')])
        transition_frame_files = sorted([f for f in os.listdir(transition_frames_dir) if f.endswith('.png')])
        
        original_last_frame_path = None
        if idle_frame_files:
            original_last_frame_path = os.path.join(idle_frames_dir, idle_frame_files[-1])
        elif transition_frame_files:
            original_last_frame_path = os.path.join(transition_frames_dir, transition_frame_files[-1])
        
        # 去除背景
        transition_nobg_dir = os.path.join(stage_dir, "transition_nobg")
        idle_nobg_dir = os.path.join(stage_dir, "idle_nobg")
        
        process_directory(transition_frames_dir, output_dir=transition_nobg_dir, tolerance=tolerance,
                         num_workers=None, auto_crop=auto_crop, crop_padding=crop_padding)
        process_directory(idle_frames_dir, output_dir=idle_nobg_dir, tolerance=tolerance,
                         num_workers=None, auto_crop=auto_crop, crop_padding=crop_padding)
        
        # 缩放帧
        transition_resized = FrameProcessor.resize_frames_to_width(transition_nobg_dir, target_width)
        idle_resized = FrameProcessor.resize_frames_to_width(idle_nobg_dir, target_width)
        
        # 保存最终帧
        stage_frame_count = 0
        
        for frame_idx, frame in enumerate(transition_resized):
            filename = f"{plant_id}-stage{stage_idx + 1}-transition-frame{frame_idx + 1}.png"
            output_path = os.path.join(final_output_dir, filename)
            frame.save(output_path, "PNG")
            stage_frame_count += 1
            if frame_idx == 0:
                all_preview_frames.append(frame)
        
        for frame_idx, frame in enumerate(idle_resized):
            filename = f"{plant_id}-stage{stage_idx + 1}-idle-frame{frame_idx + 1}.png"
            output_path = os.path.join(final_output_dir, filename)
            frame.save(output_path, "PNG")
            stage_frame_count += 1
            if frame_idx == 0:
                all_preview_frames.append(frame)
        
        return {
            "transition_count": len(transition_resized),
            "idle_count": len(idle_resized),
            "stage_frame_count": stage_frame_count,
            "original_last_frame_path": original_last_frame_path
        }
    
    def _generate_config(
        self, 
        output_base: str, 
        plant_id: str, 
        base_prompt: str, 
        stages: int,
        stage_results: List[Dict],
        final_output_dir: str
    ) -> str:
        """生成配置文件"""
        animations = {}
        for res in stage_results:
            if res["success"]:
                stage_num = res['stage']
                animations[f"stage{stage_num}_transition"] = {
                    "frames": res.get('transition_frames', 0),
                    "loop": False,
                    "description": f"Stage {stage_num} transition animation"
                }
                animations[f"stage{stage_num}_idle"] = {
                    "frames": res.get('idle_frames', 0),
                    "loop": True,
                    "description": f"Stage {stage_num} idle animation"
                }
        
        config = {
            "item_id": plant_id,
            "display_name": plant_id.replace("_", " ").title(),
            "description": base_prompt,
            "preset_type": "plant",
            "growth_stages": stages,
            "animations_per_stage": 2,
            "total_animations": stages * 2,
            "texture_directory": final_output_dir.replace("\\", "/"),
            "animations": animations,
            "naming_format": {
                "transition": "{plant_id}-stage{X}-transition-frame{Y}.png",
                "idle": "{plant_id}-stage{X}-idle-frame{Y}.png"
            },
            "parameters": {
                "plant_type": "草",
                "lifespan": 100.0,
                "has_fruit": False
            },
            "collision": {
                "enabled": True,
                "type": "circle"
            }
        }
        
        config_path = os.path.join(output_base, f"{plant_id}_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        return config_path
    
    def _generate_metadata(
        self,
        output_base: str,
        plant_id: str,
        timestamp: str,
        stages: int,
        total_frames: int,
        target_width: int,
        base_prompt: str,
        stage_results: List[Dict],
        all_video_paths: List[str]
    ):
        """生成元数据文件"""
        metadata_path = os.path.join(output_base, "metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(f"=== Plant Stage Generation / 植物阶段生成 ===\n\n")
            f.write(f"Plant ID / 植物ID: {plant_id}\n")
            f.write(f"Generation Time / 生成时间: {timestamp}\n")
            f.write(f"Total Stages / 总阶段数: {stages}\n")
            f.write(f"Total Animations / 总动画数: {stages * 2}\n")
            f.write(f"Total Frames / 总帧数: {total_frames}\n")
            f.write(f"Target Width / 目标宽度: {target_width}px\n")
            f.write(f"Base Prompt / 基础提示词: {base_prompt}\n\n")
            f.write(f"=== Stage Results / 阶段结果 ===\n")
            for res in stage_results:
                if res["success"]:
                    f.write(f"Stage {res['stage']}: OK\n")
                    f.write(f"  Transition frames: {res.get('transition_frames', 0)}\n")
                    f.write(f"  Idle frames: {res.get('idle_frames', 0)}\n")
                    f.write(f"  Total: {res.get('total_frames', 0)} frames\n")
                else:
                    f.write(f"Stage {res['stage']}: FAILED - {res.get('error', 'Unknown')}\n")
            
            f.write(f"\n=== All Videos / 所有视频 ===\n")
            for vp in all_video_paths:
                f.write(f"  {vp}\n")
    
    def _generate_summary(
        self,
        plant_id: str,
        stages: int,
        stage_results: List[Dict],
        total_frames: int,
        target_width: int,
        all_video_paths: List[str],
        output_base: str,
        videos_dir: str,
        final_output_dir: str,
        config_path: str
    ) -> str:
        """生成摘要"""
        successful_stages = sum(1 for r in stage_results if r["success"])
        total_transition_frames = sum(r.get('transition_frames', 0) for r in stage_results if r["success"])
        total_idle_frames = sum(r.get('idle_frames', 0) for r in stage_results if r["success"])
        
        video_list_str = ""
        for i, vp in enumerate(all_video_paths):
            video_list_str += f"  阶段 {i+1}: {os.path.basename(vp)}\n"
        
        lang = get_current_language()
        
        if lang == "zh":
            return f"""[OK] 植物生成完成!

植物ID: {plant_id}
成功阶段: {successful_stages}/{stages}
总动画数: {successful_stages * 2} (每阶段: 过渡+循环)
总帧数: {total_frames}
  - 过渡动画帧: {total_transition_frames}
  - 循环动画帧: {total_idle_frames}
目标宽度: {target_width}px

=== 生成的视频 ===
{video_list_str}
=== 输出路径 ===
输出目录: {output_base}
视频目录: {videos_dir}
最终帧目录: {final_output_dir}
配置文件: {config_path}

可直接用于 Godot AIItemLoader!
"""
        else:
            return f"""[OK] Plant generation complete!

Plant ID: {plant_id}
Successful stages: {successful_stages}/{stages}
Total animations: {successful_stages * 2}
Total frames: {total_frames}
  - Transition frames: {total_transition_frames}
  - Idle frames: {total_idle_frames}
Target width: {target_width}px

=== Generated Videos ===
{video_list_str}
=== Output Paths ===
Output directory: {output_base}
Videos directory: {videos_dir}
Final frames directory: {final_output_dir}
Config file: {config_path}

Ready for Godot AIItemLoader!
"""


# 便捷函数
def generate_plant_stages(image, ref_images, prompt, plant_id, stages, target_width, frames_per_stage,
                          tolerance, auto_crop, crop_padding, model_name, duration,
                          backend="gemini", resolution="720p", reference_mode="last_frame"):
    """便捷函数：生成植物各生长阶段"""
    generator = PlantGenerator()
    return generator.generate(
        image, ref_images, prompt, plant_id, stages, target_width, frames_per_stage,
        tolerance, auto_crop, crop_padding, model_name, duration,
        backend, resolution, reference_mode
    )
