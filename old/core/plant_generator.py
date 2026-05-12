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
import numpy as np
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
from .image_grid_generator import ImageGridGenerator

# 导入VLM碰撞检测模块
import sys
test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test")
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)
from vlm_colide import DoubaoVLMCollisionGenerator


def generate_grid_animation_prompt(base_prompt: str) -> str:
    """
    生成4格网格动画的提示词
    
    Args:
        base_prompt: 基础植物描述
    
    Returns:
        4格网格动画的完整提示词
    """
    return f"""
2x2 grid animation of {base_prompt} growth stages.

TOP-LEFT: Tiny seed sprouting
TOP-RIGHT: Small seedling, thin stem
BOTTOM-LEFT: Medium plant, branches
BOTTOM-RIGHT: Full mature tree, dense foliage

All 4 animate together with gentle sway. White background, static. No shadows.
Size order: TOP-LEFT smallest → BOTTOM-RIGHT largest.
- Clean flat colors, clear outlines
- NO shadows, NO depth effects
- Show complete plant from bottom to top in each panel
- Each panel must be clearly distinct in plant size


REMEMBER: 4 separate side-by-side growth stages, NOT one continuous scene!
"""


class PlantGenerator:
    """
    植物生成器
    生成植物各生长阶段的动画帧
    """
    
    def __init__(self):
        self._api_manager = get_api_manager()
        self._frame_processor = FrameProcessor()
        self._grid_generator = ImageGridGenerator()
        self._progress_lock = threading.Lock()
        self._stage_status = {}  # 用于跟踪每个阶段的状态
        self._collision_preview_path = None  # 碰撞预览图路径
    
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
    
    def generate_from_grid(
        self,
        image_path: str,
        plant_id: str,
        plant_prompt: str = "",
        target_width: int = 512,
        frames_per_stage: int = 24,
        tolerance: int = 30,
        auto_crop: bool = True,
        crop_padding: int = 10,
        model_name: str = "doubao-seedance-1-0-pro-250528",
        duration: int = 4,
        backend: str = "seedance",
        resolution: str = "720p",
        ark_api_key: str = ""
    ) -> Generator[Tuple[Optional[str], Optional[List], Optional[str], Optional[str], str], None, None]:
        """
        使用网格工作流生成植物阶段
        1. 将输入图片转换为4格网格图
        2. 用网格图生成视频动画
        3. 提取视频帧
        4. 切割每帧为4个阶段（左上→stage1, 右上→stage2, 左下→stage3, 右下→stage4）
        
        Args:
            image_path: 输入图片路径（1:1图片）
            plant_id: 植物ID
            plant_prompt: 植物描述
            target_width: 目标宽度（默认512）
            frames_per_stage: 每阶段帧数（默认24）
            tolerance: 背景去除容差
            auto_crop: 是否自动裁剪
            crop_padding: 裁剪边距
            model_name: 视频生成模型
            duration: 视频时长
            backend: 后端
            resolution: 分辨率
            ark_api_key: ARK API密钥（用于碰撞检测）
            
        Yields:
            (preview_path, preview_frames, config_path, collision_preview_path, status_message)
        """
        if not plant_id or plant_id.strip() == "":
            yield None, None, None, None, "错误：请输入植物ID / Error: Please enter plant ID"
            return
        
        plant_id = plant_id.strip().replace(" ", "_").lower()
        base_prompt = plant_prompt if plant_prompt and plant_prompt.strip() else f"a {plant_id.replace('_', ' ')}"
        
        try:
            # 创建输出目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = os.path.join(OUTPUT_DIR, f"plant_{plant_id}_{timestamp}")
            os.makedirs(output_base, exist_ok=True)
            
            final_output_dir = os.path.join(output_base, "final_frames")
            videos_dir = os.path.join(output_base, "videos")
            os.makedirs(final_output_dir, exist_ok=True)
            os.makedirs(videos_dir, exist_ok=True)
            
            # 步骤1: 生成4格网格图
            status_msg = "[1/4] 正在生成4格网格图... / Generating 4-cell grid..."
            yield None, None, None, None, status_msg
            
            grid_img = self._grid_generator._create_plant_grid(image_path)
            grid_path = os.path.join(output_base, f"{plant_id}_grid.png")
            grid_img.save(grid_path)
            
            # 步骤2: 用网格图生成视频
            status_msg = f"[2/4] 正在生成动画视频 (后端: {backend})... / Generating animation video..."
            yield None, None, None, None, status_msg
            
            # 生成4格网格动画提示词
            grid_prompt = generate_grid_animation_prompt(base_prompt)
            
            # 加载网格图
            grid_image = load_reference_image(grid_path)
            
            # 调用视频生成API
            video_backend = self._api_manager.video_backend
            video_result = video_backend.generate_video(
                reference_image=grid_image,
                prompt=grid_prompt,
                model_name=model_name,
                duration=duration,
                resolution=resolution,
                reference_images=None
            )
            
            # 保存视频
            video_filename = f"{plant_id}_grid_animation.mp4"
            video_path = os.path.join(videos_dir, video_filename)
            
            if video_result.video_data:
                with open(video_path, "wb") as f:
                    f.write(video_result.video_data)
            else:
                shutil.copy(video_result.video_path, video_path)
            
            # 步骤3: 提取视频帧
            status_msg = f"[3/4] 正在提取视频帧... / Extracting video frames..."
            yield None, None, None, None, status_msg
            
            # 提取所有帧
            extracted_frames = extract_frames_from_video_segment(
                video_path, 
                start_time=0.5,
                end_time=0.0,  # 0表示到结尾
                max_frames=frames_per_stage
            )
            
            if not extracted_frames:
                yield None, None, None, None, "❌ 视频帧提取失败 / Failed to extract frames"
                return
            
            # 步骤4: 切割每帧为4个阶段并去除背景
            status_msg = f"[4/4] 正在处理帧... / Processing frames..."
            yield None, None, None, None, status_msg
            
            stage_results = []
            all_preview_frames = []
            
            # 计算transition和idle的分界点：前2/3是transition
            total_frames = len(extracted_frames)
            transition_frame_count = int(total_frames * 2 / 3)
            
            # 处理每一帧
            for frame_idx, frame in enumerate(extracted_frames, start=1):
                # frame 已经是 PIL Image 对象
                frame_img = frame
                frame_width, frame_height = frame_img.size
                
                # 判断当前帧是transition还是idle
                if frame_idx <= transition_frame_count:
                    anim_type = "transition"
                    anim_frame_idx = frame_idx
                else:
                    anim_type = "idle"
                    anim_frame_idx = frame_idx - transition_frame_count
                
                # 计算单元格尺寸（均分为2x2网格）
                cell_width = frame_width // 2
                cell_height = frame_height // 2
                
                # 切割为4个格子
                # 左上 = Stage 1, 右上 = Stage 2, 左下 = Stage 3, 右下 = Stage 4
                cells = [
                    (0, 0, cell_width, cell_height, 1),                           # 左上 → Stage 1
                    (cell_width, 0, frame_width, cell_height, 2),                 # 右上 → Stage 2
                    (0, cell_height, cell_width, frame_height, 3),                # 左下 → Stage 3
                    (cell_width, cell_height, frame_width, frame_height, 4)       # 右下 → Stage 4
                ]
                
                for x1, y1, x2, y2, stage_num in cells:
                    cell_img = frame_img.crop((x1, y1, x2, y2))
                    
                    # 根据阶段设置缩放比例（在原始分辨率下缩小）
                    if stage_num == 1:
                        # Stage 1 (种子): 缩小到40%
                        scale_ratio = 0.4
                    elif stage_num == 2:
                        # Stage 2 (幼苗): 缩小到60%
                        scale_ratio = 0.6
                    else:
                        # Stage 3, 4: 保持原尺寸
                        scale_ratio = 1.0
                    
                    # 获取原始cell尺寸
                    original_cell_w, original_cell_h = cell_img.size
                    
                    if scale_ratio < 1.0:
                        # 缩小植物：先缩小图片，然后放在白色画布底部
                        scaled_w = int(original_cell_w * scale_ratio)
                        scaled_h = int(original_cell_h * scale_ratio)
                        cell_img_small = cell_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
                        
                        # 创建白色画布
                        canvas = Image.new('RGB', (original_cell_w, original_cell_h), (255, 255, 255))
                        
                        # 将缩小的图片放在底部居中
                        paste_x = (original_cell_w - scaled_w) // 2
                        paste_y = original_cell_h - scaled_h
                        canvas.paste(cell_img_small, (paste_x, paste_y))
                        cell_img = canvas
                    
                    # 调整到目标宽度（放大）
                    cell_img = cell_img.resize((target_width, target_width), Image.Resampling.LANCZOS)
                    
                    # 去除背景（白色）
                    cell_img = cell_img.convert("RGBA")
                    datas = cell_img.getdata()
                    newData = []
                    for item in datas:
                        # 检测白色或接近白色的像素
                        if item[0] > 240 and item[1] > 240 and item[2] > 240:
                            newData.append((255, 255, 255, 0))  # 透明
                        else:
                            newData.append(item)
                    cell_img.putdata(newData)
                    
                    # 保存到 final_frames 根目录，按transition/idle分类命名
                    filename = f"{plant_id}-stage{stage_num}-{anim_type}-frame{anim_frame_idx}.png"
                    cell_path = os.path.join(final_output_dir, filename)
                    cell_img.save(cell_path)
                    
                    # 第一帧加入预览
                    if frame_idx == 1:
                        all_preview_frames.append(cell_path)
            
            # 记录结果
            for stage_num in range(1, 5):
                stage_results.append({
                    'stage': stage_num,
                    'frames': len(extracted_frames),
                    'output_dir': final_output_dir
                })
            
            yield None, all_preview_frames, None, None, f"✓ 已处理 {len(extracted_frames)} 帧到4个阶段"
            
            # 步骤5: 生成配置文件
            status_msg = "生成配置文件... / Generating config..."
            yield None, all_preview_frames, None, None, status_msg
            
            config_path = self._generate_grid_config(
                output_base=output_base,
                plant_id=plant_id,
                stage_results=stage_results,
                final_output_dir=final_output_dir,
                grid_path=grid_path
            )
            
            # 步骤6: 生成碰撞配置（如果提供了API密钥）
            collision_preview_path = None
            if ark_api_key and ark_api_key.strip():
                status_msg = "生成碰撞配置... / Generating collision config..."
                yield None, all_preview_frames, config_path, None, status_msg
                
                # 使用网格工作流模式生成碰撞配置
                if self._generate_collision_config(plant_id, final_output_dir, 3, ark_api_key, workflow="grid"):
                    collision_preview_path = os.path.join(final_output_dir, f"{plant_id}_collision_preview.png")
                    self._collision_preview_path = collision_preview_path
            
            # 步骤7: 生成元数据
            status_msg = "生成元数据... / Generating metadata..."
            yield None, all_preview_frames, config_path, collision_preview_path, status_msg
            
            self._generate_grid_metadata(
                output_base=output_base,
                plant_id=plant_id,
                timestamp=timestamp,
                total_frames=len(extracted_frames) * 4,  # 4个阶段
                target_width=target_width,
                stage_results=stage_results,
                grid_path=grid_path
            )
            
            # 完成
            total_frames = len(extracted_frames) * 4
            status_msg = f"✅ 完成！共生成 {total_frames} 帧（{len(extracted_frames)}帧/阶段 × 4阶段）/ Completed! Generated {total_frames} frames"
            yield None, all_preview_frames, config_path, collision_preview_path, status_msg
            
        except Exception as e:
            error_msg = f"❌ 错误 / Error: {str(e)}\n{traceback.format_exc()}"
            yield None, None, None, None, error_msg
    
    def _generate_grid_config(
        self,
        output_base: str,
        plant_id: str,
        stage_results: List[Dict],
        final_output_dir: str,
        grid_path: str
    ) -> str:
        """生成网格工作流的配置文件"""
        # 获取每个阶段的帧数（假设所有阶段帧数相同）
        frames_per_stage = stage_results[0]['frames'] if stage_results else 24
        
        # 计算transition和idle帧数：前2/3是transition，后1/3是idle
        transition_frames = int(frames_per_stage * 2 / 3)
        idle_frames = frames_per_stage - transition_frames
        
        config = {
            "item_id": plant_id,
            "display_name": plant_id.capitalize(),
            "description": plant_id,
            "preset_type": "plant",
            "growth_stages": 4,
            "animations_per_stage": 2,
            "total_animations": 8,
            "texture_directory": f"res://Assets/Items/plants/{plant_id}/final_frames",
            "animations": {
                "stage1_transition": {
                    "frames": transition_frames,
                    "loop": False,
                    "description": "Stage 1 transition animation (play once on stage change)"
                },
                "stage1_idle": {
                    "frames": idle_frames,
                    "loop": True,
                    "description": "Stage 1 idle animation (loop continuously)"
                },
                "stage2_transition": {
                    "frames": transition_frames,
                    "loop": False,
                    "description": "Stage 2 transition animation (play once on stage change)"
                },
                "stage2_idle": {
                    "frames": idle_frames,
                    "loop": True,
                    "description": "Stage 2 idle animation (loop continuously)"
                },
                "stage3_transition": {
                    "frames": transition_frames,
                    "loop": False,
                    "description": "Stage 3 transition animation (play once on stage change)"
                },
                "stage3_idle": {
                    "frames": idle_frames,
                    "loop": True,
                    "description": "Stage 3 idle animation (loop continuously)"
                },
                "stage4_transition": {
                    "frames": transition_frames,
                    "loop": False,
                    "description": "Stage 4 transition animation (play once on stage change)"
                },
                "stage4_idle": {
                    "frames": idle_frames,
                    "loop": True,
                    "description": "Stage 4 idle animation (loop continuously)"
                }
            },
            "naming_format": {
                "transition": "{plant_id}-stage{X}-transition-frame{Y}.png",
                "idle": "{plant_id}-stage{X}-idle-frame{Y}.png"
            },
            "parameters": {
                "plant_type": "植物",
                "lifespan": 10.0,
                "has_fruit": False
            },
            "collision": {
                "enabled": True,
                "type": "grid",
                "layer": 8,
                "mask": 1
            }
        }
        
        config_path = os.path.join(final_output_dir, f"{plant_id}_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return config_path
    
    def _generate_grid_metadata(
        self,
        output_base: str,
        plant_id: str,
        timestamp: str,
        total_frames: int,
        target_width: int,
        stage_results: List[Dict],
        grid_path: str
    ):
        """生成网格工作流的元数据"""
        metadata = {
            "plant_id": plant_id,
            "workflow": "grid",
            "timestamp": timestamp,
            "total_stages": 4,
            "total_frames": total_frames,
            "target_width": target_width,
            "grid_image": grid_path,
            "stages": stage_results
        }
        
        metadata_path = os.path.join(output_base, "metadata.txt")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(f"Plant ID: {plant_id}\n")
            f.write(f"Workflow: Grid-based\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Stages: 4\n")
            f.write(f"Total Frames: {total_frames}\n")
            f.write(f"Target Width: {target_width}px\n")
            f.write(f"Grid Image: {grid_path}\n")
            f.write("\nStage Details:\n")
            for stage in stage_results:
                f.write(f"  Stage {stage['stage']}: {stage['frames']} frames\n")
    
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
        backend: str = "seedance",
        resolution: str = "720p",
        reference_mode: str = "last_frame",
        ark_api_key: str = ""
    ) -> Generator[Tuple[Optional[str], Optional[List], Optional[str], Optional[str], Optional[str], str], None, None]:
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
            ark_api_key: ARK API密钥(用于VLM碰撞检测)
        
        Yields:
            (preview_path, preview_frames, config_path, video_paths, collision_preview_path, status_message)
        """
        # 验证后端
        error_msg = self._api_manager.validate_backend(backend)
        if error_msg:
            yield None, None, None, None, None, error_msg
            return
        
        # 验证输入
        if image is None and (not prompt or prompt.strip() == ""):
            yield None, None, None, None, None, "[ERROR] " + t("plant_need_input")
            return
        
        if not plant_id or plant_id.strip() == "":
            yield None, None, None, None, None, "[ERROR] " + t("plant_need_id")
            return
        
        plant_id = plant_id.strip().replace(" ", "_").lower()
        
        try:
            self._clean_old_outputs("plant")
            
            yield None, None, None, None, None, f"正在初始化 (后端: {backend})... / Initializing (backend: {backend})..."
            
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
                yield None, None, None, None, None, "[ERROR] Failed to prepare reference image"
                return
            
            yield None, None, None, None, None, "参考图片已准备好 / Reference image ready"
            
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
            yield None, None, None, None, None, f"开始并行生成 {stages} 个阶段的视频 (后端: {backend})..."
            
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
                    
                    yield None, None, None, None, None, status_msg
            
            # 按阶段顺序排序视频结果
            video_results.sort(key=lambda x: x["stage_idx"])
            
            # 检查是否所有视频都成功
            failed_stages = [r for r in video_results if not r["success"]]
            if failed_stages:
                error_msg = f"有 {len(failed_stages)} 个阶段视频生成失败\n"
                for r in failed_stages:
                    error_msg += f"  Stage {r['stage_idx'] + 1}: {r['error']}\n"
                yield None, None, None, None, None, error_msg
            
            # ===== 第二步: 串行处理每个视频的帧提取 =====
            yield None, None, None, None, None, "所有视频生成完成，开始处理帧..."
            
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
                
                yield None, None, None, None, None, f"[阶段 {stage_idx + 1}/{stages}] 提取并处理帧..."
                
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
                
                yield None, None, None, None, None, f"[阶段 {stage_idx + 1}/{stages}] ✓ 完成! {frame_result['transition_count']} 过渡帧 + {frame_result['idle_count']} 循环帧"
                
                # ===== 最后阶段生成碰撞配置 =====
                if stage_idx == stages - 1:
                    yield None, None, None, None, None, f"[阶段 {stage_idx + 1}] 正在生成碰撞体积配置..."
                    collision_success = self._generate_collision_config(
                        plant_id=plant_id,
                        final_output_dir=final_output_dir,
                        stage_idx=stage_idx,
                        ark_api_key=ark_api_key,
                        workflow="video"
                    )
                    if collision_success:
                        yield None, None, None, None, None, f"[阶段 {stage_idx + 1}] ✓ 碰撞体积配置已生成"
                    else:
                        yield None, None, None, None, None, f"[阶段 {stage_idx + 1}] ⚠ 碰撞体积配置生成失败"
            
            yield None, None, None, None, None, "正在生成配置文件... / Generating config file..."
            
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
            
            # 获取碰撞预览图路径
            collision_preview_path = None
            if self._collision_preview_path and os.path.exists(self._collision_preview_path):
                collision_preview_path = os.path.abspath(self._collision_preview_path)
            
            yield (
                os.path.abspath(preview_path) if preview_path else None,
                all_preview_frames[:12] if all_preview_frames else [],
                os.path.abspath(config_path),
                video_paths_for_preview,
                collision_preview_path,
                summary
            )
            
        except Exception as e:
            traceback.print_exc()
            yield None, None, None, None, None, f"[ERROR] {t('error')}: {str(e)}"
    
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
            
            # 加载并裁剪参考图片
            reference_image = load_reference_image(reference_image_path)
            reference_image, progress = self._crop_reference_image(
                reference_image, stage_idx, stages, ORIGINAL_WIDTH, ORIGINAL_HEIGHT
            )
            
            # 构建完整提示词
            full_prompt = generate_grid_animation_prompt(base_prompt)
            
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
        
        try:
            # 加载并裁剪参考图片
            reference_image = load_reference_image(current_reference_path)
            reference_image, progress = self._crop_reference_image(
                reference_image, stage_idx, stages, ORIGINAL_WIDTH, ORIGINAL_HEIGHT
            )
            
            # 构建完整提示词
            full_prompt = generate_grid_animation_prompt(base_prompt)
            
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
    
    def _generate_collision_config(
        self, 
        plant_id: str, 
        final_output_dir: str, 
        stage_idx: int,
        ark_api_key: str = "",
        workflow: str = "video"
    ) -> bool:
        """
        为植物的最后阶段最后一帧生成碰撞配置
        
        Args:
            plant_id: 植物ID
            final_output_dir: 最终帧输出目录
            stage_idx: 当前阶段索引
            ark_api_key: ARK API密钥
            workflow: 工作流类型 ("grid" 或 "video")
        
        Returns:
            是否成功生成碰撞配置
        """
        try:
            # 检查API密钥
            if not ark_api_key or ark_api_key.strip() == "":
                print("[Collision] ⚠ 跳过碰撞生成：未提供 ARK_API_KEY")
                print("[Collision] 提示：在Gradio界面输入ARK API Key可自动生成碰撞配置")
                return False
            
            # 根据工作流类型查找最后一帧
            if workflow == "grid":
                # 网格工作流：在 final_output_dir 下查找 stage4 的 idle 最后一帧
                # 文件名格式: {plant_id}-stage4-idle-frame*.png
                frame_pattern = f"{plant_id}-stage4-idle-frame"
                all_files = [f for f in os.listdir(final_output_dir) 
                            if f.startswith(frame_pattern) and f.endswith('.png')]
                
                if not all_files:
                    print(f"[Collision] 未找到帧文件: {final_output_dir}/{frame_pattern}*.png")
                    return False
                
                # 按帧号数值排序（从文件名提取帧号）
                def extract_frame_number(filename):
                    # 从 "{plant_id}-stage4-idle-frame{N}.png" 提取 N
                    try:
                        return int(filename.replace(frame_pattern, "").replace(".png", ""))
                    except:
                        return 0
                
                all_files_sorted = sorted(all_files, key=extract_frame_number)
                last_frame_filename = all_files_sorted[-1]
                last_frame_path = os.path.join(final_output_dir, last_frame_filename)
                
                print(f"[Collision] 找到 {len(all_files_sorted)} 个idle帧，使用最后一帧: {last_frame_filename}")
            else:
                # 视频工作流：查找 idle-frame
                idle_frame_pattern = f"{plant_id}-stage{stage_idx + 1}-idle-frame"
                all_files = sorted([f for f in os.listdir(final_output_dir) 
                                   if f.startswith(idle_frame_pattern) and f.endswith('.png')])
                
                if not all_files:
                    print(f"[Collision] 未找到idle帧: {idle_frame_pattern}*.png")
                    return False
                
                last_frame_filename = all_files[-1]
                last_frame_path = os.path.join(final_output_dir, last_frame_filename)
            
            print(f"[Collision] 正在为最后一帧生成碰撞配置: {last_frame_filename}")
            
            # 初始化VLM碰撞检测器
            detector = DoubaoVLMCollisionGenerator(api_key=ark_api_key)
            
            # 生成碰撞配置 (6x6网格)
            collision_config = detector.generate_collision_config(
                image_path=last_frame_path,
                item_type="plant"
            )
            
            # 检查是否有错误
            if "error" in collision_config:
                print(f"[Collision] VLM检测失败: {collision_config.get('error')}")
                return False
            
            # 确保是6x6网格配置
            if "selected_grids" not in collision_config:
                print(f"[Collision] 配置格式错误，缺少selected_grids字段")
                return False
            
            # 确定Godot物品目录路径
            # 假设Godot项目在SnowWeave的同级目录
            current_dir = os.path.dirname(os.path.dirname(__file__))
            godot_project_dir = os.path.join(os.path.dirname(current_dir), "SnowGlobe", "snow-globe")
            godot_items_dir = os.path.join(godot_project_dir, "Assets", "Items", "plants", plant_id)
            
            # 创建目录(如果不存在)
            os.makedirs(godot_items_dir, exist_ok=True)
            
            # 保存碰撞配置
            collision_json_path = os.path.join(godot_items_dir, f"{plant_id}_collision.json")
            with open(collision_json_path, 'w', encoding='utf-8') as f:
                json.dump(collision_config, f, indent=2, ensure_ascii=False)
            
            print(f"[Collision] ✓ 碰撞配置已保存: {collision_json_path}")
            
            # 同时在输出目录也保存一份备份
            backup_path = os.path.join(final_output_dir, f"{plant_id}_collision.json")
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(collision_config, f, indent=2, ensure_ascii=False)
            
            print(f"[Collision] ✓ 备份已保存: {backup_path}")
            
            # 生成可视化预览图
            visualization_path = os.path.join(final_output_dir, f"{plant_id}_collision_preview.png")
            try:
                detector.visualize_collision(
                    image_path=last_frame_path,
                    config=collision_config,
                    output_path=visualization_path
                )
                print(f"[Collision] ✓ 可视化预览已生成: {visualization_path}")
                # 保存到实例变量供后续返回
                self._collision_preview_path = visualization_path
            except Exception as vis_error:
                print(f"[Collision] ⚠ 可视化生成失败: {vis_error}")
            
            return True
            
        except Exception as e:
            print(f"[Collision] 生成碰撞配置时出错: {e}")
            traceback.print_exc()
            return False
    
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
                    "description": f"Stage {stage_num} transition animation (play once on stage change)"
                }
                animations[f"stage{stage_num}_idle"] = {
                    "frames": res.get('idle_frames', 0),
                    "loop": True,
                    "description": f"Stage {stage_num} idle animation (loop continuously)"
                }
        
        config = {
            "item_id": plant_id,
            "display_name": plant_id.replace("_", " ").title(),
            "description": base_prompt,
            "preset_type": "plant",
            "growth_stages": stages,
            "animations_per_stage": 2,
            "total_animations": stages * 2,
            "texture_directory": f"res://Assets/Items/plants/{plant_id}/final_frames",
            "animations": animations,
            "naming_format": {
                "transition": "{plant_id}-stage{X}-transition-frame{Y}.png",
                "idle": "{plant_id}-stage{X}-idle-frame{Y}.png"
            },
            "parameters": {
                "plant_type": "草",
                "lifespan": 10.0,
                "has_fruit": False
            },
            "collision": {
                "enabled": True,
                "type": "grid",
                "layer": 8,
                "mask": 1
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
