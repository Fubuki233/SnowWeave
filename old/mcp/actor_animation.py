"""
Actor Animation Pipeline (MCP Standalone)
角色动画生成完整流水线（MCP 独立版）

流程：
1. 将角色图片放入网格模板
2. 使用 Seedream 4.5 生成四视角图片
3. 切分为四张独立图片
4. 使用 Seedance 1.5 为每个视角生成动画视频
5. 从视频中提取帧并保存

依赖文件（同目录下）：
- seedream_backend.py: Seedream 图片生成后端
- seedance_backend.py: Seedance 视频生成后端
- frame_extractor.py: 视频帧提取工具
"""

import os
import shutil
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入本地工具模块（支持直接运行和作为模块导入两种方式）
try:
    from .seedream_backend import SeedreamBackend, ImageResult
    from .seedance_backend import SeedanceBackend, VideoResult
    from .frame_extractor import (
        extract_frames_from_video, save_frames, create_sprite_sheet,
        remove_background, remove_background_advanced
    )
except ImportError:
    # 直接运行脚本时使用绝对导入
    from seedream_backend import SeedreamBackend, ImageResult
    from seedance_backend import SeedanceBackend, VideoResult
    from frame_extractor import (
        extract_frames_from_video, save_frames, create_sprite_sheet,
        remove_background, remove_background_advanced
    )


# ============ 常量配置 ============

# 获取当前文件所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 网格模板路径
MESH_TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "assets", "module", "mesh-1024x_512.png")

# 默认输出目录
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

# 视角名称
VIEW_NAMES = {
    "top_left": "front",      # 左上 - 正面/主视图
    "top_right": "back",      # 右上 - 背面/后视图
    "bottom_left": "left",    # 左下 - 左视图
    "bottom_right": "right",  # 右下 - 右视图
}

# 网格布局（1024x512，每个网格 512x512）
GRID_SIZE = 512  # 每个网格的尺寸
GRID_COLS = 2    # 列数
GRID_ROWS = 1    # 行数（实际是 1024x512，但是4个网格是 2x2 排列）

# Seedream 4.5 要求最小 3686400 像素（约 1920x1920）
# 使用 1920x1920 作为输出尺寸，每个网格 960x960
# 2x2 网格布局
ACTUAL_GRID_SIZE = 960
ACTUAL_GRID_COLS = 2
ACTUAL_GRID_ROWS = 2
ACTUAL_TOTAL_WIDTH = 1920
ACTUAL_TOTAL_HEIGHT = 1920

# Seedream 输出尺寸（必须满足 3686400 像素最小要求）
SEEDREAM_OUTPUT_SIZE = "1920x1920"


@dataclass
class PipelineConfig:
    """流水线配置"""
    api_key: str  # ARK API Key
    character_name: str = "character"  # 角色名称，用于文件命名
    output_dir: str = None  # 输出目录
    
    # 视频生成参数
    video_duration: int = 4  # 视频时长（秒），使用最短时长节省成本
    video_draft: bool = True  # 使用预览模式
    video_generate_audio: bool = False  # 不生成音频
    
    # 帧提取参数
    frames_per_video: int = 24  # 每个视频提取的帧数
    target_fps: int = 16  # 目标帧率
    
    # 抠图参数
    remove_background: bool = True  # 是否移除背景
    bg_method: str = "auto"  # 背景移除方法: "white", "green", "auto", "smart"
    bg_tolerance: int = 25  # 背景色容差（建议15-25，太大会误删人物浅色部分）
    bg_edge_shrink: int = 5  # 边缘内缩像素数（建议0-2，太大会吃掉人物边缘）
    
    # 生成图片/视频时的背景色（"white" 或 "green"，"auto" 自动检测）
    generated_bg_color: str = "auto"  # 生成时使用的背景色
    
    # 跳过步骤参数
    skip_image_gen: bool = False  # 跳过图片生成，直接使用已有的多视角图片
    skip_video_gen: bool = False  # 跳过视频生成，直接使用已有的视频
    
    # 输入路径（当跳过某步骤时使用）
    input_multiview_image: str = None  # 已有的多视角图片路径
    input_videos_dir: str = None  # 已有的视频目录路径


class ActorAnimationPipeline:
    """
    角色动画生成流水线
    
    完整流程：角色图片 -> 多视角图片 -> 动画视频 -> 动画帧
    """
    
    def __init__(self, config: PipelineConfig):
        """
        初始化流水线
        
        Args:
            config: 流水线配置
        """
        self.config = config
        self.api_key = config.api_key
        
        # 设置输出目录
        if config.output_dir:
            self.output_dir = config.output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"{config.character_name}_{timestamp}")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化后端
        self.seedream = SeedreamBackend(api_key=self.api_key, watermark=False)
        self.seedance = SeedanceBackend(
            api_key=self.api_key,
            draft=config.video_draft,
            generate_audio=config.video_generate_audio,
            camera_fixed=True  # 固定相机
        )
        
        # 检测到的背景色（用于生成图片/视频）
        self.detected_bg_color = None
        
        print(f"[Pipeline] Initialized with output dir: {self.output_dir}")
    
    def _analyze_image_colors(self, image: Image.Image) -> str:
        """
        分析图片中的白色和绿色比例，决定使用什么背景色
        
        原则：
        - 如果图片白色多 -> 使用绿色背景
        - 如果图片绿色多 -> 使用白色背景
        - 如果都不多 -> 默认使用白色背景
        
        Args:
            image: PIL Image 对象
            
        Returns:
            "white" 或 "green"
        """
        import numpy as np
        
        # 转换为 RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        total_pixels = img_array.shape[0] * img_array.shape[1]
        
        # 定义颜色阈值
        # 白色: R>200, G>200, B>200
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        
        white_mask = (r > 200) & (g > 200) & (b > 200)
        white_pixels = np.sum(white_mask)
        white_ratio = white_pixels / total_pixels
        
        # 绿色: G > R+30, G > B+30, G > 100
        green_mask = (g > r + 30) & (g > b + 30) & (g > 100)
        green_pixels = np.sum(green_mask)
        green_ratio = green_pixels / total_pixels
        
        print(f"[ColorAnalysis] White pixels: {white_ratio*100:.1f}%, Green pixels: {green_ratio*100:.1f}%")
        
        # 决策逻辑
        if white_ratio > 0.15:  # 白色超过15%
            print(f"[ColorAnalysis] Image has significant white ({white_ratio*100:.1f}%), using GREEN background")
            return "green"
        elif green_ratio > 0.15:  # 绿色超过15%
            print(f"[ColorAnalysis] Image has significant green ({green_ratio*100:.1f}%), using WHITE background")
            return "white"
        else:
            # 默认使用白色背景（更常见）
            print(f"[ColorAnalysis] No dominant white/green, defaulting to WHITE background")
            return "white"
    
    def _load_mesh_template(self) -> Image.Image:
        """
        加载网格模板并缩放到目标尺寸
        Seedream 4.5 要求最小 3686400 像素，所以缩放到 1920x1920
        """
        if not os.path.exists(MESH_TEMPLATE_PATH):
            raise FileNotFoundError(f"Mesh template not found: {MESH_TEMPLATE_PATH}")
        
        template = Image.open(MESH_TEMPLATE_PATH)
        original_size = template.size
        
        # 缩放到目标尺寸 (1920x1920)
        target_size = (ACTUAL_TOTAL_WIDTH, ACTUAL_TOTAL_HEIGHT)
        if template.size != target_size:
            template = template.resize(target_size, Image.Resampling.LANCZOS)
            print(f"[Pipeline] Loaded and resized mesh template: {original_size} -> {template.size}")
        else:
            print(f"[Pipeline] Loaded mesh template: {template.size}")
        
        return template
    
    def _place_character_in_grid(
        self, 
        character_image: Image.Image, 
        template: Image.Image
    ) -> Image.Image:
        """
        将角色图片放入网格模板的左上角
        
        Args:
            character_image: 角色图片
            template: 网格模板
        
        Returns:
            合成后的图片
        """
        # 复制模板
        result = template.copy()
        
        # 获取模板尺寸
        template_width, template_height = template.size
        
        # 计算单个网格的尺寸
        # 假设是 2x2 网格排列
        grid_width = template_width // ACTUAL_GRID_COLS
        grid_height = template_height // ACTUAL_GRID_ROWS
        
        print(f"[Pipeline] Grid size: {grid_width}x{grid_height}")
        
        # 缩放角色图片以适应网格
        char_width, char_height = character_image.size
        
        # 等比缩放到网格大小
        scale = min(grid_width / char_width, grid_height / char_height)
        new_width = int(char_width * scale)
        new_height = int(char_height * scale)
        
        scaled_char = character_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 计算放置位置（左上角网格，居中）
        paste_x = (grid_width - new_width) // 2
        paste_y = (grid_height - new_height) // 2
        
        # 如果角色图片有透明通道，需要特殊处理
        if scaled_char.mode == 'RGBA':
            result.paste(scaled_char, (paste_x, paste_y), scaled_char)
        else:
            result.paste(scaled_char, (paste_x, paste_y))
        
        print(f"[Pipeline] Placed character at ({paste_x}, {paste_y}), size: {new_width}x{new_height}")
        
        return result
    
    def _build_multiview_prompt(self, character_description: str = "", bg_color: str = "white") -> str:
        """
        构建多视角生成提示词（2D游戏精灵风格）
        
        Args:
            character_description: 角色描述（可选）
            bg_color: 背景颜色 ("white" 或 "green")
        
        Returns:
            完整的提示词
        """
        # 根据背景色设置描述
        if bg_color == "green":
            bg_desc = "PURE GREEN BACKGROUND (#00FF00) for ALL 4 views"
            bg_detail = "Character floating on pure bright green (chroma key green)"
        else:
            bg_desc = "PURE WHITE BACKGROUND (#FFFFFF) for ALL 4 views"
            bg_detail = "Character floating on pure white"
        
        base_prompt = f"""Generate a 2x2 grid showing 4 DIFFERENT directional views of the same character for a 2D RPG game sprite sheet.

CRITICAL - CHARACTER ORIENTATION:
- Character is ALWAYS STANDING UPRIGHT (head on TOP, feet on BOTTOM)
- NEVER draw character upside-down or lying down
- Character's head is ALWAYS at the top of each cell
- Character's feet are ALWAYS at the bottom of each cell
- This is a standard game sprite sheet with 4 walking directions

GRID LAYOUT (STRICTLY FOLLOW):
- TOP-LEFT: FRONT view (character facing the viewer, looking at camera)
- TOP-RIGHT: BACK view (character's back facing the viewer, looking away)
- BOTTOM-LEFT: LEFT view (character facing left direction)
- BOTTOM-RIGHT: RIGHT view (character facing right direction)

VIEW REQUIREMENTS:
1. TOP-LEFT (FRONT): Character faces FORWARD toward viewer. We see the face. Standing upright.
2. TOP-RIGHT (BACK): Character faces AWAY from viewer. We see the back of head/body. Standing upright.
3. BOTTOM-LEFT (LEFT): Character faces LEFT. Side view profile facing left. Standing upright.
4. BOTTOM-RIGHT (RIGHT): Character faces RIGHT. Side view profile facing right. Standing upright.

IMPORTANT - DO NOT:
- DO NOT draw character upside-down
- DO NOT draw character lying flat
- DO NOT rotate character 180 degrees
- DO NOT show character from bird's eye view (looking down at top of head)
- ALL characters must be STANDING with HEAD UP and FEET DOWN

ART STYLE:
- 2D game sprite style (RPG Maker, Stardew Valley aesthetic)
- Slight 3/4 perspective is OK but character stays upright
- Chibi / cute proportions
- Clean, simple design suitable for game sprites
- Consistent style across all 4 views

FULL BODY REQUIREMENT:
- ALL 4 views MUST show COMPLETE FULL BODY from head to feet
- Include: head, torso, arms, hands, legs, feet - EVERYTHING
- NO half-body shots, NO cropping
- Same body proportions and scale across all 4 views
- Leave some padding/margin around the character

BACKGROUND:
- {bg_desc}
- NO gradients, NO shadows on background
- NO floor, NO ground plane
- {bg_detail}

STYLE DETAILS:
- Same character design, colors, proportions in ALL 4 views
- Clean flat lighting, minimal shadows
- Character centered in each cell
- Sharp clean edges for easy extraction"""
        
        if character_description:
            base_prompt = f"Character: {character_description}\n\n{base_prompt}"
        
        return base_prompt
    
    def _split_grid_image(self, grid_image: Image.Image, border_width: int = 3) -> Dict[str, Image.Image]:
        """
        将网格图片切分为四张独立图片，并去除边缘线
        
        Args:
            grid_image: 网格图片
            border_width: 边缘线宽度（像素），默认3像素
        
        Returns:
            字典：{视角名称: 图片}
        """
        width, height = grid_image.size
        cell_width = width // ACTUAL_GRID_COLS
        cell_height = height // ACTUAL_GRID_ROWS
        
        print(f"[Pipeline] Splitting grid {width}x{height} into {ACTUAL_GRID_COLS}x{ACTUAL_GRID_ROWS} cells")
        print(f"[Pipeline] Removing {border_width}px border from each cell")
        
        # 边缘裁剪量
        b = border_width
        
        views = {}
        
        # 左上 - 正面 (去除右边和下边的边缘线)
        views["front"] = grid_image.crop((
            b,                    # 左边去掉边缘
            b,                    # 上边去掉边缘
            cell_width - b,       # 右边去掉边缘（与右侧网格交界）
            cell_height - b       # 下边去掉边缘（与下侧网格交界）
        ))
        
        # 右上 - 背面 (去除左边和下边的边缘线)
        views["back"] = grid_image.crop((
            cell_width + b,       # 左边去掉边缘（与左侧网格交界）
            b,                    # 上边去掉边缘
            width - b,            # 右边去掉边缘
            cell_height - b       # 下边去掉边缘（与下侧网格交界）
        ))
        
        # 左下 - 左视图 (去除右边和上边的边缘线)
        views["left"] = grid_image.crop((
            b,                    # 左边去掉边缘
            cell_height + b,      # 上边去掉边缘（与上侧网格交界）
            cell_width - b,       # 右边去掉边缘（与右侧网格交界）
            height - b            # 下边去掉边缘
        ))
        
        # 右下 - 右视图 (去除左边和上边的边缘线)
        views["right"] = grid_image.crop((
            cell_width + b,       # 左边去掉边缘（与左侧网格交界）
            cell_height + b,      # 上边去掉边缘（与上侧网格交界）
            width - b,            # 右边去掉边缘
            height - b            # 下边去掉边缘
        ))
        
        for name, img in views.items():
            print(f"[Pipeline] View '{name}': {img.size}")
        
        return views
    
    def _build_animation_prompt(self, view_name: str, bg_color: str = "white") -> str:
        """
        构建动画生成提示词（星露谷风格俯视角）
        
        Args:
            view_name: 视角名称
            bg_color: 背景颜色 ("white" 或 "green")
        
        Returns:
            动画提示词
        """
        view_descriptions = {
            "front": "FRONT view, facing toward camera, standing upright",
            "back": "BACK view, facing away from camera, standing upright",
            "left": "LEFT side view, facing left direction, standing upright",
            "right": "RIGHT side view, facing right direction, standing upright",
        }
        
        view_desc = view_descriptions.get(view_name, view_name)
        
        # 根据背景色设置描述
        if bg_color == "green":
            bg_desc = "PURE GREEN BACKGROUND (#00FF00)"
            bg_detail = "Character floating on pure bright green (chroma key green)"
        else:
            bg_desc = "PURE WHITE BACKGROUND (#FFFFFF)"
            bg_detail = "Character floating on pure white"
        
        prompt = f"""Create a walking cycle animation for the character ({view_desc}).

CRITICAL - CHARACTER MUST BE UPRIGHT:
- Character is STANDING with HEAD UP and FEET DOWN
- NEVER rotate or flip the character
- Character stays in the same upright position throughout

ART STYLE:
- 2D game sprite style (RPG Maker, Stardew Valley aesthetic)
- Clean, smooth walking animation
- Character always standing upright

CRITICAL - DO NOT DO THESE:
- DO NOT rotate the character body
- DO NOT turn the character around
- DO NOT flip the character upside-down
- DO NOT change the viewing angle
- DO NOT make the character spin
- DO NOT change which direction character faces
- DO NOT move the character across the screen

THE CHARACTER MUST:
- Keep the EXACT same viewing angle throughout the entire video
- Stay in the EXACT same screen position (walking in place)
- Maintain fixed direction - {view_desc}
- Stay STANDING UPRIGHT at all times

WALKING ANIMATION REQUIREMENTS:
- Natural walking cycle with LEGS MOVING
- Left and right legs alternate stepping
- Arms swing naturally with the walk
- Slight body bounce up and down
- Hair and clothing move with the walk
- Smooth, loopable walk cycle
- Character walks IN PLACE (does not move across screen)

BACKGROUND:
- {bg_desc}
- NO shadows on the floor
- NO environment, NO ground
- {bg_detail}

ANIMATION STYLE:
- Full body movement including legs
- Game sprite walk cycle animation
- Clean 2D animation style
- No special effects, no particles"""
        
        return prompt
    
    def run(
        self,
        character_image_path: str,
        character_description: str = "",
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        运行完整流水线
        
        Args:
            character_image_path: 角色图片路径
            character_description: 角色描述（可选）
            progress_callback: 进度回调函数 (step, total_steps, message)
        
        Returns:
            结果字典，包含生成的所有文件路径
        """
        total_steps = 6
        results = {
            "output_dir": self.output_dir,
            "views": {},
            "videos": {},
            "frames": {},
            "sprite_sheets": {},
        }
        
        def report_progress(step: int, message: str):
            print(f"[Pipeline] Step {step}/{total_steps}: {message}")
            if progress_callback:
                progress_callback(step, total_steps, message)
        
        try:
            views = {}  # 存储切分后的视角图片
            
            # ============ 判断是否跳过图片生成 ============
            if self.config.skip_image_gen and self.config.input_multiview_image:
                # 跳过图片生成，直接使用已有的多视角图片
                report_progress(1, "Skipping image generation, loading existing multiview image...")
                report_progress(2, "Skipping Seedream generation...")
                
                multiview_image = Image.open(self.config.input_multiview_image)
                if multiview_image.mode != 'RGBA':
                    multiview_image = multiview_image.convert('RGBA')
                print(f"[Pipeline] Loaded existing multiview image: {self.config.input_multiview_image}")
                print(f"[Pipeline] Image size: {multiview_image.size}")
                
                results["multiview_image"] = self.config.input_multiview_image
                
                # Step 3: 切分视角图片
                report_progress(3, "Splitting multi-view image into individual views...")
                views = self._split_grid_image(multiview_image)
                
            elif self.config.skip_image_gen and self.config.skip_video_gen:
                # 跳过所有生成，直接处理视频
                report_progress(1, "Skipping all generation steps...")
                report_progress(2, "Skipping Seedream generation...")
                report_progress(3, "Skipping view splitting...")
                
            else:
                # 正常流程：生成图片
                # ============ Step 1: 加载和准备图片 ============
                report_progress(1, "Loading character image and mesh template...")
                
                character_image = Image.open(character_image_path)
                if character_image.mode != 'RGBA':
                    character_image = character_image.convert('RGBA')
                print(f"[Pipeline] Character image size: {character_image.size}")
                
                # 分析图片颜色，决定使用什么背景色
                if self.config.generated_bg_color == "auto":
                    self.detected_bg_color = self._analyze_image_colors(character_image)
                else:
                    self.detected_bg_color = self.config.generated_bg_color
                print(f"[Pipeline] Using {self.detected_bg_color.upper()} background for generation")
                
                # 保存原始角色图片
                char_save_path = os.path.join(self.output_dir, f"{self.config.character_name}_original.png")
                character_image.save(char_save_path)
                results["original_image"] = char_save_path
                
                # 加载网格模板
                mesh_template = self._load_mesh_template()
                
                # 将角色放入网格
                grid_with_char = self._place_character_in_grid(character_image, mesh_template)
                
                # 保存合成的网格图片
                grid_input_path = os.path.join(self.output_dir, f"{self.config.character_name}_grid_input.png")
                grid_with_char.save(grid_input_path)
                results["grid_input"] = grid_input_path
                
                # ============ Step 2: 生成多视角图片 ============
                report_progress(2, f"Generating multi-view images with Seedream 4.5 ({self.detected_bg_color} background)...")
                
                multiview_prompt = self._build_multiview_prompt(character_description, self.detected_bg_color)
                
                image_result = self.seedream.generate_multi_view_image(
                    reference_image=grid_with_char,
                    prompt=multiview_prompt,
                    size=SEEDREAM_OUTPUT_SIZE,  # 1920x1920，满足 3686400 像素最小要求
                    timeout=180
                )
                
                # 保存生成的多视角图片
                multiview_path = os.path.join(self.output_dir, f"{self.config.character_name}_multiview.png")
                image_result.image.save(multiview_path)
                results["multiview_image"] = multiview_path
                
                # ============ Step 3: 切分视角图片 ============
                report_progress(3, "Splitting multi-view image into individual views...")
                views = self._split_grid_image(image_result.image)
            
            # 保存切分后的视角图片
            if views:
                views_dir = os.path.join(self.output_dir, "views")
                os.makedirs(views_dir, exist_ok=True)
                
                for view_name, view_image in views.items():
                    view_path = os.path.join(views_dir, f"{self.config.character_name}_{view_name}.png")
                    view_image.save(view_path)
                    results["views"][view_name] = view_path
                    print(f"[Pipeline] Saved view: {view_path}")
            
            # ============ 判断是否跳过视频生成 ============
            videos_dir = os.path.join(self.output_dir, "videos")
            
            if self.config.skip_video_gen and self.config.input_videos_dir:
                # 跳过视频生成，直接使用已有的视频
                report_progress(4, "Skipping video generation, loading existing videos...")
                
                # 查找已有的视频文件
                for view_name in ["front", "back", "left", "right"]:
                    # 尝试多种命名格式
                    possible_names = [
                        f"{self.config.character_name}_{view_name}_animation.mp4",
                        f"{view_name}_animation.mp4",
                        f"{view_name}.mp4",
                        f"{self.config.character_name}_{view_name}.mp4",
                    ]
                    
                    for video_name in possible_names:
                        video_path = os.path.join(self.config.input_videos_dir, video_name)
                        if os.path.exists(video_path):
                            results["videos"][view_name] = video_path
                            print(f"[Pipeline] Found existing video for {view_name}: {video_path}")
                            break
                    
                    if view_name not in results["videos"]:
                        print(f"[Pipeline] Warning: No video found for view {view_name}")
                
            else:
                # 正常流程：生成视频
                # ============ Step 4: 为每个视角异步生成动画视频 ============
                report_progress(4, "Generating animation videos for each view (async)...")
                
                os.makedirs(videos_dir, exist_ok=True)
                
                def generate_single_video(view_name: str, view_image: Image.Image) -> Tuple[str, str]:
                    """生成单个视角的视频"""
                    print(f"\n[Pipeline] Starting video generation for view: {view_name}")
                    
                    # 使用检测到的背景色（如果没有检测，默认白色）
                    bg_color = self.detected_bg_color if self.detected_bg_color else "white"
                    animation_prompt = self._build_animation_prompt(view_name, bg_color)
                    
                    video_result = self.seedance.generate_video(
                        reference_image=view_image,
                        prompt=animation_prompt,
                        duration=self.config.video_duration,
                        draft=self.config.video_draft,
                        generate_audio=self.config.video_generate_audio,
                    )
                    
                    # 复制视频到输出目录
                    video_filename = f"{self.config.character_name}_{view_name}_animation.mp4"
                    video_save_path = os.path.join(videos_dir, video_filename)
                    shutil.copy(video_result.video_path, video_save_path)
                    
                    print(f"[Pipeline] Completed video for view: {view_name}")
                    return view_name, video_save_path
                
                # 异步并行生成 4 个视频
                print(f"[Pipeline] Starting parallel video generation for {len(views)} views...")
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    # 提交所有任务
                    future_to_view = {
                        executor.submit(generate_single_video, view_name, view_image): view_name
                        for view_name, view_image in views.items()
                    }
                    
                    # 收集结果
                    for future in as_completed(future_to_view):
                        view_name = future_to_view[future]
                        try:
                            result_view_name, video_path = future.result()
                            results["videos"][result_view_name] = video_path
                            print(f"[Pipeline] Saved video: {video_path}")
                        except Exception as e:
                            print(f"[Pipeline] ERROR generating video for {view_name}: {e}")
                            raise
            
            print(f"[Pipeline] All {len(results['videos'])} videos generated successfully!")
            
            # ============ Step 5: 从视频提取帧 ============
            report_progress(5, "Extracting frames from videos...")
            
            frames_dir = os.path.join(self.output_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            for view_name, video_path in results["videos"].items():
                print(f"\n[Pipeline] Extracting frames from: {view_name}")
                
                # 提取帧
                frames = extract_frames_from_video(
                    video_path=video_path,
                    start_time=0,
                    end_time=0,  # 整个视频
                    max_frames=self.config.frames_per_video,
                    target_fps=self.config.target_fps
                )
                
                if not frames:
                    print(f"[Pipeline] Warning: No frames extracted from {view_name}")
                    continue
                
                # 保存帧（带抠图）
                view_frames_dir = os.path.join(frames_dir, view_name)
                os.makedirs(view_frames_dir, exist_ok=True)
                
                # 确定抠图方法：如果用户设置了 auto，且我们检测到了背景色，则使用对应方法
                bg_method = self.config.bg_method
                if bg_method == "auto" and self.detected_bg_color:
                    bg_method = self.detected_bg_color  # "white" 或 "green"
                    print(f"[Pipeline] Using detected background color for removal: {bg_method}")
                
                frame_prefix = f"{self.config.character_name}_{view_name}"
                saved_paths = save_frames(
                    frames=frames,
                    output_dir=view_frames_dir,
                    prefix=frame_prefix,
                    start_index=1,
                    remove_bg=self.config.remove_background,
                    bg_method=bg_method,
                    bg_tolerance=self.config.bg_tolerance,
                    bg_edge_shrink=self.config.bg_edge_shrink
                )
                
                results["frames"][view_name] = saved_paths
                
                # 创建 Sprite Sheet（也需要抠图）
                if self.config.remove_background:
                    # 对帧进行抠图处理
                    processed_frames = [
                        remove_background_advanced(
                            f, 
                            method=bg_method, 
                            tolerance=self.config.bg_tolerance,
                            edge_shrink=self.config.bg_edge_shrink
                        )
                        for f in frames
                    ]
                    sprite_sheet, frame_size = create_sprite_sheet(processed_frames)
                else:
                    sprite_sheet, frame_size = create_sprite_sheet(frames)
                    
                sheet_path = os.path.join(frames_dir, f"{self.config.character_name}_{view_name}_spritesheet.png")
                sprite_sheet.save(sheet_path)
                results["sprite_sheets"][view_name] = sheet_path
                print(f"[Pipeline] Saved sprite sheet: {sheet_path}")
            
            # ============ Step 6: 生成摘要 ============
            report_progress(6, "Generating summary...")
            
            summary = self._generate_summary(results)
            results["summary"] = summary
            
            # 保存摘要
            summary_path = os.path.join(self.output_dir, "README.md")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            results["summary_file"] = summary_path
            
            print(f"\n{'='*60}")
            print(f"[Pipeline] COMPLETED!")
            print(f"{'='*60}")
            print(summary)
            
            return results
            
        except Exception as e:
            print(f"\n[Pipeline] ERROR: {str(e)}")
            raise
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """生成流水线执行摘要"""
        summary = f"""# Actor Animation Pipeline Results

## Character: {self.config.character_name}

### Generated Files

#### Views (4 directions)
"""
        for view_name, path in results.get("views", {}).items():
            summary += f"- **{view_name}**: `{os.path.basename(path)}`\n"
        
        summary += "\n#### Animation Videos\n"
        for view_name, path in results.get("videos", {}).items():
            summary += f"- **{view_name}**: `{os.path.basename(path)}`\n"
        
        summary += "\n#### Extracted Frames\n"
        for view_name, paths in results.get("frames", {}).items():
            summary += f"- **{view_name}**: {len(paths)} frames\n"
        
        summary += "\n#### Sprite Sheets\n"
        for view_name, path in results.get("sprite_sheets", {}).items():
            summary += f"- **{view_name}**: `{os.path.basename(path)}`\n"
        
        summary += f"""
### Configuration
- Video Duration: {self.config.video_duration}s
- Draft Mode: {self.config.video_draft}
- Frames per Video: {self.config.frames_per_video}

### Output Directory
`{self.output_dir}`
"""
        return summary


# ============ 便捷函数 ============

def run_pipeline(
    api_key: str,
    character_image_path: str = None,
    character_name: str = None,
    character_description: str = None,
    output_dir: str = None,
    video_duration: int = None,
    frames_per_video: int = None,
    remove_background: bool = None,
    bg_method: str = None,
    bg_tolerance: int = None,
    bg_edge_shrink: int = None,
    generated_bg_color: str = None,
    skip_image_gen: bool = None,
    skip_video_gen: bool = None,
    input_multiview_image: str = None,
    input_videos_dir: str = None,
) -> Dict[str, Any]:
    """
    运行角色动画生成流水线（便捷函数）
    
    Args:
        api_key: ARK API Key
        character_image_path: 角色图片路径（跳过图片生成时可选）
        character_name: 角色名称（用于文件命名）
        character_description: 角色描述（可选）
        output_dir: 输出目录（可选）
        video_duration: 视频时长（秒）
        frames_per_video: 每个视频提取的帧数
        remove_background: 是否移除背景（抠图）
        bg_method: 背景移除方法 ("white", "green", "auto")
        bg_tolerance: 背景色容差
        bg_edge_shrink: 边缘内缩像素数（去除边框）
        generated_bg_color: 生成图片/视频时的背景色 ("white", "green", "auto"自动检测)
        skip_image_gen: 跳过图片生成
        skip_video_gen: 跳过视频生成
        input_multiview_image: 已有的多视角图片路径（跳过图片生成时使用）
        input_videos_dir: 已有的视频目录路径（跳过视频生成时使用）
    
    所有参数的默认值定义在 PipelineConfig 中
    
    Returns:
        结果字典
    """
    # 构建配置，只传入非 None 的参数（使用 PipelineConfig 的默认值）
    config_kwargs = {"api_key": api_key}
    if character_name is not None: config_kwargs["character_name"] = character_name
    if output_dir is not None: config_kwargs["output_dir"] = output_dir
    if video_duration is not None: config_kwargs["video_duration"] = video_duration
    if frames_per_video is not None: config_kwargs["frames_per_video"] = frames_per_video
    if remove_background is not None: config_kwargs["remove_background"] = remove_background
    if bg_method is not None: config_kwargs["bg_method"] = bg_method
    if bg_tolerance is not None: config_kwargs["bg_tolerance"] = bg_tolerance
    if bg_edge_shrink is not None: config_kwargs["bg_edge_shrink"] = bg_edge_shrink
    if generated_bg_color is not None: config_kwargs["generated_bg_color"] = generated_bg_color
    if skip_image_gen is not None: config_kwargs["skip_image_gen"] = skip_image_gen
    if skip_video_gen is not None: config_kwargs["skip_video_gen"] = skip_video_gen
    if input_multiview_image is not None: config_kwargs["input_multiview_image"] = input_multiview_image
    if input_videos_dir is not None: config_kwargs["input_videos_dir"] = input_videos_dir
    
    # 固定使用预览模式和无音频
    config_kwargs["video_draft"] = True
    config_kwargs["video_generate_audio"] = False
    
    config = PipelineConfig(**config_kwargs)
    
    pipeline = ActorAnimationPipeline(config)
    return pipeline.run(character_image_path, character_description)


# ============ 命令行入口 ============

if __name__ == "__main__":
    import argparse
    
    # 从 PipelineConfig 获取默认值（唯一真实来源）
    _defaults = PipelineConfig(api_key="")
    
    parser = argparse.ArgumentParser(
        description="Actor Animation Pipeline - Generate multi-view character animations"
    )
    parser.add_argument("--image", help="Path to character image (optional if skipping image generation)")
    parser.add_argument("--api-key", required=True, help="ARK API Key")
    parser.add_argument("--name", default=None, 
                        help=f"Character name for file naming (default: {_defaults.character_name})")
    parser.add_argument("--description", default=None, help="Character description")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--duration", type=int, default=None, 
                        help=f"Video duration in seconds (default: {_defaults.video_duration})")
    parser.add_argument("--frames", type=int, default=None, 
                        help=f"Frames to extract per video (default: {_defaults.frames_per_video})")
    
    # 抠图参数
    parser.add_argument("--no-remove-bg", action="store_true", help="Disable background removal")
    parser.add_argument("--bg-method", default=None, choices=["white", "green", "auto"],
                        help=f"Background removal method (default: {_defaults.bg_method})")
    parser.add_argument("--bg-tolerance", type=int, default=None, 
                        help=f"Background color tolerance (default: {_defaults.bg_tolerance})")
    parser.add_argument("--bg-edge-shrink", type=int, default=None,
                        help=f"Edge shrink pixels to remove border (default: {_defaults.bg_edge_shrink})")
    
    # 生成背景色参数
    parser.add_argument("--gen-bg-color", default=None, choices=["white", "green", "auto"],
                        help=f"Background color for generation (default: {_defaults.generated_bg_color}, auto=detect from image)")
    
    # 跳过步骤参数
    parser.add_argument("--skip-image-gen", action="store_true",
                        help="Skip image generation, use existing multiview image")
    parser.add_argument("--skip-video-gen", action="store_true",
                        help="Skip video generation, use existing videos")
    parser.add_argument("--input-multiview", default=None,
                        help="Path to existing multiview image (use with --skip-image-gen)")
    parser.add_argument("--input-videos-dir", default=None,
                        help="Path to directory containing existing videos (use with --skip-video-gen)")
    
    args = parser.parse_args()
    
    # 处理 remove_background: 只有明确指定 --no-remove-bg 时才为 False
    remove_bg = None if not args.no_remove_bg else False
    
    results = run_pipeline(
        api_key=args.api_key,
        character_image_path=args.image,
        character_name=args.name,
        character_description=args.description,
        output_dir=args.output,
        video_duration=args.duration,
        frames_per_video=args.frames,
        remove_background=remove_bg,
        bg_method=args.bg_method,
        bg_tolerance=args.bg_tolerance,
        bg_edge_shrink=args.bg_edge_shrink,
        generated_bg_color=args.gen_bg_color,
        skip_image_gen=args.skip_image_gen if args.skip_image_gen else None,
        skip_video_gen=args.skip_video_gen if args.skip_video_gen else None,
        input_multiview_image=args.input_multiview,
        input_videos_dir=args.input_videos_dir,
    )
    
    print(f"\nResults saved to: {results['output_dir']}")
