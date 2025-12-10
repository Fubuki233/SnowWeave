"""
SnowWeave Gradio App
Gradio 前端界面 - 只处理 UI 展示和用户交互，业务逻辑调用 core 模块
"""

import gradio as gr
import os

# 导入核心模块
from core.config import (
    OUTPUT_DIR,
    DEFAULT_IMAGE_PATH,
    AVAILABLE_BACKENDS,
    GEMINI_MODELS,
    SEEDANCE_MODELS,
    DEFAULT_MODEL,
    LANGUAGES,
    t,
    set_language,
    get_current_language
)
from core.api_manager import get_api_manager
from core.video_generator import VideoGenerator
from core.frame_processor import FrameProcessor
from core.pipeline import FullPipeline
from core.plant_generator import PlantGenerator


# ============ UI 回调函数 ============

def validate_api_ui(api_key: str, backend: str) -> str:
    """验证并初始化 API"""
    if not api_key:
        lang = get_current_language()
        if lang == "zh":
            return "[ERROR] 请输入API密钥"
        return "[ERROR] Please enter API key"
    
    api_manager = get_api_manager()
    return api_manager.initialize(api_key, backend)


def set_model_ui(model_name: str) -> str:
    """设置当前模型"""
    api_manager = get_api_manager()
    return api_manager.set_model(model_name)


def get_models_for_backend_ui(backend: str):
    """根据后端更新模型列表"""
    if backend == "seedance":
        return gr.update(choices=list(SEEDANCE_MODELS.keys()), value="doubao-seedance-1-0-pro-fast-251015")
    return gr.update(choices=list(GEMINI_MODELS.keys()), value=DEFAULT_MODEL)


def generate_video_ui(image, action: str, model_name: str, duration: int):
    """生成动画视频 - UI 回调 (生成器函数，支持流式更新)"""
    generator = VideoGenerator()
    yield from generator.generate(image, action, model_name, duration)


def generate_video_batch_ui(image, actions_text: str, model_name: str, duration: int, backend: str, resolution: str, max_workers: int):
    """批量生成动画视频 - UI 回调 (生成器函数，支持流式更新)"""
    # 解析动作列表（每行一个动作）
    actions = [line.strip() for line in actions_text.strip().split('\n') if line.strip()]
    
    if not actions:
        lang = get_current_language()
        if lang == "zh":
            yield None, None, "[ERROR] 请至少输入一个动作描述"
        else:
            yield None, None, "[ERROR] Please enter at least one action description"
        return
    
    # 如果只有一个动作，使用单个生成函数
    if len(actions) == 1:
        generator = VideoGenerator()
        yield from generator.generate(image, actions[0], model_name, duration, backend, resolution)
    else:
        # 多个动作，使用批量生成
        generator = VideoGenerator()
        yield from generator.generate_multiple(image, actions, model_name, duration, backend, resolution, max_workers)



def extract_frames_ui(video, start_time: float, end_time: float, max_frames: int):
    """从视频提取帧 - UI 回调"""
    processor = FrameProcessor()
    return processor.extract_frames(video, start_time, end_time, max_frames)


def remove_background_ui(uploaded_files, tolerance: int, auto_crop: bool, crop_padding: int, progress=gr.Progress()):
    """去除背景 - UI 回调"""
    def progress_callback(value, desc):
        progress(value, desc=desc)
    
    processor = FrameProcessor()
    return processor.remove_background(uploaded_files, tolerance, auto_crop, crop_padding, progress_callback)


def full_pipeline_ui(image, action, start_time, end_time, max_frames, tolerance, auto_crop, crop_padding, model_name, duration, backend, resolution, max_workers, progress=gr.Progress()):
    """完整流水线 - UI 回调"""
    def progress_callback(value, desc):
        progress(value, desc=desc)
    
    pipeline = FullPipeline()
    return pipeline.run(
        image, action, start_time, end_time, max_frames,
        tolerance, auto_crop, crop_padding, model_name, duration,
        backend, resolution, max_workers,
        progress_callback
    )


def generate_plant_stages_ui(
    image, ref_images, prompt, plant_id, stages, target_width, frames_per_stage,
    tolerance, auto_crop, crop_padding, model_name, duration,
    backend, resolution, reference_mode, progress=gr.Progress()
):
    """植物生成 - UI 回调 (生成器函数，支持流式更新)"""
    generator = PlantGenerator()
    yield from generator.generate(
        image, ref_images, prompt, plant_id, stages, target_width, frames_per_stage,
        tolerance, auto_crop, crop_padding, model_name, duration,
        backend, resolution, reference_mode
    )


# ============ Gradio 界面定义 ============

def create_app():
    """创建 Gradio 应用"""
    
    with gr.Blocks(title="SnowWeave") as app:
        # 标题和语言选择
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("""
        # SnowWeave
        ### Sprite Animation Generation Pipeline / Sprite流水线
        """)
            with gr.Column(scale=1):
                lang_dropdown = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="zh",
                    label="Language / 语言",
                    interactive=True
                )
                lang_status = gr.Textbox(label="Status / 状态", lines=1, interactive=False, visible=False)
        
        lang_dropdown.change(
            fn=set_language,
            inputs=[lang_dropdown],
            outputs=[lang_status]
        )
        
        with gr.Tabs():
            # Tab 0: API设置
            with gr.Tab("Settings / 设置"):
                _create_settings_tab()
            
            # Tab 1: 生成视频
            with gr.Tab("Generate Video / 生成视频"):
                _create_video_tab()
            
            # Tab 2: 提取帧
            with gr.Tab("Extract Frames / 提取帧"):
                _create_extract_tab()
            
            # Tab 3: 去除背景
            with gr.Tab("Remove Background / 去除背景"):
                _create_background_tab()
            
            # Tab 4: 完整流程
            with gr.Tab("Full Pipeline / 完整流程"):
                _create_pipeline_tab()
            
            # Tab 5: 植物生成
            with gr.Tab("Plant Generator / 植物生成"):
                _create_plant_tab()
    
    return app


def _create_settings_tab():
    """创建设置标签页"""
    gr.Markdown("""
### Configure Video Generation API / 配置视频生成 API
Please configure the API key before using video generation features. / 在使用视频生成功能前，需要先配置API密钥。

**Available Backends / 可用后端:**
- **Gemini Veo** (Google): [Get API Key](https://aistudio.google.com/apikey) - 国际访问
- **Seedance** (字节跳动豆包): [获取 API Key](https://www.volcengine.com/product/doubao) - 国内访问
    """)
    
    with gr.Row():
        with gr.Column():
            backend_dropdown = gr.Dropdown(
                label="Video Backend / 视频后端",
                choices=list(AVAILABLE_BACKENDS.keys()),
                value="gemini",
                info="Select video generation backend / 选择视频生成后端"
            )
            api_key_input = gr.Textbox(
                label="API Key / API密钥",
                type="password",
                placeholder="Enter your API key / 输入你的API密钥"
            )
            api_set_btn = gr.Button("Save and Verify / 保存并验证", variant="primary", size="lg")
        
        with gr.Column():
            api_status = gr.Textbox(label="Status / 状态", lines=3, interactive=False)
            gr.Markdown("""
            **Gemini Veo**: Google 的视频生成 API，需要国际网络访问
            
            **Seedance**: 字节跳动豆包的视频生成 API，国内可直接访问
            """)
    
    api_set_btn.click(
        fn=validate_api_ui,
        inputs=[api_key_input, backend_dropdown],
        outputs=[api_status]
    )


def _create_video_tab():
    """创建视频生成标签页"""
    gr.Markdown("""
    ### Generate Character Animation with AI / 使用AI生成角色动画视频
    1. Upload character reference image / 上传角色参考图片
    2. Describe the desired actions (one per line for batch generation) / 描述想要的动作（每行一个动作，支持批量生成）
    3. Select backend and model / 选择后端和模型
    4. Wait for AI to generate animation videos / 等待AI生成动画视频
    """)
    
    with gr.Row():
        with gr.Column():
            gen_image = gr.Image(
                label="Upload Character Image / 上传角色图片", 
                type="numpy", 
                value=DEFAULT_IMAGE_PATH if os.path.exists(DEFAULT_IMAGE_PATH) else None
            )
            gen_action = gr.Textbox(
                label="Action Descriptions (one per line) / 动作描述（每行一个）",
                placeholder="Example / 例如:\nwalking animation, side view, loop\nrunning animation, side view, loop\nattack animation, front view",
                value="walking animation, side view, loop",
                lines=3
            )
            
            with gr.Row():
                gen_backend = gr.Dropdown(
                    label="Backend / 后端",
                    choices=AVAILABLE_BACKENDS,
                    value="gemini",
                    info="Gemini for global, Seedance for China / Gemini适用全球，Seedance适用国内"
                )
                gen_model = gr.Dropdown(
                    label="Select Model / 选择模型",
                    choices=list(GEMINI_MODELS.keys()),
                    value=DEFAULT_MODEL,
                    info="Different models may have different quality / 不同模型可能有不同质量"
                )
            
            with gr.Row():
                gen_duration = gr.Slider(
                    label="Video Duration (seconds) / 视频长度(秒)",
                    minimum=2,
                    maximum=8,
                    value=4,
                    step=1,
                    info="API limited to 2-8 seconds / API限制2-8秒"
                )
                gen_resolution = gr.Dropdown(
                    label="Resolution (Seedance only) / 分辨率（仅Seedance）",
                    choices=["720p", "480p"],
                    value="480p",
                    info="Only effective with Seedance backend / 仅对Seedance后端有效"
                )
            
            gen_max_workers = gr.Slider(
                label="Max Parallel Workers / 最大并行数",
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                info="Number of videos to generate in parallel / 同时生成的视频数量"
            )
            
            gen_btn = gr.Button("Generate Animation Video(s) / 生成动画视频", variant="primary", size="lg")
        
        with gr.Column():
            gen_video_output = gr.Video(label="Generated Video(s) / 生成的视频", autoplay=False)
            gen_image_output = gr.Image(label="Reference Image / 参考图片", type="filepath")
            gen_status = gr.Textbox(label="Status / 状态", lines=8)
    
    # 后端切换时更新模型列表
    gen_backend.change(
        fn=get_models_for_backend_ui,
        inputs=[gen_backend],
        outputs=[gen_model]
    )
    
    gen_btn.click(
        fn=generate_video_batch_ui,
        inputs=[gen_image, gen_action, gen_model, gen_duration, gen_backend, gen_resolution, gen_max_workers],
        outputs=[gen_video_output, gen_image_output, gen_status]
    )


def _create_extract_tab():
    """创建帧提取标签页"""
    gr.Markdown("""
    ### Extract Sprite Frames from Video / 从视频中提取Sprite帧
    1. Upload video file / 上传视频文件
    2. Set extraction parameters (time range, frame count) / 设置提取参数（时间段、帧数）
    3. Automatically generate Sprite Sheet / 自动生成Sprite Sheet
    """)
    
    with gr.Row():
        with gr.Column():
            ext_video = gr.Video(label="Upload Video / 上传视频")
            
            with gr.Row():
                ext_start = gr.Number(label="Start Time (seconds) / 开始时间(秒)", value=0, minimum=0)
                ext_end = gr.Number(label="End Time (seconds) / 结束时间(秒)", value=0, minimum=0)
            
            ext_max_frames = gr.Slider(
                label="Max Frames / 最大帧数",
                minimum=1,
                maximum=100,
                value=24,
                step=1
            )
            
            gr.Markdown("Tip / 提示: Setting both start and end time to 0 means parsing the entire video / 开始和结束时间都设为0表示解析整个视频")
            
            ext_btn = gr.Button("Extract Frames / 提取帧", variant="primary", size="lg")
        
        with gr.Column():
            ext_sheet_output = gr.Image(label="Sprite Sheet")
            ext_gallery = gr.Gallery(label="Extracted Frames / 提取的帧", columns=4, height="auto")
            ext_status = gr.Textbox(label="Status / 状态", lines=4)
    
    ext_btn.click(
        fn=extract_frames_ui,
        inputs=[ext_video, ext_start, ext_end, ext_max_frames],
        outputs=[ext_sheet_output, ext_gallery, ext_status]
    )


def _create_background_tab():
    """创建背景去除标签页"""
    gr.Markdown("""
    ### Automatically Remove Green Screen Background / 自动去除绿幕背景
    1. Upload single or multiple images / 上传单张图片或多张图片
    2. Adjust tolerance and crop parameters / 调整容差和裁剪参数
    3. Automatically detect and remove background / 自动检测并移除背景
    """)
    
    with gr.Row():
        with gr.Column():
            rm_input = gr.File(
                label="Upload Images / 上传图片",
                file_count="multiple",
                file_types=["image"],
                type="filepath"
            )
            
            rm_tolerance = gr.Slider(
                label="Color Tolerance / 颜色容差",
                minimum=0,
                maximum=255,
                value=180,
                step=1,
                info="Higher value removes wider color range / 值越大,移除的颜色范围越广"
            )
            
            rm_auto_crop = gr.Checkbox(
                label="Auto Crop Transparent Edges / 自动裁剪透明边缘",
                value=False
            )
            
            rm_padding = gr.Slider(
                label="Crop Padding (pixels) / 裁剪边距(像素)",
                minimum=0,
                maximum=50,
                value=0,
                step=1
            )
            
            rm_btn = gr.Button("Remove Background / 去除背景", variant="primary", size="lg")
        
        with gr.Column():
            rm_sheet_output = gr.Image(label="Processed Sprite Sheet / 处理后的Sprite Sheet")
            rm_gallery = gr.Gallery(label="Processed Frames / 处理后的帧", columns=4, height="auto")
            rm_status = gr.Textbox(label="Status / 状态", lines=4)
    
    rm_btn.click(
        fn=remove_background_ui,
        inputs=[rm_input, rm_tolerance, rm_auto_crop, rm_padding],
        outputs=[rm_sheet_output, rm_gallery, rm_status]
    )


def _create_pipeline_tab():
    """创建完整流程标签页"""
    gr.Markdown("""
    ### One-Click Complete Process / 一键完成全流程
    Upload character image → Generate video → Extract frames → Remove background → Output Sprite Sheet /
    上传角色图片 → 生成视频 → 提取帧 → 去除背景 → 输出Sprite Sheet
    """)
    
    with gr.Row():
        with gr.Column():
            full_image = gr.Image(
                label="Upload Character Image / 上传角色图片", 
                type="numpy", 
                value=DEFAULT_IMAGE_PATH if os.path.exists(DEFAULT_IMAGE_PATH) else None
            )
            full_action = gr.Textbox(
                label="Action Descriptions (one per line) / 动作描述（每行一个）",
                placeholder="Example / 例如:\nwalking animation, side view, loop\nrunning animation, side view, loop\nattack animation, front view",
                value="walking animation, side view, loop",
                lines=3,
                info="Enter multiple actions for batch generation / 输入多个动作进行批量生成"
            )
            
            gr.Markdown("#### Extraction Parameters / 提取参数 (Set both to 0 to use entire video / 都设为零则截取整个视频)")
            with gr.Row():
                full_start = gr.Number(label="Start Time (sec) / 开始时间(秒)", value=0)
                full_end = gr.Number(label="End Time (sec) / 结束时间(秒)", value=0)
            
            full_max_frames = gr.Slider(
                label="Max Frames / 最大帧数",
                minimum=1,
                maximum=100,
                value=24,
                step=1
            )
            
            gr.Markdown("#### Background Removal Parameters / 背景去除参数")
            full_tolerance = gr.Slider(
                label="Color Tolerance / 颜色容差",
                minimum=0,
                maximum=255,
                value=180,
                step=1
            )
            
            full_auto_crop = gr.Checkbox(
                label="Auto Crop / 自动裁剪",
                value=False
            )
            
            full_padding = gr.Slider(
                label="Crop Padding / 裁剪边距",
                minimum=0,
                maximum=50,
                value=0,
                step=1
            )
            
            gr.Markdown("#### Model & Backend Settings / 模型和后端设置")
            
            full_backend = gr.Dropdown(
                label="Video Backend / 视频后端",
                choices=AVAILABLE_BACKENDS,
                value="gemini",
                info="gemini=国际, seedance=国内"
            )
            
            full_model = gr.Dropdown(
                label="Video Generation Model / 视频生成模型",
                choices=list(GEMINI_MODELS.keys()),
                value=DEFAULT_MODEL,
                info="Select different model / 选择不同的模型"
            )
            
            full_resolution = gr.Dropdown(
                label="Resolution / 分辨率 (Seedance)",
                choices=["480p", "720p", "1080p"],
                value="720p",
                info="Only effective with Seedance / 仅对Seedance有效"
            )
            
            full_duration = gr.Slider(
                label="Video Duration (sec) / 视频长度(秒)",
                minimum=4,
                maximum=8,
                value=6,
                step=1,
                info="Video generation duration, API limited to 4-8 sec / 视频生成的时长,API限制4-8秒"
            )
            
            full_max_workers = gr.Slider(
                label="Max Parallel Workers / 最大并行数",
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                info="Number of videos to generate in parallel / 同时生成的视频数量"
            )
            
            # 后端切换时更新模型列表
            full_backend.change(
                fn=get_models_for_backend_ui,
                inputs=[full_backend],
                outputs=[full_model]
            )
            
            full_btn = gr.Button("Start Full Pipeline / 开始完整流程", variant="primary", size="lg")
        
        with gr.Column():
            gr.Markdown("#### Generated Videos / 生成的视频")
            full_video_output = gr.File(
                label="All Generated Videos (download) / 所有生成的视频（下载）",
                file_count="multiple",
                type="filepath"
            )
            
            full_sheet_output = gr.Image(label="Final Sprite Sheet / 最终Sprite Sheet", type="filepath")
            full_ref_output = gr.Image(label="Reference Image / 参考图片", type="filepath")
            full_gallery = gr.Gallery(label="Final Frames Preview / 最终帧预览", columns=4, height="auto")
            full_status = gr.Textbox(label="Execution Status / 执行状态", lines=10)
    
    full_btn.click(
        fn=full_pipeline_ui,
        inputs=[
            full_image, full_action, full_start, full_end, full_max_frames,
            full_tolerance, full_auto_crop, full_padding, full_model, full_duration,
            full_backend, full_resolution, full_max_workers
        ],
        outputs=[full_video_output, full_sheet_output, full_ref_output, full_gallery, full_status]
    )


def _create_plant_tab():
    """创建植物生成标签页"""
    gr.Markdown("""
    ### Generate Plant Growth Stages / 生成植物生长阶段
    Generate animated frames for each growth stage of a plant. /
    为植物的每个生长阶段生成动画帧。
    
    **Output Format / 输出格式:** `{plant_id}-stage{X}-frame{Y}.png`
    
    **Example / 示例:** `oak_tree-stage1-frame1.png`, `oak_tree-stage2-frame1.png`
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Input / 输入 (At least one required / 至少需要一个)")
            
            plant_image = gr.Image(
                label="Reference Image (First Frame) / 参考图片 (首帧)", 
                type="numpy"
            )
            
            plant_ref_images = gr.File(
                label="Additional Reference Images (Lite I2V only) / 额外参考图 (仅Lite I2V)",
                file_count="multiple",
                file_types=["image"],
                type="filepath"
            )
            gr.Markdown("*Lite I2V模型支持多参考图，在提示词中用[图1][图2]引用*", elem_classes="info-text")
            
            plant_prompt = gr.Textbox(
                label="Plant Description (Optional) / 植物描述 (可选)",
                placeholder="Example / 例如: a small oak tree, a green cactus, wheat plant\nLite I2V: [图1]的植物在[图2]的土地上生长",
                value=""
            )
            
            plant_id_input = gr.Textbox(
                label="Plant ID / 植物ID (Required / 必填)",
                placeholder="Example / 例如: oak_tree, cactus, wheat",
                value="",
                info="Used for file naming / 用于文件命名"
            )
            
            gr.Markdown("#### Growth Parameters / 生长参数")
            
            plant_stages = gr.Slider(
                label="Growth Stages / 生长阶段数",
                minimum=1,
                maximum=8,
                value=4,
                step=1,
                info="Number of growth stages / 生长阶段的数量"
            )
            
            plant_width = gr.Slider(
                label="Output Width (px) / 输出宽度 (像素)",
                minimum=32,
                maximum=512,
                value=256,
                step=16,
                info="All frames will be scaled to this width / 所有帧将缩放到此宽度"
            )
            
            plant_frames = gr.Slider(
                label="Frames Per Stage / 每阶段帧数",
                minimum=1,
                maximum=24,
                value=24,
                step=1,
                info="Animation frames per growth stage / 每个生长阶段的动画帧数"
            )
            
            gr.Markdown("#### Background Removal / 背景去除")
            
            plant_tolerance = gr.Slider(
                label="Color Tolerance / 颜色容差",
                minimum=0,
                maximum=255,
                value=180,
                step=1
            )
            
            plant_auto_crop = gr.Checkbox(
                label="Auto Crop / 自动裁剪",
                value=False
            )
            
            plant_padding = gr.Slider(
                label="Crop Padding / 裁剪边距",
                minimum=0,
                maximum=50,
                value=5,
                step=1
            )
            
            gr.Markdown("#### Backend & Model Settings / 后端和模型设置")
            
            plant_reference_mode = gr.Radio(
                label="Reference Image Mode / 参考图片模式",
                choices=[
                    ("Use previous stage's last frame / 使用上一阶段最后帧", "last_frame"),
                    ("Always use original input image / 始终使用原始输入图片", "input_image")
                ],
                value="last_frame",
                info="How to select reference image for subsequent stages / 如何为后续阶段选择参考图片"
            )
            
            plant_backend = gr.Dropdown(
                label="Video Backend / 视频后端",
                choices=list(AVAILABLE_BACKENDS.keys()),
                value="gemini",
                info="gemini=国际, seedance=国内"
            )
            
            plant_model = gr.Dropdown(
                label="Video Generation Model / 视频生成模型",
                choices=list(GEMINI_MODELS.keys()),
                value=DEFAULT_MODEL
            )
            
            plant_resolution = gr.Dropdown(
                label="Resolution / 分辨率 (Seedance)",
                choices=["480p", "720p", "1080p"],
                value="480p",
                info="Seedance 专用参数"
            )
            
            plant_duration = gr.Slider(
                label="Video Duration (sec) / 视频时长 (秒)",
                minimum=2,
                maximum=10,
                value=2,
                step=1
            )
            
            # 更新模型列表的回调
            plant_backend.change(
                fn=get_models_for_backend_ui,
                inputs=[plant_backend],
                outputs=[plant_model]
            )
            
            plant_btn = gr.Button("Generate Plant Stages / 生成植物阶段", variant="primary", size="lg")
        
        with gr.Column():
            plant_preview = gr.Image(label="Preview Sheet / 预览合成图", type="filepath")
            plant_gallery = gr.Gallery(label="Generated Frames / 生成的帧", columns=4, height="auto")
            
            # 使用 File 组件显示所有视频文件
            gr.Markdown("#### Generated Videos / 生成的视频")
            plant_videos = gr.File(
                label="All Stage Videos (click to download) / 所有阶段视频（点击下载）", 
                file_count="multiple",
                type="filepath"
            )
            
            plant_config = gr.File(label="Config File / 配置文件")
            plant_status = gr.Textbox(label="Status / 状态", lines=15)
    
    plant_btn.click(
        fn=generate_plant_stages_ui,
        inputs=[
            plant_image, plant_ref_images, plant_prompt, plant_id_input, plant_stages, plant_width, plant_frames,
            plant_tolerance, plant_auto_crop, plant_padding, plant_model, plant_duration,
            plant_backend, plant_resolution, plant_reference_mode
        ],
        outputs=[plant_preview, plant_gallery, plant_config, plant_videos, plant_status]
    )


# ============ 主程序入口 ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SnowWeave - Sprite Animation Generation Pipeline / Sprite动画生成流水线')
    parser.add_argument('--share', action='store_true', help='Create public share link / 创建公共分享链接')
    parser.add_argument('--server-name', default='0.0.0.0', help='Server address (default: 0.0.0.0) / 服务器地址')
    parser.add_argument('--server-port', type=int, default=7860, help='Server port (default: 7860) / 服务器端口')
    parser.add_argument('--root-path', default=None, help='Reverse proxy root path / 反向代理根路径')
    parser.add_argument('--max-file-size', default='100mb', help='Max file upload size / 最大文件上传大小')
    args = parser.parse_args()
    
    print("="*70)
    print("  SnowWeave - Sprite Animation Generation Pipeline")
    print("  Sprite动画生成流水线")
    print("="*70)
    print("\nStarting Gradio server / 启动Gradio服务器...")
    print(f"  - Address / 地址: {args.server_name}:{args.server_port}")
    if args.share:
        print("  - Mode / 模式: Public share / 公共分享")
    if args.root_path:
        print(f"  - Reverse proxy path / 反向代理路径: {args.root_path}")
    print("\nPress Ctrl+C to stop server / 按 Ctrl+C 停止服务器")
    print("="*70 + "\n")
    
    # 创建应用
    app = create_app()
    
    # 获取输出目录的绝对路径
    abs_output_dir = os.path.abspath(OUTPUT_DIR)
    
    app.queue(
        max_size=20,
        api_open=False
    ).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
        max_file_size=args.max_file_size,
        allowed_paths=[abs_output_dir, os.path.dirname(abs_output_dir)],
        root_path=args.root_path
    )
