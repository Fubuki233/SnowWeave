"""
Sprite动画生成流水线 - Gradio Web界面
提供可视化操作界面

运行方法:
    python gradio_app.py
    
然后在浏览器打开显示的URL
"""

import gradio as gr
import os
import sys
import time
from datetime import datetime
from PIL import Image
import tempfile
import shutil
from google import genai

# 修复 Windows 上的 asyncio ProactorEventLoop 警告
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 导入流水线功能
from generate_sprite_animation import (
    load_reference_image,
    generate_animation_video
)
from extract_sprite_frames import (
    extract_frames_from_video_segment,
    create_sprite_sheet,
    save_individual_frames
)
from remove_background import (
    process_directory,
    process_image,
    detect_background_color
)

# 创建输出目录
OUTPUT_DIR = "gradio_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局API客户端和密钥
gemini_client = None
current_api_key = ""

# 可用模型列表
AVAILABLE_MODELS = {
    "veo-3.1-generate-preview": "Veo 3.1 (预览版，最新)",
    "veo-3.1-fast-generate-preview": "Veo 3.1 Fast (预览版，快速)",
    "veo-3.0-generate-001": "Veo 3.0 (稳定版)",
    "veo-3.0-fast-generate-001": "Veo 3.0 Fast (稳定版，快速)",
    "veo-2.0-generate-001": "Veo 2.0 (旧版)",
}
DEFAULT_MODEL = "veo-2.0-generate-001"
current_model = DEFAULT_MODEL

def clean_old_outputs(output_type="video"):
    """清理旧的输出文件"""
    try:
        pattern = f"{output_type}_*" if output_type else "*"
        for item in os.listdir(OUTPUT_DIR):
            item_path = os.path.join(OUTPUT_DIR, item)
            if os.path.isdir(item_path) and item.startswith(output_type):
                shutil.rmtree(item_path)
                print(f"已删除旧输出: {item_path}")
    except Exception as e:
        print(f"清理输出时出错: {e}")

def initialize_api(api_key):
    """初始化Gemini API客户端"""
    global gemini_client, current_api_key
    try:
        gemini_client = genai.Client(api_key=api_key)
        current_api_key = api_key
        return "✅ API密钥验证成功！"
    except Exception as e:
        return f"❌ API密钥验证失败: {str(e)}"

def get_current_api_key():
    """获取当前保存的API密钥"""
    return current_api_key

def set_model(model_name):
    """设置当前使用的模型"""
    global current_model
    current_model = model_name
    return f"✅ 已切换到模型: {AVAILABLE_MODELS.get(model_name, model_name)}"

def generate_video_ui(image, action, model_name):
    """生成动画视频"""
    if gemini_client is None:
        return None, None, "❌ 请先在设置中配置API密钥"
    
    if image is None:
        return None, None, "请先上传图片"
    
    try:
        # 清理旧的视频输出
        clean_old_outputs("video")
        
        yield None, None, "🎬 正在加载图片..."
        
        # 保存临时图片
        temp_img_path = os.path.join(tempfile.gettempdir(), f"temp_{int(time.time())}.png")
        Image.fromarray(image).save(temp_img_path)
        
        yield None, "🎨 正在生成动画视频 (这可能需要几分钟)..."
        
        # 加载图片
        reference_image = load_reference_image(temp_img_path)
        
        # 构建提示词
        full_prompt = f"""
Create a smooth sprite animation of a STYLIZED, NON-REALISTIC game character performing {action} IN PLACE.

IMPORTANT - CHARACTER STYLE:
- This is a FICTIONAL GAME CHARACTER, not a real person
- Use CARTOON/PIXEL ART style with simplified features
- ABSTRACT or STYLIZED representation only
- NO photorealistic human features
- Game sprite aesthetic (像素/卡通风格游戏角色)

CRITICAL REQUIREMENTS:
- START IMMEDIATELY with the character visible - NO fade in effect
- Character STAYS IN THE CENTER, does NOT move left or right across the screen
- Only the character's body/limbs animate, position remains FIXED
- Smooth, fluid animation with natural motion
- Complete {action} cycle IN PLACE
- Keep the exact same character design, colors, and art style
- Loop-able animation cycle

VISUAL STYLE REQUIREMENTS:
- NO physics effects (no particles, debris, dust, etc.)
- NO lighting effects (no shadows, highlights, glows, reflections)
- NO post-processing effects (no blur, bloom, color grading)
- Flat, clean animation with solid colors only
- Simple sprite animation style without any special effects
- Background MUST be PURE CHROMA GREEN (#00FF00, RGB 0,255,0)

Style: Clean pixel art / 2D game sprite animation with smooth motion, no effects
Camera: Fixed, character stays in center and animates in place
Background: Pure chroma green (#00FF00) for entire duration
Effects: NONE - no physics, lighting, or post-processing effects
"""
        
        # 生成视频
        video = generate_animation_video(reference_image, full_prompt, gemini_client, model_name)
        
        if video is None:
            yield None, None, "❌ 视频生成失败: API 返回空结果"
            return
        
        yield None, None, "📥 正在下载视频..."
        
        # 保存视频和参考图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_DIR, f"video_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存视频
        output_path = os.path.join(output_dir, "animation.mp4")
        video_data = gemini_client.files.download(file=video.video)
        with open(output_path, "wb") as f:
            f.write(video_data)
        
        # 保存参考图片
        reference_path = os.path.join(output_dir, "reference_image.png")
        reference_image.save(reference_path)
        
        # 保存元数据
        metadata_path = os.path.join(output_dir, "metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(f"生成时间: {timestamp}\n")
            f.write(f"动作描述: {action}\n")
            f.write(f"使用模型: {model_name}\n")
            f.write(f"视频文件: animation.mp4\n")
            f.write(f"参考图片: reference_image.png\n")
        
        # 清理临时文件
        os.remove(temp_img_path)
        
        # 确保返回绝对路径
        abs_video_path = os.path.abspath(output_path)
        abs_ref_path = os.path.abspath(reference_path)
        
        yield abs_video_path, abs_ref_path, f"""✅ 视频生成完成!

📁 输出目录: {output_dir}
📹 视频文件: animation.mp4
🖼️ 参考图片: reference_image.png
📝 元数据: metadata.txt

💾 可直接下载视频和图片
"""
        
    except Exception as e:
        yield None, None, f"❌ 错误: {str(e)}"

def extract_frames_ui(video, start_time, end_time, max_frames):
    """从视频提取帧"""
    if video is None:
        return None, None, "请先上传视频"
    
    try:
        yield None, None, "✂️ 正在提取帧..."
        
        # 提取帧
        frames = extract_frames_from_video_segment(
            video,
            float(start_time),
            float(end_time),
            int(max_frames)
        )
        
        if not frames:
            yield None, None, "❌ 没有提取到帧"
            return
        
        yield None, None, f"💾 正在保存 {len(frames)} 帧..."
        
        # 保存帧
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_DIR, f"frames_{timestamp}")
        frames_dir = os.path.join(output_dir, "frames")
        save_individual_frames(frames, output_dir=frames_dir)
        
        # 创建sprite sheet
        sprite_sheet, _ = create_sprite_sheet(frames, frame_size=None)
        sheet_path = os.path.join(output_dir, "sprite_sheet.png")
        sprite_sheet.save(sheet_path)
        
        # 创建预览网格
        preview_images = [frame for frame in frames[:8]]  # 最多8帧预览
        
        yield sheet_path, preview_images, f"✅ 提取完成!\n共 {len(frames)} 帧\nSprite Sheet: {sheet_path}\n帧目录: {frames_dir}"
        
    except Exception as e:
        yield None, None, f"❌ 错误: {str(e)}"

def remove_background_ui(input_path, tolerance, auto_crop, crop_padding, progress=gr.Progress()):
    """去除背景"""
    if input_path is None:
        return None, None, "请先提供输入"
    
    try:
        progress(0, desc="🎨 开始处理...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_DIR, f"nobg_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 判断是目录还是单个文件
        if os.path.isdir(input_path):
            progress(0.2, desc="📂 处理目录中的图片...")
            
            # 处理目录
            nobg_dir = os.path.join(output_dir, "frames")
            process_directory(
                input_path,
                output_dir=nobg_dir,
                tolerance=int(tolerance),
                num_workers=None,
                auto_crop=auto_crop,
                crop_padding=int(crop_padding)
            )
            
            progress(0.8, desc="📦 创建sprite sheet...")
            
            # 创建sprite sheet
            nobg_files = sorted([f for f in os.listdir(nobg_dir) if f.endswith('.png')])
            if nobg_files:
                final_frames = [Image.open(os.path.join(nobg_dir, f)) for f in nobg_files]
                final_sheet, _ = create_sprite_sheet(final_frames, frame_size=None)
                sheet_path = os.path.join(output_dir, "sprite_sheet.png")
                final_sheet.save(sheet_path)
                
                preview_images = final_frames[:8]
            else:
                sheet_path = None
                preview_images = []
            
            progress(1.0, desc="✅ 完成!")
            return sheet_path, preview_images, f"✅ 背景去除完成!\nSprite Sheet: {sheet_path}\n帧目录: {nobg_dir}"
            
        else:
            progress(0.3, desc="🖼️ 处理单张图片...")
            
            # 处理单个文件
            output_path = os.path.join(output_dir, "output.png")
            process_image(
                input_path,
                output_path=output_path,
                tolerance=int(tolerance),
                auto_crop=auto_crop,
                crop_padding=int(crop_padding)
            )
            
            progress(1.0, desc="✅ 完成!")
            
            result_img = Image.open(output_path)
            return output_path, [result_img], f"✅ 背景去除完成!\n保存路径: {output_path}"
        
    except Exception as e:
        return None, None, f"❌ 错误: {str(e)}"

def full_pipeline_ui(image, action, start_time, end_time, max_frames, tolerance, auto_crop, crop_padding, model_name, progress=gr.Progress()):
    """完整流水线"""
    if gemini_client is None:
        return None, None, None, None, "❌ 请先在设置中配置API密钥"
    
    if image is None:
        return None, None, None, None, "请先上传图片"
    
    try:
        # 清理旧的完整流程输出
        clean_old_outputs("full")
        
        # 步骤1: 生成视频
        progress(0, desc="🎬 步骤1/4: 生成动画视频...")
        
        temp_img_path = os.path.join(tempfile.gettempdir(), f"temp_{int(time.time())}.png")
        Image.fromarray(image).save(temp_img_path)
        
        reference_image = load_reference_image(temp_img_path)
        
        full_prompt = f"""
Create a smooth sprite animation of a STYLIZED, NON-REALISTIC game character performing {action} IN PLACE.

IMPORTANT - CHARACTER STYLE:
- This is a FICTIONAL GAME CHARACTER, not a real person
- Use CARTOON/PIXEL ART style with simplified features
- ABSTRACT or STYLIZED representation only
- NO photorealistic human features
- Game sprite aesthetic (像素/卡通风格游戏角色)

CRITICAL REQUIREMENTS:
- START IMMEDIATELY with the character visible - NO fade in effect
- Character STAYS IN THE CENTER, does NOT move left or right across the screen
- Only the character's body/limbs animate, position remains FIXED
- Smooth, fluid animation with natural motion
- Complete {action} cycle IN PLACE
- Keep the exact same character design, colors, and art style
- Loop-able animation cycle

VISUAL STYLE REQUIREMENTS:
- NO physics effects (no particles, debris, dust, etc.)
- NO lighting effects (no shadows, highlights, glows, reflections)
- NO post-processing effects (no blur, bloom, color grading)
- Flat, clean animation with solid colors only
- Simple sprite animation style without any special effects
- Background MUST be PURE CHROMA GREEN (#00FF00, RGB 0,255,0)

Style: Clean pixel art / 2D game sprite animation with smooth motion, no effects
Camera: Fixed, character stays in center and animates in place
Background: Pure chroma green (#00FF00) for entire duration
Effects: NONE
"""
        
        video = generate_animation_video(reference_image, full_prompt, gemini_client, model_name)
        
        if video is None:
            return None, None, None, None, "❌ 视频生成失败: API 返回空结果"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = os.path.join(OUTPUT_DIR, f"full_{timestamp}")
        os.makedirs(output_base, exist_ok=True)
        
        # 保存视频
        video_path = os.path.join(output_base, "animation.mp4")
        video_data = gemini_client.files.download(file=video.video)
        with open(video_path, "wb") as f:
            f.write(video_data)
        
        # 保存参考图片
        reference_path = os.path.join(output_base, "reference_image.png")
        reference_image.save(reference_path)
        
        # 保存元数据
        metadata_path = os.path.join(output_base, "metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(f"=== SnowWeave 完整流程输出 ===\n\n")
            f.write(f"生成时间: {timestamp}\n")
            f.write(f"动作描述: {action}\n")
            f.write(f"使用模型: {model_name}\n\n")
            f.write(f"=== 视频生成参数 ===\n")
            f.write(f"提取时间范围: {start_time}s - {end_time}s\n")
            f.write(f"最大帧数: {max_frames}\n\n")
            f.write(f"=== 背景去除参数 ===\n")
            f.write(f"颜色容差: {tolerance}\n")
            f.write(f"自动裁剪: {auto_crop}\n")
            f.write(f"裁剪边距: {crop_padding}px\n\n")
            f.write(f"=== 输出文件 ===\n")
            f.write(f"视频: animation.mp4\n")
            f.write(f"参考图片: reference_image.png\n")
            f.write(f"原始提取帧: 1_extracted_frames/\n")
            f.write(f"去背景帧: 2_nobg_frames/\n")
            f.write(f"原始Sprite Sheet: 1_original_sprite_sheet.png\n")
            f.write(f"最终Sprite Sheet: 3_final_sprite_sheet.png\n")
        
        os.remove(temp_img_path)
        
        # 步骤2: 提取帧
        progress(0.3, desc="✂️ 步骤2/4: 提取帧...")
        
        frames = extract_frames_from_video_segment(
            video_path,
            float(start_time),
            float(end_time),
            int(max_frames)
        )
        
        frames_dir = os.path.join(output_base, "1_extracted_frames")
        save_individual_frames(frames, output_dir=frames_dir)
        
        original_sheet, _ = create_sprite_sheet(frames, frame_size=None)
        original_sheet_path = os.path.join(output_base, "1_original_sprite_sheet.png")
        original_sheet.save(original_sheet_path)
        
        # 步骤3: 去除背景
        progress(0.6, desc="🎨 步骤3/4: 去除背景...")
        
        nobg_dir = os.path.join(output_base, "2_nobg_frames")
        process_directory(
            frames_dir,
            output_dir=nobg_dir,
            tolerance=int(tolerance),
            num_workers=None,
            auto_crop=auto_crop,
            crop_padding=int(crop_padding)
        )
        
        # 步骤4: 创建最终sprite sheet
        progress(0.9, desc="📦 步骤4/4: 生成最终Sprite Sheet...")
        
        nobg_files = sorted([f for f in os.listdir(nobg_dir) if f.endswith('.png')])
        final_frames = [Image.open(os.path.join(nobg_dir, f)) for f in nobg_files]
        
        final_sheet, _ = create_sprite_sheet(final_frames, frame_size=None)
        final_sheet_path = os.path.join(output_base, "3_final_sprite_sheet.png")
        final_sheet.save(final_sheet_path)
        
        preview_images = final_frames[:8]
        
        progress(1.0, desc="✅ 完成!")
        
        summary = f"""✅ 完整流程执行完成!

📁 输出目录: {output_base}

生成的文件:
  📹 视频文件: animation.mp4
  🖼️ 参考图片: reference_image.png
  📝 元数据文件: metadata.txt
  1️⃣ 原始提取帧: 1_extracted_frames/ ({len(frames)} 帧)
  2️⃣ 去背景帧: 2_nobg_frames/ ({len(final_frames)} 帧)
  3️⃣ 原始Sprite Sheet: 1_original_sprite_sheet.png
  4️⃣ 最终Sprite Sheet: 3_final_sprite_sheet.png

🎮 可直接在游戏引擎中使用最终Sprite Sheet!
💾 可下载最终Sprite Sheet和参考图片
"""
        
        abs_video_path = os.path.abspath(video_path)
        abs_sheet_path = os.path.abspath(final_sheet_path)
        abs_ref_path = os.path.abspath(reference_path)
        
        return abs_video_path, abs_sheet_path, abs_ref_path, preview_images, summary
        
    except Exception as e:
        return None, None, None, None, f"❌ 错误: {str(e)}"

# 创建Gradio界面
with gr.Blocks(title="Sprite动画生成流水线") as app:
    gr.Markdown("""
    # 🎬 Sprite动画生成流水线
    ### AI驱动的游戏动画自动化生成工具
    """)
    
    with gr.Tabs():
        # Tab 0: API设置
        with gr.Tab("⚙️ 设置"):
            gr.Markdown("""
            ### 配置Gemini API密钥
            在使用视频生成功能前，需要先配置API密钥。
            
            获取API密钥: [Google AI Studio](https://aistudio.google.com/apikey)
            """)
            
            with gr.Row():
                with gr.Column():
                    api_key_input = gr.Textbox(
                        label="Gemini API密钥",
                        type="password",
                        placeholder="输入你的API密钥",
                        value="AIzaSyBhrZZhFDdKbI4uvA_xh6HscNi2p3FYEpc"
                    )
                    api_set_btn = gr.Button("💾 保存并验证", variant="primary", size="lg")
                
                with gr.Column():
                    api_status = gr.Textbox(label="状态", lines=3, interactive=False)
            
            api_set_btn.click(
                fn=lambda api_key: "❌ 请输入API密钥" if not api_key else initialize_api(api_key),
                inputs=[api_key_input],
                outputs=[api_status]
            )
            
            gr.Markdown("""
            ---
            ### 💡 提示
            - API密钥会在当前会话中保存，关闭浏览器后需重新输入
            - 视频生成功能需要API密钥，其他功能（提取帧、去背景）无需密钥
            - 获取密钥后，点击"保存并验证"即可使用
            """)
        
        # Tab 1: 生成视频
        with gr.Tab("🎨 生成视频"):
            gr.Markdown("""
            ### 使用AI生成角色动画视频
            1. 上传角色参考图片
            2. 描述想要的动作
            3. 等待AI生成动画视频
            """)
            
            with gr.Row():
                with gr.Column():
                    gen_image = gr.Image(label="上传角色图片", type="numpy")
                    gen_action = gr.Textbox(
                        label="动作描述",
                        placeholder="例如: walking, running, attack, jump",
                        value="walking animation, side view, loop"
                    )
                    gen_model = gr.Dropdown(
                        label="选择模型",
                        choices=list(AVAILABLE_MODELS.keys()),
                        value=DEFAULT_MODEL,
                        info="不同模型可能有不同的质量和安全策略"
                    )
                    gen_btn = gr.Button("🎬 生成动画视频", variant="primary", size="lg")
                
                with gr.Column():
                    gen_video_output = gr.Video(label="生成的视频", autoplay=False)
                    gen_image_output = gr.Image(label="参考图片", type="filepath")
                    gen_status = gr.Textbox(label="状态", lines=5)
            
            gen_btn.click(
                fn=generate_video_ui,
                inputs=[gen_image, gen_action, gen_model],
                outputs=[gen_video_output, gen_image_output, gen_status]
            )
        
        # Tab 2: 提取帧
        with gr.Tab("✂️ 提取帧"):
            gr.Markdown("""
            ### 从视频中提取Sprite帧
            1. 上传视频文件
            2. 设置提取参数（时间段、帧数）
            3. 自动生成Sprite Sheet
            """)
            
            with gr.Row():
                with gr.Column():
                    ext_video = gr.Video(label="上传视频")
                    
                    with gr.Row():
                        ext_start = gr.Number(label="开始时间(秒)", value=0, minimum=0)
                        ext_end = gr.Number(label="结束时间(秒)", value=0, minimum=0)
                    
                    ext_max_frames = gr.Slider(
                        label="最大帧数",
                        minimum=1,
                        maximum=100,
                        value=24,
                        step=1
                    )
                    
                    gr.Markdown("💡 提示: 开始和结束时间都设为0表示解析整个视频")
                    
                    ext_btn = gr.Button("✂️ 提取帧", variant="primary", size="lg")
                
                with gr.Column():
                    ext_sheet_output = gr.Image(label="Sprite Sheet")
                    ext_gallery = gr.Gallery(label="提取的帧", columns=4, height="auto")
                    ext_status = gr.Textbox(label="状态", lines=4)
            
            ext_btn.click(
                fn=extract_frames_ui,
                inputs=[ext_video, ext_start, ext_end, ext_max_frames],
                outputs=[ext_sheet_output, ext_gallery, ext_status]
            )
        
        # Tab 3: 去除背景
        with gr.Tab("🖼️ 去除背景"):
            gr.Markdown("""
            ### 自动去除绿幕背景
            1. 提供帧图片目录路径（或使用上一步的输出）
            2. 调整容差和裁剪参数
            3. 自动检测并移除背景
            """)
            
            with gr.Row():
                with gr.Column():
                    rm_input = gr.Textbox(
                        label="输入路径",
                        placeholder="输入帧图片目录的完整路径",
                        info="例如: gradio_outputs/frames_20231122_123456/frames"
                    )
                    
                    rm_tolerance = gr.Slider(
                        label="颜色容差",
                        minimum=0,
                        maximum=255,
                        value=30,
                        step=1,
                        info="值越大,移除的颜色范围越广"
                    )
                    
                    rm_auto_crop = gr.Checkbox(
                        label="自动裁剪透明边缘",
                        value=False
                    )
                    
                    rm_padding = gr.Slider(
                        label="裁剪边距(像素)",
                        minimum=0,
                        maximum=50,
                        value=0,
                        step=1
                    )
                    
                    rm_btn = gr.Button("🖼️ 去除背景", variant="primary", size="lg")
                
                with gr.Column():
                    rm_sheet_output = gr.Image(label="处理后的Sprite Sheet")
                    rm_gallery = gr.Gallery(label="处理后的帧", columns=4, height="auto")
                    rm_status = gr.Textbox(label="状态", lines=4)
            
            rm_btn.click(
                fn=remove_background_ui,
                inputs=[rm_input, rm_tolerance, rm_auto_crop, rm_padding],
                outputs=[rm_sheet_output, rm_gallery, rm_status]
            )
        
        # Tab 4: 完整流程
        with gr.Tab("🚀 完整流程"):
            gr.Markdown("""
            ### 一键完成全流程
            上传角色图片 → 生成视频 → 提取帧 → 去除背景 → 输出Sprite Sheet
            """)
            
            with gr.Row():
                with gr.Column():
                    full_image = gr.Image(label="上传角色图片", type="numpy")
                    full_action = gr.Textbox(
                        label="动作描述",
                        value="walking animation, side view, loop"
                    )
                    
                    gr.Markdown("#### 提取参数")
                    with gr.Row():
                        full_start = gr.Number(label="开始时间(秒)", value=0)
                        full_end = gr.Number(label="结束时间(秒)", value=1.0)
                    
                    full_max_frames = gr.Slider(
                        label="最大帧数",
                        minimum=1,
                        maximum=100,
                        value=8,
                        step=1
                    )
                    
                    gr.Markdown("#### 背景去除参数")
                    full_tolerance = gr.Slider(
                        label="颜色容差",
                        minimum=0,
                        maximum=255,
                        value=30,
                        step=1
                    )
                    
                    full_auto_crop = gr.Checkbox(
                        label="自动裁剪",
                        value=False
                    )
                    
                    full_padding = gr.Slider(
                        label="裁剪边距",
                        minimum=0,
                        maximum=50,
                        value=0,
                        step=1
                    )
                    
                    gr.Markdown("#### 模型选择")
                    full_model = gr.Dropdown(
                        label="视频生成模型",
                        choices=list(AVAILABLE_MODELS.keys()),
                        value=DEFAULT_MODEL,
                        info="选择不同的Veo模型"
                    )
                    
                    full_btn = gr.Button("🚀 开始完整流程", variant="primary", size="lg")
                
                with gr.Column():
                    full_video_output = gr.Video(label="生成的动画视频", autoplay=False)
                    full_sheet_output = gr.Image(label="最终Sprite Sheet", type="filepath")
                    full_ref_output = gr.Image(label="参考图片", type="filepath")
                    full_gallery = gr.Gallery(label="最终帧预览", columns=4, height="auto")
                    full_status = gr.Textbox(label="执行状态", lines=10)
            
            full_btn.click(
                fn=full_pipeline_ui,
                inputs=[
                    full_image, full_action, full_start, full_end, full_max_frames,
                    full_tolerance, full_auto_crop, full_padding, full_model
                ],
                outputs=[full_video_output, full_sheet_output, full_ref_output, full_gallery, full_status]
            )
    
    gr.Markdown("""
    ---
    ### 💡 使用提示
    - **生成视频**: 需要Gemini API密钥,视频生成约需2-5分钟
    - **提取帧**: 时间设为0-0表示解析整个视频,最大帧数会自动限制
    - **去除背景**: 自动检测四角背景色,调整容差可控制去除范围
    - **完整流程**: 一键完成所有步骤,适合快速生成游戏素材
    
    📁 所有输出保存在: `gradio_outputs/` 目录
    """)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sprite动画生成流水线 - Gradio界面')
    parser.add_argument('--share', action='store_true', help='创建公共分享链接(用于临时远程访问)')
    parser.add_argument('--server-name', default='0.0.0.0', help='服务器地址(默认: 0.0.0.0)')
    parser.add_argument('--server-port', type=int, default=7860, help='服务器端口(默认: 7860)')
    parser.add_argument('--root-path', default=None, help='反向代理根路径(例如: /gradio)')
    parser.add_argument('--max-file-size', default='100mb', help='最大文件上传大小(默认: 100mb)')
    args = parser.parse_args()
    
    print("="*70)
    print("  🎬 Sprite动画生成流水线 - Gradio界面")
    print("="*70)
    print("\n启动Gradio服务器...")
    print(f"  - 地址: {args.server_name}:{args.server_port}")
    if args.share:
        print("  - 模式: 公共分享 (share=True)")
    if args.root_path:
        print(f"  - 反向代理路径: {args.root_path}")
    print("\n按 Ctrl+C 停止服务器")
    print("="*70 + "\n")
    
    app.queue(
        max_size=20,
        api_open=False
    ).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
        max_file_size=args.max_file_size,
        allowed_paths=[OUTPUT_DIR],
        root_path=args.root_path
    )
