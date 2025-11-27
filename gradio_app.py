import gradio as gr
import os
import sys
import time
from datetime import datetime
from PIL import Image
import tempfile
import shutil
from google import genai


from lib.generate_sprite_animation import (
    load_reference_image,
    generate_animation_video
)
from lib.extract_sprite_frames import (
    extract_frames_from_video_segment,
    create_sprite_sheet,
    save_individual_frames
)
from lib.remove_background import (
    process_directory,
    process_image,
    detect_background_color
)

OUTPUT_DIR = "gradio_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_PATH = os.path.join(SCRIPT_DIR, "test.png")

gemini_client = None
current_api_key = ""

AVAILABLE_MODELS = {
    "veo-3.1-generate-preview": "Veo 3.1 (Preview, Latest / 预览版，最新)",
    "veo-3.1-fast-generate-preview": "Veo 3.1 Fast (Preview, Fast / 预览版，快速)",
    "veo-3.0-generate-001": "Veo 3.0 (Stable / 稳定版)",
    "veo-3.0-fast-generate-001": "Veo 3.0 Fast (Stable, Fast / 稳定版，快速)",
    "veo-2.0-generate-001": "Veo 2.0 (Legacy / 旧版)",
}
DEFAULT_MODEL = "veo-3.1-generate-preview"
current_model = DEFAULT_MODEL

LANGUAGES = {
    "zh": "中文",
    "en": "English"
}
current_language = "zh"

TRANSLATIONS = {
    "api_success": {"zh": "API密钥验证成功！", "en": "API key verified successfully!"},
    "api_failed": {"zh": "API密钥验证失败", "en": "API key verification failed"},
    "api_required": {"zh": "请先在设置中配置API密钥", "en": "Please configure API key in settings first"},
    "upload_image": {"zh": "请先上传图片", "en": "Please upload an image first"},
    "upload_video": {"zh": "请先上传视频", "en": "Please upload a video first"},
    "model_switched": {"zh": "已切换到模型", "en": "Switched to model"},
    "loading_image": {"zh": "正在加载图片...", "en": "Loading image..."},
    "generating_video": {"zh": "正在生成动画视频 (这可能需要几分钟)...", "en": "Generating animation video (this may take several minutes)..."},
    "downloading_video": {"zh": "正在下载视频...", "en": "Downloading video..."},
    "video_complete": {"zh": "视频生成完成!", "en": "Video generation complete!"},
    "video_failed": {"zh": "视频生成失败: API 返回空结果", "en": "Video generation failed: API returned empty result"},
    "extracting_frames": {"zh": "正在提取帧...", "en": "Extracting frames..."},
    "saving_frames": {"zh": "正在保存 {} 帧...", "en": "Saving {} frames..."},
    "no_frames": {"zh": "没有提取到帧", "en": "No frames extracted"},
    "extract_complete": {"zh": "提取完成!", "en": "Extraction complete!"},
    "processing_images": {"zh": "开始处理...", "en": "Starting processing..."},
    "processing_n_images": {"zh": "处理 {} 张图片...", "en": "Processing {} images..."},
    "processing_single": {"zh": "处理单张图片...", "en": "Processing single image..."},
    "creating_sprite": {"zh": "创建sprite sheet...", "en": "Creating sprite sheet..."},
    "complete": {"zh": "完成!", "en": "Complete!"},
    "bg_complete": {"zh": "背景去除完成!", "en": "Background removal complete!"},
    "step_generating": {"zh": "步骤1/4: 生成动画视频...", "en": "Step 1/4: Generating animation video..."},
    "step_extracting": {"zh": "步骤2/4: 提取帧...", "en": "Step 2/4: Extracting frames..."},
    "step_removing_bg": {"zh": "步骤3/4: 去除背景...", "en": "Step 3/4: Removing background..."},
    "step_final_sheet": {"zh": "步骤4/4: 生成最终Sprite Sheet...", "en": "Step 4/4: Generating final Sprite Sheet..."},
    "pipeline_complete": {"zh": "完整流程执行完成!", "en": "Full pipeline execution complete!"},
    "error": {"zh": "错误", "en": "Error"},
}

def t(key, *args):
    """获取翻译文本 / Get translated text"""
    text = TRANSLATIONS.get(key, {}).get(current_language, key)
    if args:
        return text.format(*args)
    return text

def set_language(lang):
    """设置界面语言 / Set interface language"""
    global current_language
    current_language = lang
    if lang == "zh":
        return "已切换到中文"
    else:
        return "Switched to English"

def clean_old_outputs(output_type="video"):
    """清理旧的输出文件"""
    try:
        pattern = f"{output_type}_*" if output_type else "*"
        for item in os.listdir(OUTPUT_DIR):
            item_path = os.path.join(OUTPUT_DIR, item)
            if os.path.isdir(item_path) and item.startswith(output_type):
                shutil.rmtree(item_path)
                print(f"Deleted old output / 已删除旧输出: {item_path}")
    except Exception as e:
        print(f"Error cleaning outputs / 清理输出时出错: {e}")

def initialize_api(api_key):
    """初始化Gemini API客户端"""
    global gemini_client, current_api_key
    try:
        gemini_client = genai.Client(api_key=api_key)
        current_api_key = api_key
        return "[OK] " + t("api_success")
    except Exception as e:
        return f"[ERROR] {t('api_failed')}: {str(e)}"

def get_current_api_key():
    """获取当前保存的API密钥"""
    return current_api_key

def set_model(model_name):
    """设置当前使用的模型"""
    global current_model
    current_model = model_name
    return f"[OK] {t('model_switched')}: {AVAILABLE_MODELS.get(model_name, model_name)}"

def generate_video_ui(image, action, model_name, duration):
    """生成动画视频"""
    if gemini_client is None:
        return None, None, "[ERROR] " + t("api_required")
    
    if image is None:
        return None, None, t("upload_image")
    
    try:
        # 清理旧的视频输出
        clean_old_outputs("video")
        
        yield None, None, t("loading_image")
        
        # 保存临时图片
        temp_img_path = os.path.join(tempfile.gettempdir(), f"temp_{int(time.time())}.png")
        Image.fromarray(image).save(temp_img_path)
        
        yield None, None, t("generating_video")
        
        # 加载图片
        reference_image = load_reference_image(temp_img_path)
        
        # 获取图片尺寸
        img_width, img_height = reference_image.size
        
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
- START IMMEDIATELY with the character visible - NO fade in effect No Irrelevant actions
- Character STAYS IN THE CENTER, does NOT move left or right across the screen
- Only the character's body/limbs animate, position remains FIXED
- Smooth, fluid animation with natural motion
- Complete {action} cycle IN PLACE
- Keep the exact same character design, colors, and art style
- Loop-able animation cycle
- Video dimensions MUST match the reference image: {img_width}x{img_height} pixels
- Maintain exact aspect ratio of {img_width}:{img_height}

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
Resolution: {img_width}x{img_height} (match reference image exactly)
Effects: NONE - no physics, lighting, or post-processing effects
"""
        
        # 生成视频
        video = generate_animation_video(reference_image, full_prompt, gemini_client, model_name, duration)
        
        if video is None:
            yield None, None, "[ERROR] " + t("video_failed")
            return
        
        yield None, None, t("downloading_video")
        
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
            f.write(f"Generation Time / 生成时间: {timestamp}\n")
            f.write(f"Action Description / 动作描述: {action}\n")
            f.write(f"Model Used / 使用模型: {model_name}\n")
            f.write(f"Video File / 视频文件: animation.mp4\n")
            f.write(f"Reference Image / 参考图片: reference_image.png\n")
        
        # 清理临时文件
        os.remove(temp_img_path)
        
        # 确保返回绝对路径
        abs_video_path = os.path.abspath(output_path)
        abs_ref_path = os.path.abspath(reference_path)
        
        if current_language == "zh":
            summary = f"""[OK] 视频生成完成!

输出目录: {output_dir}
视频文件: animation.mp4
参考图片: reference_image.png
元数据: metadata.txt

可直接下载视频和图片
"""
        else:
            summary = f"""[OK] Video generation complete!

Output directory: {output_dir}
Video file: animation.mp4
Reference image: reference_image.png
Metadata: metadata.txt

You can download the video and images directly
"""
        
        yield abs_video_path, abs_ref_path, summary
        
    except Exception as e:
        yield None, None, f"[ERROR] {t('error')}: {str(e)}"

def extract_frames_ui(video, start_time, end_time, max_frames):
    """从视频提取帧"""
    if video is None:
        return None, None, t("upload_video")
    
    try:
        yield None, None, t("extracting_frames")
        
        # 提取帧
        frames = extract_frames_from_video_segment(
            video,
            float(start_time),
            float(end_time),
            int(max_frames)
        )
        
        if not frames:
            yield None, None, "[ERROR] " + t("no_frames")
            return
        
        yield None, None, t("saving_frames", len(frames))
        
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
        
        if current_language == "zh":
            summary = f"[OK] 提取完成!\n共 {len(frames)} 帧\nSprite Sheet: {sheet_path}\n帧目录: {frames_dir}"
        else:
            summary = f"[OK] Extraction complete!\nTotal {len(frames)} frames\nSprite Sheet: {sheet_path}\nFrames directory: {frames_dir}"
        
        yield sheet_path, preview_images, summary
        
    except Exception as e:
        yield None, None, f"[ERROR] {t('error')}: {str(e)}"

def remove_background_ui(uploaded_files, tolerance, auto_crop, crop_padding, progress=gr.Progress()):
    """去除背景"""
    if uploaded_files is None or len(uploaded_files) == 0:
        return None, None, t("upload_image")
    
    try:
        progress(0, desc=t("processing_images"))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_DIR, f"nobg_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理上传的文件
        if isinstance(uploaded_files, list) and len(uploaded_files) > 1:
            # 多个文件
            progress(0.2, desc=t("processing_n_images", len(uploaded_files)))
            
            nobg_dir = os.path.join(output_dir, "frames")
            os.makedirs(nobg_dir, exist_ok=True)
            
            processed_images = []
            for i, file_path in enumerate(uploaded_files):
                progress_desc = f"{i+1}/{len(uploaded_files)}"
                if current_language == "zh":
                    progress(0.2 + 0.6 * (i / len(uploaded_files)), desc=f"处理 {progress_desc}...")
                else:
                    progress(0.2 + 0.6 * (i / len(uploaded_files)), desc=f"Processing {progress_desc}...")
                
                filename = os.path.basename(file_path)
                output_path = os.path.join(nobg_dir, filename)
                
                # 处理单张图片
                process_image(
                    file_path,
                    output_path=output_path,
                    tolerance=int(tolerance),
                    auto_crop=auto_crop,
                    crop_padding=int(crop_padding)
                )
                
                processed_images.append(Image.open(output_path))
            
            progress(0.8, desc=t("creating_sprite"))
            
            # 创建sprite sheet
            if processed_images:
                final_sheet, _ = create_sprite_sheet(processed_images, frame_size=None)
                sheet_path = os.path.join(output_dir, "sprite_sheet.png")
                final_sheet.save(sheet_path)
                
                preview_images = processed_images[:8]
            else:
                sheet_path = None
                preview_images = []
            
            progress(1.0, desc=t("complete"))
            
            if current_language == "zh":
                summary = f"[OK] 背景去除完成!\n共处理 {len(uploaded_files)} 张图片\nSprite Sheet: {sheet_path}\n帧目录: {nobg_dir}"
            else:
                summary = f"[OK] Background removal complete!\nProcessed {len(uploaded_files)} images\nSprite Sheet: {sheet_path}\nFrames directory: {nobg_dir}"
            
            return sheet_path, preview_images, summary
            
        else:
            # 单个文件
            progress(0.3, desc=t("processing_single"))
            
            file_path = uploaded_files[0] if isinstance(uploaded_files, list) else uploaded_files
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, filename)
            
            process_image(
                file_path,
                output_path=output_path,
                tolerance=int(tolerance),
                auto_crop=auto_crop,
                crop_padding=int(crop_padding)
            )
            
            progress(1.0, desc=t("complete"))
            
            result_img = Image.open(output_path)
            
            if current_language == "zh":
                summary = f"[OK] 背景去除完成!\n保存路径: {output_path}"
            else:
                summary = f"[OK] Background removal complete!\nSave path: {output_path}"
            
            return output_path, [result_img], summary
        
    except Exception as e:
        return None, None, f"[ERROR] {t('error')}: {str(e)}"

def full_pipeline_ui(image, action, start_time, end_time, max_frames, tolerance, auto_crop, crop_padding, model_name, duration, progress=gr.Progress()):
    """完整流水线"""
    if gemini_client is None:
        return None, None, None, None, "[ERROR] " + t("api_required")
    
    if image is None:
        return None, None, None, None, t("upload_image")
    
    try:
        # 清理旧的完整流程输出
        clean_old_outputs("full")
        
        # 步骤1: 生成视频
        progress(0, desc=t("step_generating"))
        
        temp_img_path = os.path.join(tempfile.gettempdir(), f"temp_{int(time.time())}.png")
        Image.fromarray(image).save(temp_img_path)
        
        reference_image = load_reference_image(temp_img_path)
        
        # 获取图片尺寸
        img_width, img_height = reference_image.size
        
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
- Video dimensions MUST match the reference image: {img_width}x{img_height} pixels
- Maintain exact aspect ratio of {img_width}:{img_height}

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
Resolution: {img_width}x{img_height} (match reference image exactly)
Effects: NONE
"""
        
        video = generate_animation_video(reference_image, full_prompt, gemini_client, model_name, duration)
        
        if video is None:
            return None, None, None, None, "[ERROR] " + t("video_failed")
        
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
            f.write(f"=== SnowWeave Full Pipeline Output / SnowWeave 完整流程输出 ===\n\n")
            f.write(f"Generation Time / 生成时间: {timestamp}\n")
            f.write(f"Action Description / 动作描述: {action}\n")
            f.write(f"Model Used / 使用模型: {model_name}\n\n")
            f.write(f"=== Video Generation Parameters / 视频生成参数 ===\n")
            f.write(f"Extraction Time Range / 提取时间范围: {start_time}s - {end_time}s\n")
            f.write(f"Max Frames / 最大帧数: {max_frames}\n\n")
            f.write(f"=== Background Removal Parameters / 背景去除参数 ===\n")
            f.write(f"Color Tolerance / 颜色容差: {tolerance}\n")
            f.write(f"Auto Crop / 自动裁剪: {auto_crop}\n")
            f.write(f"Crop Padding / 裁剪边距: {crop_padding}px\n\n")
            f.write(f"=== Output Files / 输出文件 ===\n")
            f.write(f"Video / 视频: animation.mp4\n")
            f.write(f"Reference Image / 参考图片: reference_image.png\n")
            f.write(f"Original Extracted Frames / 原始提取帧: 1_extracted_frames/\n")
            f.write(f"No-Background Frames / 去背景帧: 2_nobg_frames/\n")
            f.write(f"Original Sprite Sheet / 原始Sprite Sheet: 1_original_sprite_sheet.png\n")
            f.write(f"Final Sprite Sheet / 最终Sprite Sheet: 3_final_sprite_sheet.png\n")
        
        os.remove(temp_img_path)
        
        # 步骤2: 提取帧
        progress(0.3, desc=t("step_extracting"))
        
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
        progress(0.6, desc=t("step_removing_bg"))
        
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
        progress(0.9, desc=t("step_final_sheet"))
        
        nobg_files = sorted([f for f in os.listdir(nobg_dir) if f.endswith('.png')])
        final_frames = [Image.open(os.path.join(nobg_dir, f)) for f in nobg_files]
        
        final_sheet, _ = create_sprite_sheet(final_frames, frame_size=None)
        final_sheet_path = os.path.join(output_base, "3_final_sprite_sheet.png")
        final_sheet.save(final_sheet_path)
        
        preview_images = final_frames[:8]
        
        progress(1.0, desc=t("complete"))
        
        if current_language == "zh":
            summary = f"""[OK] 完整流程执行完成!

输出目录: {output_base}

生成的文件:
  视频文件: animation.mp4
  参考图片: reference_image.png
  元数据文件: metadata.txt
  1. 原始提取帧: 1_extracted_frames/ ({len(frames)} 帧)
  2. 去背景帧: 2_nobg_frames/ ({len(final_frames)} 帧)
  3. 原始Sprite Sheet: 1_original_sprite_sheet.png
  4. 最终Sprite Sheet: 3_final_sprite_sheet.png

可直接在游戏引擎中使用最终Sprite Sheet!
可下载最终Sprite Sheet和参考图片
"""
        else:
            summary = f"""[OK] Full pipeline execution complete!

Output directory: {output_base}

Generated files:
  Video file: animation.mp4
  Reference image: reference_image.png
  Metadata file: metadata.txt
  1. Original extracted frames: 1_extracted_frames/ ({len(frames)} frames)
  2. No-background frames: 2_nobg_frames/ ({len(final_frames)} frames)
  3. Original Sprite Sheet: 1_original_sprite_sheet.png
  4. Final Sprite Sheet: 3_final_sprite_sheet.png

Ready to use in game engines!
You can download the final Sprite Sheet and reference image
"""
        
        abs_video_path = os.path.abspath(video_path)
        abs_sheet_path = os.path.abspath(final_sheet_path)
        abs_ref_path = os.path.abspath(reference_path)
        
        return abs_video_path, abs_sheet_path, abs_ref_path, preview_images, summary
        
    except Exception as e:
        return None, None, None, None, f"[ERROR] {t('error')}: {str(e)}"

# 创建Gradio界面
with gr.Blocks(title="SnowWeave") as app:
    # 标题和语言选择
    with gr.Row():
        with gr.Column(scale=4):
            title_md = gr.Markdown("""
    # SnowWeave
    ### Sprite Animation Generation Pipeline / Sprite流水线
    """)
        with gr.Column(scale=1):
            lang_dropdown = gr.Dropdown(
                choices=list(LANGUAGES.keys()),
                value=current_language,
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
            gr.Markdown("""
            ### Configure Gemini API Key / 配置Gemini API密钥
            Please configure the API key before using video generation features. / 在使用视频生成功能前，需要先配置API密钥。
            
            Get API Key / 获取API密钥: [Google AI Studio](https://aistudio.google.com/apikey)
            """)
            
            with gr.Row():
                with gr.Column():
                    api_key_input = gr.Textbox(
                        label="Gemini API Key / Gemini API密钥",
                        type="password",
                        placeholder="Enter your API key / 输入你的API密钥",
                        value="AIzaSyBhrZZhFDdKbI4uvA_xh6HscNi2p3FYEpc"
                    )
                    api_set_btn = gr.Button("Save and Verify / 保存并验证", variant="primary", size="lg")
                
                with gr.Column():
                    api_status = gr.Textbox(label="Status / 状态", lines=3, interactive=False)
            
            def validate_api(api_key):
                if not api_key:
                    if current_language == "zh":
                        return "[ERROR] 请输入API密钥"
                    else:
                        return "[ERROR] Please enter API key"
                return initialize_api(api_key)
            
            api_set_btn.click(
                fn=validate_api,
                inputs=[api_key_input],
                outputs=[api_status]
            )
            

        
        # Tab 1: 生成视频
        with gr.Tab("Generate Video / 生成视频"):
            gr.Markdown("""
            ### Generate Character Animation with AI / 使用AI生成角色动画视频
            1. Upload character reference image / 上传角色参考图片
            2. Describe the desired action / 描述想要的动作
            3. Wait for AI to generate animation video / 等待AI生成动画视频
            """)
            
            with gr.Row():
                with gr.Column():
                    gen_image = gr.Image(label="Upload Character Image / 上传角色图片", type="numpy", value=DEFAULT_IMAGE_PATH if os.path.exists(DEFAULT_IMAGE_PATH) else None)
                    gen_action = gr.Textbox(
                        label="Action Description / 动作描述",
                        placeholder="Example / 例如: walking, running, attack, jump",
                        value="walking animation, side view, loop"
                    )
                    gen_model = gr.Dropdown(
                        label="Select Model / 选择模型",
                        choices=list(AVAILABLE_MODELS.keys()),
                        value=DEFAULT_MODEL,
                        info="Different models may have different quality and safety policies / 不同模型可能有不同的质量和安全策略"
                    )
                    gen_duration = gr.Slider(
                        label="Video Duration (seconds) / 视频长度(秒)",
                        minimum=4,
                        maximum=8,
                        value=6,
                        step=1,
                        info="Video generation duration, API limited to 4-8 seconds / 视频生成的时长,API限制4-8秒"
                    )
                    gen_btn = gr.Button("Generate Animation Video / 生成动画视频", variant="primary", size="lg")
                
                with gr.Column():
                    gen_video_output = gr.Video(label="Generated Video / 生成的视频", autoplay=False)
                    gen_image_output = gr.Image(label="Reference Image / 参考图片", type="filepath")
                    gen_status = gr.Textbox(label="Status / 状态", lines=5)
            
            gen_btn.click(
                fn=generate_video_ui,
                inputs=[gen_image, gen_action, gen_model, gen_duration],
                outputs=[gen_video_output, gen_image_output, gen_status]
            )
        
        # Tab 2: 提取帧
        with gr.Tab("Extract Frames / 提取帧"):
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
        
        # Tab 3: 去除背景
        with gr.Tab("Remove Background / 去除背景"):
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
        
        # Tab 4: 完整流程
        with gr.Tab("Full Pipeline / 完整流程"):
            gr.Markdown("""
            ### One-Click Complete Process / 一键完成全流程
            Upload character image → Generate video → Extract frames → Remove background → Output Sprite Sheet /
            上传角色图片 → 生成视频 → 提取帧 → 去除背景 → 输出Sprite Sheet
            """)
            
            with gr.Row():
                with gr.Column():
                    full_image = gr.Image(label="Upload Character Image / 上传角色图片", type="numpy", value=DEFAULT_IMAGE_PATH if os.path.exists(DEFAULT_IMAGE_PATH) else None)
                    full_action = gr.Textbox(
                        label="Action Description / 动作描述",
                        value="walking animation, side view, loop"
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
                    
                    gr.Markdown("#### Model Selection / 模型选择")
                    full_model = gr.Dropdown(
                        label="Video Generation Model / 视频生成模型",
                        choices=list(AVAILABLE_MODELS.keys()),
                        value=DEFAULT_MODEL,
                        info="Select different Veo model / 选择不同的Veo模型"
                    )
                    full_duration = gr.Slider(
                        label="Video Duration (sec) / 视频长度(秒)",
                        minimum=4,
                        maximum=8,
                        value=6,
                        step=1,
                        info="Video generation duration, API limited to 4-8 sec / 视频生成的时长,API限制4-8秒"
                    )
                    
                    full_btn = gr.Button("Start Full Pipeline / 开始完整流程", variant="primary", size="lg")
                
                with gr.Column():
                    full_video_output = gr.Video(label="Generated Animation Video / 生成的动画视频", autoplay=False)
                    full_sheet_output = gr.Image(label="Final Sprite Sheet / 最终Sprite Sheet", type="filepath")
                    full_ref_output = gr.Image(label="Reference Image / 参考图片", type="filepath")
                    full_gallery = gr.Gallery(label="Final Frames Preview / 最终帧预览", columns=4, height="auto")
                    full_status = gr.Textbox(label="Execution Status / 执行状态", lines=10)
            
            full_btn.click(
                fn=full_pipeline_ui,
                inputs=[
                    full_image, full_action, full_start, full_end, full_max_frames,
                    full_tolerance, full_auto_crop, full_padding, full_model, full_duration
                ],
                outputs=[full_video_output, full_sheet_output, full_ref_output, full_gallery, full_status]
            )
    
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sprite Animation Generation Pipeline - Gradio Interface / Sprite动画生成流水线 - Gradio界面')
    parser.add_argument('--share', action='store_true', help='Create public share link (for temporary remote access) / 创建公共分享链接(用于临时远程访问)')
    parser.add_argument('--server-name', default='0.0.0.0', help='Server address (default: 0.0.0.0) / 服务器地址(默认: 0.0.0.0)')
    parser.add_argument('--server-port', type=int, default=7860, help='Server port (default: 7860) / 服务器端口(默认: 7860)')
    parser.add_argument('--root-path', default=None, help='Reverse proxy root path (e.g.: /gradio) / 反向代理根路径(例如: /gradio)')
    parser.add_argument('--max-file-size', default='100mb', help='Max file upload size (default: 100mb) / 最大文件上传大小(默认: 100mb)')
    args = parser.parse_args()
    
    print("="*70)
    print("  Sprite Animation Generation Pipeline - Gradio Interface")
    print("  Sprite动画生成流水线 - Gradio界面")
    print("="*70)
    print("\nStarting Gradio server / 启动Gradio服务器...")
    print(f"  - Address / 地址: {args.server_name}:{args.server_port}")
    if args.share:
        print("  - Mode / 模式: Public share / 公共分享 (share=True)")
    if args.root_path:
        print(f"  - Reverse proxy path / 反向代理路径: {args.root_path}")
    print("\nPress Ctrl+C to stop server / 按 Ctrl+C 停止服务器")
    print("="*70 + "\n")
    
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
