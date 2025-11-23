"""
使用 Gemini Veo 生成角色动画，并切片成sprite帧
需要设置环境变量 GEMINI_API_KEY

工作流程:
1. 读取已有的角色素材图片
2. 使用 Veo 3.1 生成角色动作动画视频
3. 从视频中提取关键帧
4. 将帧切片成sprite sheet
"""

import os
import sys
import time
import base64
from io import BytesIO
from google import genai
from google.genai.types import Part, Image as GenAIImage, GenerateVideosConfig, HarmCategory, HarmBlockThreshold
from PIL import Image
import cv2
import numpy as np

# 配置 API 密钥（可选，如果不通过参数传入）
api_key = "AIzaSyBhrZZhFDdKbI4uvA_xh6HscNi2p3FYEpc"

# 创建默认客户端（如果有环境变量）
client = None
if api_key:
    client = genai.Client(api_key=api_key)

def load_reference_image(image_path):
    """加载参考图片"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片: {image_path}")
    
    # 使用 PIL 加载图片
    img = Image.open(image_path)
    return img

def generate_animation_video(reference_image, action_prompt, api_client=None, model_name="veo-2.0-generate-001"):
    """使用 Veo 生成动画视频"""
    # 使用传入的 client 或全局 client
    _client = api_client or client
    if _client is None:
        raise ValueError("未提供 API 客户端，请传入 api_client 参数或设置环境变量 GEMINI_API_KEY")
    
    print(f"正在生成动画: {action_prompt}")
    print(f"使用模型: {model_name}")
    
    # 将 PIL Image 转换为字节流并编码为base64
    img_bytes = BytesIO()
    reference_image.save(img_bytes, format='PNG')
    img_data = img_bytes.getvalue()
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    
    # 创建符合API要求的Image对象
    print("正在准备图片...")
    veo_image = GenAIImage(
        image_bytes=img_data,
        mime_type='image/png'
    )
    
    # 使用 Veo 生成视频
    print("开始生成视频 (4秒时长)...")
    
    # 尝试设置最宽松的安全设置
    try:
        operation = _client.models.generate_videos(
            model=model_name,
            prompt=action_prompt,
            image=veo_image,
            config=GenerateVideosConfig(
                duration_seconds=5,  # 最短时长为4秒
                safety_settings=[
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT", 
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
            )
        )
        print("   (已设置宽松安全过滤)")
    except Exception as e:
        # 如果不支持 safety_settings，则使用默认配置
        print(f"   (注意: 安全设置不被支持，使用默认配置)")
        operation = _client.models.generate_videos(
            model=model_name,
            prompt=action_prompt,
            image=veo_image,
            config=GenerateVideosConfig(
                duration_seconds=5
            )
        )
    
    # 轮询操作状态直到视频准备好
    print("等待视频生成完成...")
    while not operation.done:
        print(".", end="", flush=True)
        time.sleep(10)
        operation = _client.operations.get(operation)
    
    print("\n✓ 视频生成完成!")
    
    # 检查是否有错误
    if operation.error:
        error_msg = f"API 错误 (代码 {operation.error.get('code')}): {operation.error.get('message')}"
        print(f"\n❌ {error_msg}")
        
        # 特殊提示
        if operation.error.get('code') == 3:
            print("\n💡 提示: 这是安全设置问题。可能的原因:")
            print("   - 图片包含人物面部，触发了安全过滤")
            print("   - 建议使用非人物角色（动物、机器人、抽象角色等）")
            print("   - 或使用简化的、卡通化的人物图像")
        
        raise RuntimeError(error_msg)
    
    # 检查操作是否成功
    if operation.response is None:
        print(f"ERROR: operation.response 为 None，但没有 error 信息")
        raise RuntimeError(f"视频生成失败: operation.response 为 None（原因未知）")
    
    if not hasattr(operation.response, 'generated_videos'):
        print(f"ERROR: response 没有 generated_videos 属性")
        raise RuntimeError(f"视频生成失败: 未找到 generated_videos 属性")
    
    if not operation.response.generated_videos:
        print(f"ERROR: generated_videos 为空")
        raise RuntimeError(f"视频生成失败: generated_videos 为空列表")
    
    print(f"✓ 成功获取 {len(operation.response.generated_videos)} 个视频")
    return operation.response.generated_videos[0]

def extract_frames_from_video(video_path, num_frames=8):
    """从视频中提取指定数量的均匀分布的帧"""
    print(f"正在从视频提取 {num_frames} 帧...")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算帧索引，均匀分布
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 转换 BGR 到 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    print(f"✓ 提取了 {len(frames)} 帧")
    return frames

def create_sprite_sheet(frames, frame_size=(64, 64)):
    """将帧组合成横向 sprite sheet"""
    print(f"正在创建 sprite sheet (每帧 {frame_size[0]}x{frame_size[1]})...")
    
    # 调整每一帧的大小
    resized_frames = [frame.resize(frame_size, Image.Resampling.LANCZOS) for frame in frames]
    
    # 创建 sprite sheet (横向排列)
    sheet_width = frame_size[0] * len(frames)
    sheet_height = frame_size[1]
    sprite_sheet = Image.new('RGBA', (sheet_width, sheet_height), (0, 0, 0, 0))
    
    # 粘贴每一帧
    for i, frame in enumerate(resized_frames):
        x_offset = i * frame_size[0]
        # 转换为 RGBA 如果需要
        if frame.mode != 'RGBA':
            frame = frame.convert('RGBA')
        sprite_sheet.paste(frame, (x_offset, 0))
    
    print("✓ Sprite sheet 创建完成!")
    return sprite_sheet, resized_frames

def save_individual_frames(frames, output_dir="frames"):
    """保存单独的帧图片"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"正在保存单独帧到 {output_dir}/ ...")
    
    for i, frame in enumerate(frames):
        output_path = os.path.join(output_dir, f"frame_{i:03d}.png")
        frame.save(output_path)
    
    print(f"✓ 保存了 {len(frames)} 个帧")

def main():
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python generate_sprite_animation.py <图片路径> [动作描述]")
        print("\n示例:")
        print('  python generate_sprite_animation.py character.png "walking animation"')
        print('  python generate_sprite_animation.py goblin.png "running and jumping"')
        sys.exit(1)
    
    reference_image_path = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else "walking animation, side view, loop"
    
    try:
        # 1. 加载参考图片
        print(f"加载参考图片: {reference_image_path}")
        reference_image = load_reference_image(reference_image_path)
        print(f"✓ 图片已加载: {reference_image.size}")
        
        # 2. 生成动画提示词 - 流畅动画，原地移动，纯绿背景用于抠图
        full_prompt = f"""
Create a smooth sprite animation of the character {action} IN PLACE (not moving across screen).

CRITICAL REQUIREMENTS:
- START IMMEDIATELY with the character visible - NO fade in effect
- Character MUST face RIGHT and perform the animation IN THE SAME POSITION
- Character STAYS IN THE CENTER, does NOT move left or right across the screen
- Only the character's body/limbs animate, position remains FIXED
- Smooth, fluid animation with natural motion
- Complete {action} cycle IN PLACE
- Pure side view with character facing RIGHT direction
- Keep the exact same character design, colors, and art style
- Loop-able animation cycle

VISUAL STYLE REQUIREMENTS:
- NO physics effects (no particles, debris, dust, etc.)
- NO lighting effects (no shadows, highlights, glows, reflections)
- NO post-processing effects (no blur, bloom, color grading)
- Flat, clean animation with solid colors only
- Simple sprite animation style without any special effects

BACKGROUND REQUIREMENTS FOR POST-PRODUCTION:
- Background MUST be PURE CHROMA GREEN (#00FF00, RGB 0,255,0)
- Solid, uniform green color across entire background
- NO gradients, NO textures, NO variations in the green
- This green screen is SPECIFICALLY for video editing and background removal in post-production
- The green background will be keyed out and replaced later
- Character should NOT contain any green colors to avoid keying issues
- Keep background perfectly flat and uniform for clean chroma key

IMPORTANT: 
- BEGIN: Start with character fully visible immediately, NO fade in
- BACKGROUND: Solid chroma green (#00FF00) throughout entire video for post-production keying
- END: After the animation cycle completes (around 2 seconds), character disappears but background stays green
- Do NOT use any fade effects - instant start, character vanishes at end, green background remains

Style: Clean pixel art / 2D game sprite animation with smooth motion, no effects
Camera: Fixed, character stays in center and animates in place
Background: Pure chroma green (#00FF00) for entire duration - FOR POST-PRODUCTION EDITING
Transitions: None - instant start, instant character removal at end, green background constant
Effects: NONE - no physics, lighting, or post-processing effects
"""
        
        # 3. 生成动画视频
        if client is None:
            raise ValueError("未设置 GEMINI_API_KEY 环境变量")
        video = generate_animation_video(reference_image, full_prompt, client)
        
        # 4. 下载视频
        temp_video_path = "temp_animation.mp4"
        print(f"正在下载视频到 {temp_video_path}...")
        
        # 使用 client.files.download 下载视频
        video_data = client.files.download(file=video.video)
        
        # 保存视频数据
        with open(temp_video_path, "wb") as f:
            f.write(video_data)
        
        print("✓ 视频已下载")
    except Exception as e:
        print(f"\n× 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
