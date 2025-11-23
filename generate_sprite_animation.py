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
from google.genai.types import Part, Image as GenAIImage, GenerateVideosConfig
from PIL import Image
import cv2
import numpy as np

# 配置 API 密钥
api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyAYEeSNAB9ikYV4GoTK-5CM51yE5ljAQYs")
if not api_key:
    raise ValueError("请设置环境变量 GEMINI_API_KEY")

# 创建客户端
client = genai.Client(api_key=api_key)

def load_reference_image(image_path):
    """加载参考图片"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片: {image_path}")
    
    # 使用 PIL 加载图片
    img = Image.open(image_path)
    return img

def generate_animation_video(reference_image, action_prompt):
    """使用 Veo 3.1 生成动画视频"""
    print(f"正在生成动画: {action_prompt}")
    
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
    
    # 使用 Veo 3.1 生成视频，限制时长为4秒（最短）
    print("开始生成视频 (4秒时长)...")
    operation = client.models.generate_videos(
        model="veo-2.0-generate-001",
        prompt=action_prompt,
        image=veo_image,
        config=GenerateVideosConfig(
            duration_seconds=5  # 最短时长为4秒
        )
    )
    
    # 轮询操作状态直到视频准备好
    print("等待视频生成完成...")
    while not operation.done:
        print(".", end="", flush=True)
        time.sleep(10)
        operation = client.operations.get(operation)
    
    print("\n✓ 视频生成完成!")
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
        video = generate_animation_video(reference_image, full_prompt)
        
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
