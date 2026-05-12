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

api_key = "AIzaSyBhrZZhFDdKbI4uvA_xh6HscNi2p3FYEpc"

client = None
if api_key:
    client = genai.Client(api_key=api_key)

def load_reference_image(image_path):
    """加载参考图片"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片: {image_path}")
    
    img = Image.open(image_path)
    return img

def generate_animation_video(reference_image, action_prompt, api_client=None, model_name="veo-2.0-generate-001", duration_seconds=5):
    """使用 Veo 生成动画视频"""
    _client = api_client or client
    if _client is None:
        raise ValueError("未提供 API 客户端，请传入 api_client 参数或设置环境变量 GEMINI_API_KEY")
    
    duration_seconds = int(duration_seconds)
    if duration_seconds < 4:
        duration_seconds = 4
    elif duration_seconds > 8:
        duration_seconds = 8
    
    # 获取输入图片的尺寸
    img_width, img_height = reference_image.size
    print(f"输入图片尺寸: {img_width}x{img_height}")
    
    print(f"正在生成动画: {action_prompt}")
    print(f"使用模型: {model_name}")
    print(f"视频长度: {duration_seconds}秒")
    print(f"视频尺寸: {img_width}x{img_height}")
    
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
    print(f"开始生成视频 ({duration_seconds}秒时长)...")
    print(f"注意: 输入图片尺寸为 {img_width}x{img_height}, API将自动处理尺寸")
    
    # 尝试设置最宽松的安全设置

    operation = _client.models.generate_videos(
            model=model_name,
            prompt=action_prompt,
            image=veo_image,
            config=GenerateVideosConfig(
                duration_seconds=duration_seconds,
            )
        )

    
    print("等待视频生成完成...")
    while not operation.done:
        print(".", end="", flush=True)
        time.sleep(10)
        operation = _client.operations.get(operation)
    
    print("\n 视频生成完成!")
    
    # 检查是否有错误
    if operation.error:
        error_msg = f"API 错误 (代码 {operation.error.get('code')}): {operation.error.get('message')}"
        print(f"\n {error_msg}")
        
        # 特殊提示
        if operation.error.get('code') == 3:
            print("\n提示: 这是安全设置问题。可能的原因:")
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
    
    print(f"成功获取 {len(operation.response.generated_videos)} 个视频")
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
    print(f"提取了 {len(frames)} 帧")
    return frames

def create_sprite_sheet(frames, frame_size=(64, 64)):
    """将帧组合成横向 sprite sheet"""
    print(f"正在创建 sprite sheet (每帧 {frame_size[0]}x{frame_size[1]})...")
    
    resized_frames = [frame.resize(frame_size, Image.Resampling.LANCZOS) for frame in frames]
    
    sheet_width = frame_size[0] * len(frames)
    sheet_height = frame_size[1]
    sprite_sheet = Image.new('RGBA', (sheet_width, sheet_height), (0, 0, 0, 0))
    
    for i, frame in enumerate(resized_frames):
        x_offset = i * frame_size[0]
        if frame.mode != 'RGBA':
            frame = frame.convert('RGBA')
        sprite_sheet.paste(frame, (x_offset, 0))
    
    return sprite_sheet, resized_frames

def save_individual_frames(frames, output_dir="frames"):
    """保存单独的帧图片"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"正在保存单独帧到 {output_dir}/ ...")
    
    for i, frame in enumerate(frames):
        output_path = os.path.join(output_dir, f"frame_{i:03d}.png")
        frame.save(output_path)
    
    print(f" 保存了 {len(frames)} 个帧")
