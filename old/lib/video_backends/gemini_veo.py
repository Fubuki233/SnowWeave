"""
Google Gemini Veo Video Backend
Google Gemini Veo 视频生成后端
"""

import os
import time
import base64
import tempfile
from io import BytesIO
from typing import Optional
from PIL import Image

from .base import VideoBackend, VideoResult


class GeminiVeoBackend(VideoBackend):
    """Google Gemini Veo 视频生成后端"""
    
    name = "gemini"
    
    available_models = {
        "veo-3.1-generate-preview": "Veo 3.1 (Preview, Latest)",
        "veo-3.1-fast-generate-preview": "Veo 3.1 Fast (Preview)",
        "veo-3.0-generate-001": "Veo 3.0 (Stable)",
        "veo-3.0-fast-generate-001": "Veo 3.0 Fast (Stable)",
        "veo-2.0-generate-001": "Veo 2.0 (Legacy)",
    }
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化 Gemini 客户端"""
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            print(f"[GeminiVeoBackend] Failed to initialize client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """检查后端是否可用"""
        return self.client is not None
    
    def generate_video(
        self,
        reference_image: Image.Image,
        prompt: str,
        model_name: Optional[str] = None,
        duration: int = 5,
        **kwargs
    ) -> VideoResult:
        """
        使用 Veo 生成视频
        
        Args:
            reference_image: 参考图片
            prompt: 提示词
            model_name: 模型名称
            duration: 视频时长（秒），范围 4-8
        
        Returns:
            VideoResult: 视频生成结果
        """
        from google.genai.types import Image as GenAIImage, GenerateVideosConfig
        
        if not self.is_available():
            raise RuntimeError("Gemini client not initialized")
        
        model_name = model_name or self.get_default_model()
        
        # 限制时长范围
        duration = max(4, min(8, int(duration)))
        
        img_width, img_height = reference_image.size
        print(f"[GeminiVeoBackend] Image size: {img_width}x{img_height}")
        print(f"[GeminiVeoBackend] Model: {model_name}")
        print(f"[GeminiVeoBackend] Duration: {duration}s")
        
        # 将 PIL Image 转换为字节流
        img_bytes = BytesIO()
        reference_image.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        # 创建 Veo Image 对象
        veo_image = GenAIImage(
            image_bytes=img_data,
            mime_type='image/png'
        )
        
        print("[GeminiVeoBackend] Generating video...")
        
        # 调用 API 生成视频
        operation = self.client.models.generate_videos(
            model=model_name,
            prompt=prompt,
            image=veo_image,
            config=GenerateVideosConfig(
                duration_seconds=duration,
            )
        )
        
        # 等待生成完成
        print("[GeminiVeoBackend] Waiting for generation...", end="", flush=True)
        while not operation.done:
            print(".", end="", flush=True)
            time.sleep(10)
            operation = self.client.operations.get(operation)
        print()
        
        # 检查错误
        if operation.error:
            error_msg = f"API Error (code {operation.error.get('code')}): {operation.error.get('message')}"
            raise RuntimeError(error_msg)
        
        if operation.response is None or not hasattr(operation.response, 'generated_videos'):
            raise RuntimeError("Video generation failed: No response")
        
        if not operation.response.generated_videos:
            raise RuntimeError("Video generation failed: Empty video list")
        
        video_obj = operation.response.generated_videos[0]
        print(f"[GeminiVeoBackend] Video generated successfully")
        
        # 下载视频数据
        video_data = self.client.files.download(file=video_obj.video)
        
        # 保存到临时文件
        temp_path = os.path.join(tempfile.gettempdir(), f"veo_video_{int(time.time())}.mp4")
        with open(temp_path, "wb") as f:
            f.write(video_data)
        
        return VideoResult(
            video_path=temp_path,
            video_data=video_data,
            duration=duration,
            width=img_width,
            height=img_height,
            backend=self.name,
            raw_response=video_obj
        )
