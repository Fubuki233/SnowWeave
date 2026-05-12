"""
Seedream Image Generation Backend
豆包 Seedream 4.5 图片生成后端

用于生成多视角角色图片
"""

import os
import time
import base64
import requests
from io import BytesIO
from typing import Optional, List, Tuple
from PIL import Image
from dataclasses import dataclass


@dataclass
class ImageResult:
    """图片生成结果"""
    image: Image.Image  # PIL Image
    image_data: bytes  # 原始图片数据
    size: str  # 尺寸信息
    raw_response: dict = None  # 原始响应


class SeedreamBackend:
    """豆包 Seedream 4.5 图片生成后端"""
    
    API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
    MODEL_ID = "doubao-seedream-4-5-251128"
    
    def __init__(self, api_key: str, **kwargs):
        """
        初始化 Seedream 后端
        
        Args:
            api_key: 火山引擎 ARK API Key
            **kwargs: 其他配置
        """
        self.api_key = api_key
        self.watermark = kwargs.get("watermark", False)
    
    def is_available(self) -> bool:
        """检查后端是否可用"""
        return bool(self.api_key)
    
    def _get_headers(self) -> dict:
        """获取 API 请求头"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _image_to_base64_url(self, image: Image.Image, max_size: int = 1024) -> str:
        """
        将 PIL Image 转换为 base64 data URL
        
        Args:
            image: PIL Image
            max_size: 最大边长，超过会缩放
        """
        # 如果图片太大，先缩放
        width, height = image.size
        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"[SeedreamBackend] Resized image from {width}x{height} to {new_width}x{new_height}")
        
        img_bytes = BytesIO()
        # 确保是 RGB 模式（JPEG 不支持 RGBA）
        if image.mode == 'RGBA':
            # 创建白色背景
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 使用 JPEG 格式减小数据量
        image.save(img_bytes, format='JPEG', quality=85)
        img_data = img_bytes.getvalue()
        b64_str = base64.b64encode(img_data).decode('utf-8')
        
        size_kb = len(img_data) / 1024
        print(f"[SeedreamBackend] Image encoded: {size_kb:.1f} KB")
        
        return f"data:image/jpeg;base64,{b64_str}"
    
    def generate_multi_view_image(
        self,
        reference_image: Image.Image,
        prompt: str,
        size: str = "1024x512",
        timeout: int = 120
    ) -> ImageResult:
        """
        生成多视角图片（将参考图放入网格，生成四个视角）
        Args:
            reference_image: 参考图片（已放入网格的图片）
            prompt: 提示词
            size: 输出尺寸
            timeout: 超时时间（秒）
        
        Returns:
            ImageResult: 图片生成结果
        """
        if not self.is_available():
            raise RuntimeError("Seedream API key not provided")
        
        url = f"{self.API_BASE}/images/generations"
        
        # 构建请求体
        payload = {
            "model": self.MODEL_ID,
            "prompt": prompt,
            "image": self._image_to_base64_url(reference_image),
            "size": size,
            "response_format": "b64_json",
            "watermark": self.watermark,
            "sequential_image_generation": "disabled",  # 生成单图
        }
        
        print(f"[SeedreamBackend] Creating image generation task...")
        print(f"[SeedreamBackend] Model: {self.MODEL_ID}")
        print(f"[SeedreamBackend] Size: {size}")
        print(f"[SeedreamBackend] Prompt: {prompt[:100]}...")
        
        # 添加重试机制
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url, 
                    headers=self._get_headers(), 
                    json=payload, 
                    timeout=timeout
                )
                break  # 成功则跳出循环
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 递增等待时间
                    print(f"[SeedreamBackend] Connection error, retry {attempt + 1}/{max_retries} in {wait_time}s...")
                    import time
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed after {max_retries} retries: {str(e)}")
        
        if response.status_code != 200:
            error_detail = response.text
            raise RuntimeError(f"Failed to generate image: HTTP {response.status_code} - {error_detail}")
        
        result = response.json()
        
        # 检查错误
        if "error" in result:
            raise RuntimeError(f"API Error: {result['error']}")
        
        # 获取图片数据
        data = result.get("data", [])
        if not data:
            raise RuntimeError(f"No image data in response: {result}")
        
        image_info = data[0]
        
        # 解码 base64 图片
        if "b64_json" in image_info:
            image_data = base64.b64decode(image_info["b64_json"])
            image = Image.open(BytesIO(image_data))
        elif "url" in image_info:
            # 如果返回的是 URL，下载图片
            img_response = requests.get(image_info["url"], timeout=60)
            image_data = img_response.content
            image = Image.open(BytesIO(image_data))
        else:
            raise RuntimeError(f"No image data in response: {image_info}")
        
        size_str = image_info.get("size", f"{image.width}x{image.height}")
        
        print(f"[SeedreamBackend] Image generated: {size_str}")
        
        return ImageResult(
            image=image,
            image_data=image_data,
            size=size_str,
            raw_response=result
        )
    
    def generate_text_to_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        timeout: int = 120
    ) -> ImageResult:
        """
        文生图
        
        Args:
            prompt: 提示词
            size: 输出尺寸
            timeout: 超时时间（秒）
        
        Returns:
            ImageResult: 图片生成结果
        """
        if not self.is_available():
            raise RuntimeError("Seedream API key not provided")
        
        url = f"{self.API_BASE}/images/generations"
        
        # 构建请求体
        payload = {
            "model": self.MODEL_ID,
            "prompt": prompt,
            "size": size,
            "response_format": "b64_json",
            "watermark": self.watermark,
            "sequential_image_generation": "disabled",
        }
        
        print(f"[SeedreamBackend] Text-to-Image generation...")
        print(f"[SeedreamBackend] Model: {self.MODEL_ID}")
        print(f"[SeedreamBackend] Size: {size}")
        
        response = requests.post(
            url, 
            headers=self._get_headers(), 
            json=payload, 
            timeout=timeout
        )
        
        if response.status_code != 200:
            error_detail = response.text
            raise RuntimeError(f"Failed to generate image: HTTP {response.status_code} - {error_detail}")
        
        result = response.json()
        
        if "error" in result:
            raise RuntimeError(f"API Error: {result['error']}")
        
        data = result.get("data", [])
        if not data:
            raise RuntimeError(f"No image data in response: {result}")
        
        image_info = data[0]
        
        if "b64_json" in image_info:
            image_data = base64.b64decode(image_info["b64_json"])
            image = Image.open(BytesIO(image_data))
        elif "url" in image_info:
            img_response = requests.get(image_info["url"], timeout=60)
            image_data = img_response.content
            image = Image.open(BytesIO(image_data))
        else:
            raise RuntimeError(f"No image data in response: {image_info}")
        
        size_str = image_info.get("size", f"{image.width}x{image.height}")
        
        print(f"[SeedreamBackend] Image generated: {size_str}")
        
        return ImageResult(
            image=image,
            image_data=image_data,
            size=size_str,
            raw_response=result
        )
