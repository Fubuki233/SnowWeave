"""
Video Backend Base Classes
视频生成后端基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any
from PIL import Image
import os


@dataclass
class VideoResult:
    """视频生成结果"""
    video_path: str  # 本地视频文件路径
    video_data: Optional[bytes] = None  # 视频字节数据
    duration: float = 0.0  # 视频时长（秒）
    width: int = 0
    height: int = 0
    backend: str = ""  # 使用的后端名称
    raw_response: Any = None  # 原始API响应


class VideoBackend(ABC):
    """视频生成后端抽象基类"""
    
    # 后端名称
    name: str = "base"
    
    # 支持的模型列表
    available_models: dict = {}
    
    def __init__(self, api_key: str, **kwargs):
        """
        初始化后端
        
        Args:
            api_key: API密钥
            **kwargs: 其他配置参数
        """
        self.api_key = api_key
        self.config = kwargs
    
    @abstractmethod
    def generate_video(
        self,
        reference_image: Image.Image,
        prompt: str,
        model_name: Optional[str] = None,
        duration: int = 5,
        **kwargs
    ) -> VideoResult:
        """
        生成视频
        
        Args:
            reference_image: 参考图片 (PIL Image)
            prompt: 提示词
            model_name: 模型名称
            duration: 视频时长（秒）
            **kwargs: 其他参数
        
        Returns:
            VideoResult: 视频生成结果
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查后端是否可用"""
        pass
    
    def get_models(self) -> dict:
        """获取支持的模型列表"""
        return self.available_models
    
    def get_default_model(self) -> str:
        """获取默认模型"""
        if self.available_models:
            return list(self.available_models.keys())[0]
        return ""


def get_backend(backend_name: str, api_key: str, **kwargs) -> VideoBackend:
    """
    获取视频生成后端实例
    
    Args:
        backend_name: 后端名称 ("seedance", "doubao")
        api_key: API密钥
        **kwargs: 其他配置参数
    
    Returns:
        VideoBackend: 后端实例
    """
    from .seedance import SeedanceBackend
    
    backends = {
        "seedance": SeedanceBackend,
        "doubao": SeedanceBackend,
    }
    
    backend_class = backends.get(backend_name.lower())
    if backend_class is None:
        raise ValueError(f"Unknown backend: {backend_name}. Available: {list(backends.keys())}")
    
    return backend_class(api_key, **kwargs)
