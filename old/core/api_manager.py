"""
SnowWeave API Manager Module
API 管理模块 - 管理视频生成后端和 API 客户端
"""

from typing import Optional, Dict, Any
from google import genai

from lib.video_backends import GeminiVeoBackend, SeedanceBackend
from lib.video_backends.base import VideoBackend

from .config import (
    AVAILABLE_BACKENDS, 
    GEMINI_MODELS, 
    SEEDANCE_MODELS, 
    DEFAULT_MODEL,
    t
)


class APIManager:
    """
    API 管理器
    管理视频生成后端和 API 客户端的初始化、切换
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._gemini_client: Optional[genai.Client] = None
        self._video_backend: Optional[VideoBackend] = None
        self._current_api_key: str = ""
        self._current_backend_name: str = "gemini"
        self._current_model: str = DEFAULT_MODEL
        self._initialized = True
    
    @property
    def gemini_client(self) -> Optional[genai.Client]:
        """获取 Gemini 客户端"""
        return self._gemini_client
    
    @property
    def video_backend(self) -> Optional[VideoBackend]:
        """获取当前视频后端"""
        return self._video_backend
    
    @property
    def backend_name(self) -> str:
        """获取当前后端名称"""
        return self._current_backend_name
    
    @property
    def api_key(self) -> str:
        """获取当前 API 密钥"""
        return self._current_api_key
    
    @property
    def current_model(self) -> str:
        """获取当前模型"""
        return self._current_model
    
    @current_model.setter
    def current_model(self, model_name: str):
        """设置当前模型"""
        self._current_model = model_name
    
    def is_initialized(self) -> bool:
        """检查是否已初始化 API"""
        return self._video_backend is not None
    
    def initialize(self, api_key: str, backend: str = "gemini") -> str:
        """
        初始化 API 客户端
        
        Args:
            api_key: API 密钥
            backend: 后端名称 ('gemini' 或 'seedance')
        
        Returns:
            状态消息
        """
        try:
            print(f"[APIManager] Initializing backend: {backend}")
            
            if backend == "gemini":
                self._gemini_client = genai.Client(api_key=api_key)
                self._video_backend = GeminiVeoBackend(api_key)
            elif backend == "seedance":
                self._video_backend = SeedanceBackend(api_key)
                self._gemini_client = None  # Seedance 不需要 Gemini client
            else:
                return f"[ERROR] Unknown backend: {backend}"
            
            self._current_api_key = api_key
            self._current_backend_name = backend
            
            print(f"[APIManager] Success! backend={self._current_backend_name}")
            backend_display = AVAILABLE_BACKENDS.get(backend, backend)
            return f"[OK] {t('api_success')} (Backend: {backend_display})"
            
        except Exception as e:
            print(f"[APIManager] Error: {e}")
            return f"[ERROR] {t('api_failed')}: {str(e)}"
    
    def set_model(self, model_name: str) -> str:
        """
        设置当前使用的模型
        
        Args:
            model_name: 模型名称
        
        Returns:
            状态消息
        """
        self._current_model = model_name
        
        # 获取模型显示名称
        all_models = {**GEMINI_MODELS, **SEEDANCE_MODELS}
        display_name = all_models.get(model_name, model_name)
        return f"[OK] {t('model_switched')}: {display_name}"
    
    def get_models_for_backend(self, backend: str = None) -> list:
        """
        根据后端获取可用模型列表
        
        Args:
            backend: 后端名称，默认使用当前后端
        
        Returns:
            模型名称列表
        """
        if backend is None:
            backend = self._current_backend_name
        
        if backend == "seedance":
            return list(SEEDANCE_MODELS.keys())
        return list(GEMINI_MODELS.keys())
    
    def validate_backend(self, requested_backend: str) -> Optional[str]:
        """
        验证请求的后端是否与当前配置匹配
        
        Args:
            requested_backend: 请求使用的后端
        
        Returns:
            错误消息，如果验证通过则返回 None
        """
        if self._video_backend is None:
            return f"[ERROR] {t('api_required')} (请先在设置中配置任意后端)"
        
        if requested_backend != self._current_backend_name:
            return f"[ERROR] 当前配置的后端是 {self._current_backend_name}，但选择了 {requested_backend}。请在设置中切换后端或选择 {self._current_backend_name}"
        
        return None


# 全局 API 管理器实例
api_manager = APIManager()


def get_api_manager() -> APIManager:
    """获取全局 API 管理器实例"""
    return api_manager
