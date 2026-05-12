# Video Generation Backends
# 视频生成后端

from .base import VideoBackend, VideoResult
from .gemini_veo import GeminiVeoBackend
from .seedance import SeedanceBackend

__all__ = [
    'VideoBackend',
    'VideoResult', 
    'GeminiVeoBackend',
    'SeedanceBackend'
]
