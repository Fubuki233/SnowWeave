"""
SnowWeave MCP (Model Control Pipeline) Module
角色动画生成流水线模块

包含：
- actor_animation.py: 完整的角色动画生成流水线
- seedream_backend.py: Seedream 4.5 图片生成后端
- seedance_backend.py: Seedance 1.5 视频生成后端
- frame_extractor.py: 视频帧提取工具

使用方法：
```python
from mcp.actor_animation import run_pipeline

results = run_pipeline(
    api_key="your-ark-api-key",
    character_image_path="path/to/character.png",
    character_name="my_character",
)
```

或者使用命令行：
```bash
python -m mcp.actor_animation character.png --api-key YOUR_KEY --name my_character
```
"""

from .actor_animation import (
    ActorAnimationPipeline,
    PipelineConfig,
    run_pipeline,
)
from .seedream_backend import SeedreamBackend, ImageResult
from .seedance_backend import SeedanceBackend, VideoResult
from .frame_extractor import (
    extract_frames_from_video,
    save_frames,
    create_sprite_sheet,
)

__all__ = [
    # Main pipeline
    "ActorAnimationPipeline",
    "PipelineConfig",
    "run_pipeline",
    # Backends
    "SeedreamBackend",
    "SeedanceBackend",
    "ImageResult",
    "VideoResult",
    # Frame tools
    "extract_frames_from_video",
    "save_frames",
    "create_sprite_sheet",
]
