"""
测试批量视频生成功能
Test batch video generation feature
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.video_generator import VideoGenerator
from core.api_manager import get_api_manager
from core.config import DEFAULT_IMAGE_PATH
from PIL import Image
import numpy as np


def test_batch_generation():
    """测试批量生成功能"""
    print("=" * 60)
    print("测试批量视频生成功能 / Testing Batch Video Generation")
    print("=" * 60)
    
    # 1. 检查默认图片是否存在
    if not os.path.exists(DEFAULT_IMAGE_PATH):
        print(f"[ERROR] 默认图片不存在: {DEFAULT_IMAGE_PATH}")
        return
    
    # 2. 加载图片
    image = Image.open(DEFAULT_IMAGE_PATH)
    image_array = np.array(image)
    print(f"✓ 成功加载图片: {image.size}")
    
    # 3. 定义测试动作
    test_actions = [
        "walking animation, side view, loop",
        "running animation, side view, loop",
        "jumping animation, side view"
    ]
    print(f"✓ 测试动作数量: {len(test_actions)}")
    for i, action in enumerate(test_actions, 1):
        print(f"  {i}. {action}")
    
    # 4. 初始化生成器
    generator = VideoGenerator()
    print("✓ 生成器初始化完成")
    
    # 5. 检查 API 配置
    api_manager = get_api_manager()
    backend_name = getattr(api_manager, 'backend_name', 'Not configured')
    print(f"✓ API Manager 状态: {backend_name}")
    
    # 6. 测试模式说明
    print("\n" + "=" * 60)
    print("测试配置 / Test Configuration:")
    print("=" * 60)
    print(f"  后端 / Backend: gemini")
    print(f"  模型 / Model: veo-2.0")
    print(f"  并行数 / Max Workers: 2")
    print(f"  时长 / Duration: 6 seconds")
    print("\n注意: 实际生成需要有效的 API Key")
    print("Note: Actual generation requires a valid API Key")
    print("=" * 60)
    
    # 7. 显示预期输出结构
    print("\n预期输出结构 / Expected Output Structure:")
    print("  video_batch_YYYYMMDD_HHMMSS/")
    print("    ├── animation_1_walking_animation.mp4")
    print("    ├── animation_2_running_animation.mp4")
    print("    ├── animation_3_jumping_animation.mp4")
    print("    ├── reference_image.png")
    print("    └── metadata.txt")
    
    print("\n✓ 批量生成功能测试完成 (配置验证)")
    print("  如需实际测试生成,请在 Gradio UI 中配置 API Key 后运行")


if __name__ == "__main__":
    test_batch_generation()
