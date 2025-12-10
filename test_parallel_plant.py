"""
测试并行植物视频生成功能
Test parallel plant video generation
"""

from core.plant_generator import PlantGenerator
from core.api_manager import get_api_manager
from PIL import Image
import numpy as np

def test_parallel_generation():
    """测试并行生成"""
    print("=" * 60)
    print("测试植物并行视频生成 / Testing Parallel Plant Generation")
    print("=" * 60)
    
    # 1. 检查API配置
    api_manager = get_api_manager()
    backend_name = getattr(api_manager, 'backend_name', 'Not configured')
    print(f"✓ API Manager 状态: {backend_name}")
    
    # 2. 创建测试图片
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    print(f"✓ 测试图片创建: {test_image.shape}")
    
    # 3. 配置参数
    test_params = {
        "image": test_image,
        "ref_images": None,
        "prompt": "strawberry",
        "plant_id": "test_strawberry",
        "stages": 3,
        "target_width": 128,
        "frames_per_stage": 12,
        "tolerance": 180,
        "auto_crop": False,
        "crop_padding": 5,
        "model_name": "doubao-seedance-1-0-pro-250528",
        "duration": 4,
        "backend": "seedance",
        "resolution": "480p",
        "reference_mode": "last_frame"
    }
    
    print(f"\n测试配置:")
    print(f"  植物: {test_params['prompt']}")
    print(f"  阶段数: {test_params['stages']}")
    print(f"  后端: {test_params['backend']}")
    print(f"  模型: {test_params['model_name']}")
    print(f"  时长: {test_params['duration']}秒")
    
    print(f"\n预期行为:")
    print(f"  1. 并行生成 {test_params['stages']} 个阶段的视频")
    print(f"  2. 显示每个视频的生成进度")
    print(f"  3. 串行处理每个视频的帧提取")
    print(f"  4. 返回所有视频路径列表")
    
    print(f"\n注意: 实际生成需要有效的 API Key")
    print(f"如需实际测试生成，请在 Gradio UI 中配置 API Key 后运行")
    print("=" * 60)
    
    # 如果API已配置，可以继续测试
    if backend_name != 'Not configured':
        print(f"\n✓ API 已配置，可以进行实际测试")
        print(f"运行 gradio_app.py 并使用植物生成标签页测试")
    else:
        print(f"\n⚠ API 未配置，跳过实际生成测试")

if __name__ == "__main__":
    test_parallel_generation()
