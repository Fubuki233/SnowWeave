"""
测试植物网格工作流
"""
import os
import sys
from PIL import Image

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from SnowWeave.old.core.image_grid_generator import ImageGridGenerator
from SnowWeave.old.core.plant_generator import PlantGenerator


def test_grid_workflow():
    """测试网格工作流"""
    print("=" * 60)
    print("植物网格工作流测试")
    print("=" * 60)
    
    # 准备测试图片
    test_image = input("\n请输入测试图片路径 (1:1图片): ").strip()
    if not os.path.exists(test_image):
        print(f"❌ 文件不存在: {test_image}")
        return
    
    plant_id = input("请输入植物ID (例如: test_plant): ").strip() or "test_plant"
    
    print("\n开始处理...")
    
    # 使用PlantGenerator的网格工作流
    generator = PlantGenerator()
    
    print("\n生成进度:")
    print("-" * 60)
    
    for preview, frames, config, videos, collision, status in generator.generate_from_grid(
        image_path=test_image,
        plant_id=plant_id,
        target_width=512,
        frames_per_stage=1,
        ark_api_key=""
    ):
        print(f"状态: {status}")
        if frames:
            print(f"  已生成帧数: {len(frames)}")
        if config:
            print(f"  配置文件: {config}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成!")
    print("=" * 60)


def test_simple_grid():
    """简单测试ImageGridGenerator"""
    print("\n" + "=" * 60)
    print("ImageGridGenerator 简单测试")
    print("=" * 60)
    
    test_image = input("\n请输入测试图片路径: ").strip()
    if not os.path.exists(test_image):
        print(f"❌ 文件不存在: {test_image}")
        return
    
    generator = ImageGridGenerator()
    
    # 测试process_plant_stages
    result = generator.process_plant_stages(
        image_path=test_image,
        plant_id="test_plant",
        output_dir="test_grid_output",
        frames_per_stage=1
    )
    
    print(f"\n✅ 生成完成!")
    print(f"网格图: {result['grid_path']}")
    print(f"总阶段数: {result['total_stages']}")
    print(f"\n各阶段文件:")
    for stage_info in result['stages']:
        print(f"  Stage {stage_info['stage']}: {len(stage_info['frames'])} 帧")
        for frame_path in stage_info['frames']:
            print(f"    - {frame_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试植物网格工作流")
    parser.add_argument("--simple", action="store_true", help="运行简单测试")
    args = parser.parse_args()
    
    if args.simple:
        test_simple_grid()
    else:
        test_grid_workflow()
