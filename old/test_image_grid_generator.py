"""
图片网格生成器测试脚本
"""
import os
from SnowWeave.old.core.image_grid_generator import ImageGridGenerator


def test_generator():
    """测试图片生成器"""
    generator = ImageGridGenerator()
    
    # 获取输入
    print("=" * 50)
    print("图片网格生成器测试")
    print("=" * 50)
    
    image_path = input("\n请输入图片路径: ").strip()
    if not os.path.exists(image_path):
        print(f"❌ 文件不存在: {image_path}")
        return
    
    print("\n支持的类型:")
    print("  - plant: 将图片填充到所有4个网格")
    print("  - building: 将图片填充到第一个网格")
    print("  - 其他: 默认行为（第一个网格 + 原图缩放）")
    
    image_type = input("\n请输入图片类型 (plant/building/other): ").strip() or "other"
    
    print(f"\n处理中...")
    
    try:
        # 生成Base64数组
        grid_base64_array, original_base64_array = generator.generate(image_path, image_type)
        
        print("\n✅ 生成成功!")
        print(f"   网格Base64数组长度: {len(grid_base64_array)}")
        print(f"   原图Base64数组长度: {len(original_base64_array)}")
        
        if grid_base64_array:
            print(f"   网格Base64长度: {len(grid_base64_array[0])} 字符")
        if original_base64_array:
            print(f"   原图Base64长度: {len(original_base64_array[0])} 字符")
        
        # 可选：保存到文件
        save_files = input("\n是否保存到文件? (y/n): ").strip().lower()
        if save_files == 'y':
            output_dir = input("输出目录 (默认: output): ").strip() or "output"
            result = generator.save_result(image_path, image_type, output_dir)
            print(f"\n✅ 已保存到:")
            for file_path in result['files']:
                print(f"   - {file_path}")
        
        # 显示Base64前100个字符作为预览
        print("\n📋 Base64预览 (前100字符):")
        if grid_base64_array:
            print(f"   网格: {grid_base64_array[0][:100]}...")
        if original_base64_array:
            print(f"   原图: {original_base64_array[0][:100]}...")
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def test_all_types():
    """测试所有类型（需要提供测试图片）"""
    generator = ImageGridGenerator()
    
    test_image = input("请输入测试图片路径: ").strip()
    if not os.path.exists(test_image):
        print(f"文件不存在: {test_image}")
        return
    
    types = ['plant', 'building', 'other']
    
    for img_type in types:
        print(f"\n{'='*50}")
        print(f"测试类型: {img_type}")
        print('='*50)
        
        try:
            result = generator.save_result(test_image, img_type, f"test_output/{img_type}")
            print(f"✅ 成功生成 {img_type} 类型")
            print(f"   文件: {result['files']}")
        except Exception as e:
            print(f"❌ 失败: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        test_all_types()
    else:
        test_generator()
