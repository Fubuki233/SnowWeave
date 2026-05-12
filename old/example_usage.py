"""
图片网格生成器 - 简单API示例
展示如何在代码中使用ImageGridGenerator
"""
from core.image_grid_generator import ImageGridGenerator


def example_1_plant_type():
    """示例1: 生成Plant类型网格"""
    generator = ImageGridGenerator()
    
    # Plant类型：填充所有4个网格，第二个数组为空
    image_path = "your_plant_image.png"  # 替换为实际路径
    image_type = "plant"
    
    try:
        grid_base64_array, original_base64_array = generator.generate(image_path, image_type)
        
        # 返回结果
        # grid_base64_array: ['base64_string']  - 包含4个填充网格的图片
        # original_base64_array: []  - 空数组
        
        print(f"Plant类型生成成功")
        print(f"网格Base64数组: {len(grid_base64_array)} 个元素")
        print(f"原图Base64数组: {len(original_base64_array)} 个元素")
        
        return grid_base64_array, original_base64_array
        
    except FileNotFoundError:
        print(f"文件不存在，请替换为实际的图片路径")
        return None, None


def example_2_building_type():
    """示例2: 生成Building类型网格"""
    generator = ImageGridGenerator()
    
    # Building类型：只填充第一个网格，返回原图缩放版本
    image_path = "your_building_image.png"  # 替换为实际路径
    image_type = "building"
    
    try:
        grid_base64_array, original_base64_array = generator.generate(image_path, image_type)
        
        # 返回结果
        # grid_base64_array: ['base64_string']  - 只有第一个网格有内容
        # original_base64_array: ['base64_string']  - 原图缩放到512x512
        
        print(f"Building类型生成成功")
        print(f"网格Base64数组: {len(grid_base64_array)} 个元素")
        print(f"原图Base64数组: {len(original_base64_array)} 个元素")
        
        return grid_base64_array, original_base64_array
        
    except FileNotFoundError:
        print(f"文件不存在，请替换为实际的图片路径")
        return None, None


def example_3_custom_usage():
    """示例3: 自定义使用方式"""
    generator = ImageGridGenerator()
    
    # 配置
    config = {
        'plant': {
            'path': 'assets/tree.png',
            'type': 'plant'
        },
        'building': {
            'path': 'assets/house.png',
            'type': 'building'
        },
        'item': {
            'path': 'assets/item.png',
            'type': 'other'
        }
    }
    
    results = {}
    
    for name, cfg in config.items():
        try:
            grid_b64, orig_b64 = generator.generate(cfg['path'], cfg['type'])
            results[name] = {
                'grid_base64': grid_b64,
                'original_base64': orig_b64
            }
            print(f"✅ {name} 处理成功")
        except Exception as e:
            print(f"❌ {name} 处理失败: {e}")
            results[name] = None
    
    return results


def example_4_save_to_file():
    """示例4: 保存到文件"""
    generator = ImageGridGenerator()
    
    image_path = "input.png"
    image_type = "plant"
    output_dir = "generated_grids"
    
    try:
        result = generator.save_result(image_path, image_type, output_dir)
        
        print(f"已保存到文件:")
        print(f"  网格Base64: {result['grid_base64']}")
        print(f"  原图Base64: {result['original_base64']}")
        print(f"  文件路径: {result['files']}")
        
        return result
        
    except FileNotFoundError:
        print(f"文件不存在，请替换为实际的图片路径")
        return None


def quick_generate(image_path: str, image_type: str = "plant"):
    """
    快速生成接口
    
    Args:
        image_path: 图片路径
        image_type: 图片类型 ('plant', 'building', 'other')
        
    Returns:
        (grid_base64_array, original_base64_array)
    """
    generator = ImageGridGenerator()
    return generator.generate(image_path, image_type)


# 使用示例
if __name__ == "__main__":
    print("图片网格生成器 - API示例\n")
    
    print("=" * 60)
    print("快速使用示例:")
    print("=" * 60)
    
    # 快速使用
    print("\n使用 quick_generate 函数:")
    print("```python")
    print('grid, orig = quick_generate("my_image.png", "plant")')
    print("```")
    
    print("\n返回值说明:")
    print("  - Plant类型: ([grid_base64], [])")
    print("  - Building类型: ([grid_base64], [original_512_base64])")
    print("  - Other类型: ([grid_base64], [original_512_base64])")
    
    print("\n" + "=" * 60)
    print("详细使用方式，请查看上面的示例函数:")
    print("  - example_1_plant_type()")
    print("  - example_2_building_type()")
    print("  - example_3_custom_usage()")
    print("  - example_4_save_to_file()")
    print("=" * 60)
