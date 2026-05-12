"""
VLM碰撞体生成演示
使用豆包视觉模型分析等轴视角图片，自动生成碰撞体配置
"""

import os
import json
import requests
from pathlib import Path


class DoubaoVLMCollisionGenerator:
    """豆包VLM碰撞体生成器"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ARK_API_KEY')
        if not self.api_key:
            raise ValueError("请设置 ARK_API_KEY 环境变量或传入api_key参数")
        
        self.api_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        self.model = "doubao-seed-1-6-vision-250815"
    
    def create_grid_image(self, image_path: str, grid_size: int = 6) -> str:
        """
        创建带网格标注的图片
        
        Args:
            image_path: 原始图片路径
            grid_size: 网格大小（6表示6x6=36格）
        
        Returns:
            base64编码的网格图片
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            print("⚠️ 需要安装Pillow: pip install Pillow")
            return None
        
        from base64 import b64encode
        from io import BytesIO
        
        # 加载图片
        img = Image.open(image_path).convert('RGBA')
        width, height = img.size
        
        # 创建绘图层
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # 计算网格大小
        cell_width = width / grid_size
        cell_height = height / grid_size
        
        # 绘制网格线和编号
        for i in range(grid_size + 1):
            # 垂直线
            x = int(i * cell_width)
            draw.line([(x, 0), (x, height)], fill=(255, 0, 0, 200), width=2)
            # 水平线
            y = int(i * cell_height)
            draw.line([(0, y), (width, y)], fill=(255, 0, 0, 200), width=2)
        
        # 标注每个格子的编号
        cell_num = 0
        for row in range(grid_size):
            for col in range(grid_size):
                x = int((col + 0.5) * cell_width)
                y = int((row + 0.5) * cell_height)
                draw.text((x-10, y-10), str(cell_num), fill=(255, 255, 0, 255), font=None)
                cell_num += 1
        
        # 合并图层
        result = Image.alpha_composite(img, overlay)
        
        # 转为base64
        buffer = BytesIO()
        result.save(buffer, format='PNG')
        image_data = b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{image_data}"
    
    def generate_collision_config(self, image_path: str, item_type: str = "plant") -> dict:
        """
        分析图片并生成网格碰撞体配置
        
        Args:
            image_path: 本地图片路径
            item_type: 物品类型 ("plant" 或 "building")
        
        Returns:
            碰撞体配置字典
        """
        
        # 生成6x6网格图片
        print("📐 生成网格标注图片...")
        image_url = self.create_grid_image(image_path, grid_size=6)
        if not image_url:
            raise Exception("网格图片生成失败")
        prompt = self._build_grid_prompt(item_type)
        
        # 构造API请求
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "temperature": 0.3,
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 发送请求
        print(f"🔍 正在分析图片: {image_url}")
        response = requests.post(self.api_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"API请求失败: {response.status_code} - {response.text}")
        
        result = response.json()
        
        # 提取VLM返回的文本
        vlm_response = result['choices'][0]['message']['content']
        print(f"📝 VLM原始响应:\n{vlm_response}\n")
        
        # 解析JSON配置
        try:
            # 尝试从响应中提取JSON（VLM可能返回markdown格式）
            collision_config = self._extract_json_from_response(vlm_response)
            print("✅ 成功解析碰撞体配置")
            return collision_config
        except Exception as e:
            print(f"⚠️ JSON解析失败: {e}")
            print(f"原始响应: {vlm_response}")
            return {"error": str(e), "raw_response": vlm_response}
    
    def _build_grid_prompt(self, item_type: str) -> str:
        """构造网格模式的VLM提示词"""
        
        if item_type == "plant":
            return """你是游戏碰撞检测专家。图片已划分为6x6网格（编号0-35）。

**核心规则**: 
- **只选择树干本身**，不要选择周围的地面、草地、阴影
- **树冠叶子不需要碰撞**
- **地面/泥土不需要碰撞**
- 碰撞体应该是植物主体的实心部分

**网格布局** (6行×6列):
```
行0: [0  1  2  3  4  5]   ← 树冠区域(跳过)
行1: [6  7  8  9  10 11]  ← 树冠区域(跳过)
行2: [12 13 14 15 16 17]  ← 可能有树干上部
行3: [18 19 20 21 22 23]  ← 树干中部(重点)
行4: [24 25 26 27 28 29]  ← 树干下部(重点)
行5: [30 31 32 33 34 35]  ← 地面接触(选1-2格即可)
```

**最佳选择**: 只选2-4个格子，覆盖树干的中下部实心区域

**输出JSON格式**:
```json
{
  "selected_grids": [26, 27],
  "grid_info": {
    "total": 36,
    "rows": 6,
    "cols": 6
  },
  "reason": "只选择行4的中心格子26、27，这是树干最粗最实的部分"
}
```

仅返回JSON，无需额外说明。"""
        
        elif item_type == "building":
            return """你是游戏碰撞检测专家。图片已划分为6x6网格（编号0-35）。

**任务**: 选择需要设置碰撞体的网格编号

**规则**:
1. **墙体/实心** → 需要碰撞
2. **门/窗/空洞** → 不需要碰撞
3. **建筑底部** → 必须包含
4. 选择最少的格子覆盖主要结构

**输出JSON格式**:
```json
{
  "selected_grids": [14, 15, 20, 21, 26, 27],
  "grid_info": {
    "total": 36,
    "rows": 6,
    "cols": 6
  },
  "reason": "墙体和底部区域"
}
```

仅返回JSON，无需额外说明。"""
        
        return "分析图片并返回碰撞网格JSON"
    
    def _extract_json_from_response(self, response_text: str) -> dict:
        """从VLM响应中提取JSON"""
        # 尝试直接解析
        try:
            return json.loads(response_text)
        except:
            pass
        
        # 尝试提取markdown代码块中的JSON
        import re
        json_match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        # 尝试提取任何JSON对象
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        
        raise ValueError("未能从响应中提取有效JSON")
    
    def save_config(self, config: dict, output_path: str):
        """保存配置到文件"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"💾 配置已保存到: {output_path}")
    
    def visualize_collision(self, image_path: str, config: dict, output_path: str):
        """
        在图片上可视化网格碰撞体配置
        
        Args:
            image_path: 原始图片路径
            config: 碰撞体配置
            output_path: 输出图片路径
        """
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            print("⚠️ 需要安装Pillow: pip install Pillow")
            return
        
        # 加载图片
        img = Image.open(image_path).convert('RGBA')
        width, height = img.size
        
        # 创建绘图层
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # 绘制选中的网格
        if 'selected_grids' not in config:
            print("⚠️ 配置中没有selected_grids字段")
            return
            
        grid_size = config.get('grid_info', {}).get('rows', 5)
        cell_width = width / grid_size
        cell_height = height / grid_size
        
        selected = config['selected_grids']
        for grid_num in selected:
            row = grid_num // grid_size
            col = grid_num % grid_size
            
            x1 = int(col * cell_width)
            y1 = int(row * cell_height)
            x2 = int((col + 1) * cell_width)
            y2 = int((row + 1) * cell_height)
            
            # 填充选中的格子
            draw.rectangle([x1, y1, x2, y2], 
                         fill=(255, 0, 0, 100), 
                         outline=(255, 0, 0, 255), 
                         width=3)
            
            # 标注编号
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            draw.text((cx-10, cy-10), str(grid_num), fill=(255, 255, 0, 255))
        
        # 添加图例
        legend_y = 20
        draw.rectangle([10, legend_y, 250, legend_y + 80], fill=(0, 0, 0, 180))
        draw.text((20, legend_y + 10), "碰撞网格可视化:", fill=(255, 255, 255, 255))
        draw.rectangle([20, legend_y + 35, 40, legend_y + 55], 
                     fill=(255, 0, 0, 100), outline=(255, 0, 0, 255), width=2)
        draw.text((50, legend_y + 38), "选中区域（碰撞）", fill=(255, 255, 255, 255))
        draw.text((20, legend_y + 60), f"总计: {len(selected)}/{grid_size*grid_size} 格", 
                 fill=(255, 255, 255, 255))
        
        # 合并图层
        result = Image.alpha_composite(img, overlay)
        
        # 保存结果
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        print(f"🎨 可视化结果已保存到: {output_path}")
        
        # 返回选中的网格信息
        print(f"✅ 选中网格: {config['selected_grids']}")
        print(f"📝 原因: {config.get('reason', 'N/A')}")


def demo_plant_analysis():
    """演示：分析植物图片"""
    print("=" * 60)
    print("🌱 植物碰撞体生成演示")
    print("=" * 60 + "\n")
    
    # 初始化生成器
    generator = DoubaoVLMCollisionGenerator()
    
    # 使用本地图片
    from base64 import b64encode
    image_path = r"D:\SnowGlobe\SnowWeave\gradio_outputs\plant_durian_tree_20251221_023711\final_frames\durian_tree-stage4-transition-frame24.png"
    
    print(f"📁 读取本地图片: {image_path}")
    with open(image_path, "rb") as f:
        image_data = b64encode(f.read()).decode()
        image_url = f"data:image/png;base64,{image_data}"
    print(f"✅ 图片已编码 (大小: {len(image_data)} bytes)")
    print()
    
    try:
        # 生成碰撞体配置（6x6网格）
        config = generator.generate_collision_config(
            image_path=image_path,
            item_type="plant"
        )
        
        # 打印结果
        print("\n" + "=" * 60)
        print("📋 生成的碰撞体配置:")
        print("=" * 60)
        print(json.dumps(config, indent=2, ensure_ascii=False))
        
        # 保存配置
        generator.save_config(
            config,
            output_path="collision_configs/oak_stage4_collision.json"
        )
        
        # 可视化碰撞体
        print("\n🎨 生成可视化图片...")
        generator.visualize_collision(
            image_path=image_path,
            config=config,
            output_path="collision_configs/oak_stage4_visualization.png"
        )
        
    except Exception as e:
        print(f"❌ 错误: {e}")


def demo_batch_processing():
    """演示：批量处理多个阶段的植物"""
    print("\n" + "=" * 60)
    print("🔄 批量处理演示")
    print("=" * 60 + "\n")
    
    generator = DoubaoVLMCollisionGenerator()
    
    # 模拟多个生长阶段
    stages = [
        {
            "name": "strawberry_stage1",
            "url": "https://example.com/strawberry_stage1.png"
        },
        {
            "name": "strawberry_stage2",
            "url": "https://example.com/strawberry_stage2.png"
        },
        {
            "name": "strawberry_stage3",
            "url": "https://example.com/strawberry_stage3.png"
        },
    ]
    
    results = {}
    
    for stage in stages:
        print(f"\n处理: {stage['name']}")
        try:
            config = generator.generate_collision_config(
                image_url=stage['url'],
                item_type="plant"
            )
            results[stage['name']] = config
            
            # 保存单个配置
            generator.save_config(
                config,
                output_path=f"collision_configs/{stage['name']}_collision.json"
            )
        except Exception as e:
            print(f"⚠️ {stage['name']} 处理失败: {e}")
            results[stage['name']] = {"error": str(e)}
    
    # 保存汇总配置
    generator.save_config(
        results,
        output_path="collision_configs/strawberry_all_stages.json"
    )
    
    print("\n✅ 批量处理完成!")


def demo_building_analysis():
    """演示：分析建筑图片"""
    print("\n" + "=" * 60)
    print("🏠 建筑碰撞体生成演示")
    print("=" * 60 + "\n")
    
    generator = DoubaoVLMCollisionGenerator()
    
    # 示例：分析门的图片
    image_url = "https://example.com/treasure_room_door.png"
    
    try:
        config = generator.generate_collision_config(
            image_url=image_url,
            item_type="building"
        )
        
        print("\n📋 建筑碰撞体配置:")
        print(json.dumps(config, indent=2, ensure_ascii=False))
        
        generator.save_config(
            config,
            output_path="collision_configs/treasure_door_collision.json"
        )
        
    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║        VLM自动碰撞体生成系统 - 演示程序                   ║
║        Powered by 豆包视觉模型                            ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # 检查API Key
    if not os.getenv('ARK_API_KEY'):
        print("⚠️ 请先设置环境变量: ARK_API_KEY")
        print("   Windows: set ARK_API_KEY=your_api_key")
        print("   Linux/Mac: export ARK_API_KEY=your_api_key")
        exit(1)
    
    # 运行演示
    try:
        # 1. 单个植物分析
        demo_plant_analysis()
        
        # 2. 批量处理（取消注释以运行）
        # demo_batch_processing()
        
        # 3. 建筑分析（取消注释以运行）
        # demo_building_analysis()
        
    except KeyboardInterrupt:
        print("\n\n👋 已取消")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()
