# 图片网格生成器 (Image Grid Generator)

一个灵活的图片网格生成工具，支持将1:1的输入图片按照不同规则处理成网格图，并返回Base64编码。

## 功能特性

- ✅ 支持多种图片类型处理规则
- ✅ 返回Base64编码的二维数组
- ✅ 可扩展的类型系统
- ✅ 保持图片透明度
- ✅ 自动调整图片大小并居中

## 安装依赖

```bash
pip install Pillow
```

## 快速开始

### 基础使用

```python
from core.image_grid_generator import ImageGridGenerator

# 创建生成器实例
generator = ImageGridGenerator()

# 生成网格
grid_base64_array, original_base64_array = generator.generate(
    image_path="my_plant.png",
    image_type="plant"
)
```

### 快速API

```python
from example_usage import quick_generate

# 一行代码生成
grid, orig = quick_generate("my_image.png", "plant")
```

## 图片类型说明

### 1. Plant 类型

**用途**: 植物、可平铺资源

**规则**: 
- 将输入图片缩放到 512x512
- 填充到1024x1024画布的4个网格中（2x2布局）
- 使用 `core/grid_1024_512.png` 作为模板

**返回值**:
```python
(
    [grid_base64],  # 包含4个填充网格的图片
    []              # 空数组
)
```

**示例**:
```python
grid_b64, orig_b64 = generator.generate("tree.png", "plant")
# grid_b64: ['iVBORw0KGgo...']  (1024x1024的网格图)
# orig_b64: []
```

### 2. Building 类型

**用途**: 建筑物、单体对象

**规则**:
- 将输入图片缩放到 512x512
- 只填充到第一个网格（左上角）
- 返回原图的512x512缩放版本

**返回值**:
```python
(
    [grid_base64],      # 只有第一个网格有内容
    [original_base64]   # 原图缩放到512x512
)
```

**示例**:
```python
grid_b64, orig_b64 = generator.generate("house.png", "building")
# grid_b64: ['iVBORw0KGgo...']  (网格图，只有左上角有内容)
# orig_b64: ['iVBORw0KGgo...']  (512x512的原图)
```

### 3. Other 类型（默认）

**用途**: 其他类型对象

**规则**:
- 默认行为：第一个网格 + 原图缩放
- 与Building类型类似

**返回值**:
```python
(
    [grid_base64],      # 第一个网格有内容
    [original_base64]   # 原图缩放到512x512
)
```

## 完整示例

### 示例1: 处理单张图片

```python
from core.image_grid_generator import ImageGridGenerator

generator = ImageGridGenerator()

# 处理plant类型
grid, orig = generator.generate("assets/oak_tree.png", "plant")

print(f"网格数组长度: {len(grid)}")  # 1
print(f"原图数组长度: {len(orig)}")  # 0
print(f"Base64长度: {len(grid[0])}")  # 例如: 125000+
```

### 示例2: 批量处理

```python
from core.image_grid_generator import ImageGridGenerator

generator = ImageGridGenerator()

images = [
    ("tree1.png", "plant"),
    ("tree2.png", "plant"),
    ("house.png", "building"),
]

results = {}
for path, img_type in images:
    grid, orig = generator.generate(path, img_type)
    results[path] = {
        'grid': grid,
        'original': orig
    }
```

### 示例3: 保存到文件

```python
from core.image_grid_generator import ImageGridGenerator

generator = ImageGridGenerator()

result = generator.save_result(
    image_path="input.png",
    image_type="plant",
    output_dir="output"
)

print("生成的文件:", result['files'])
# ['output/plant_grid.png']
```

### 示例4: 在Web应用中使用

```python
from core.image_grid_generator import ImageGridGenerator
import json

def process_image_api(image_path: str, image_type: str):
    """API端点示例"""
    generator = ImageGridGenerator()
    
    try:
        grid_b64, orig_b64 = generator.generate(image_path, image_type)
        
        return {
            'success': True,
            'data': {
                'grid_base64': grid_b64,
                'original_base64': orig_b64
            }
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# 使用
response = process_image_api("upload/tree.png", "plant")
print(json.dumps(response, indent=2))
```

## 网格配置

当前配置（可在 `ImageGridGenerator` 类中修改）:

```python
GRID_SIZE = 1024        # 网格总大小
CELL_SIZE = 512         # 单个格子大小
CELLS_PER_ROW = 2       # 每行格子数
CELLS_PER_COL = 2       # 每列格子数
```

网格布局:
```
+-------+-------+
|  0,0  |  1,0  |  (512x512 each)
+-------+-------+
|  0,1  |  1,1  |
+-------+-------+
```

## 扩展新类型

要添加新的图片类型处理规则，在 `ImageGridGenerator` 类中添加新方法:

```python
def _create_custom_grid(self, image_path: str) -> Image.Image:
    """自定义类型处理"""
    grid = Image.open(self.GRID_TEMPLATE_PATH).convert('RGBA')
    resized_img = self._load_and_resize_image(image_path, (self.CELL_SIZE, self.CELL_SIZE))
    
    # 自定义逻辑：例如填充特定位置
    grid.paste(resized_img, (self.CELL_SIZE, self.CELL_SIZE), resized_img)  # 右下角
    
    return grid
```

然后在 `generate()` 方法中添加对应的分支:

```python
elif image_type == 'custom':
    grid_img = self._create_custom_grid(image_path)
    # ... 返回逻辑
```

## 测试

### 交互式测试

```bash
cd D:\SnowGlobe\SnowWeave
python test_image_grid_generator.py
```

按提示输入图片路径和类型。

### 批量测试所有类型

```bash
python test_image_grid_generator.py --all
```

## API参考

### ImageGridGenerator

主类，处理图片网格生成。

#### 方法

##### `generate(image_path: str, image_type: str) -> Tuple[List[str], List[str]]`

生成图片网格并返回Base64数组。

**参数**:
- `image_path` (str): 输入图片路径
- `image_type` (str): 图片类型 ('plant', 'building', 'other')

**返回**:
- `Tuple[List[str], List[str]]`: (grid_base64_array, original_base64_array)

**异常**:
- `FileNotFoundError`: 图片文件不存在

##### `save_result(image_path: str, image_type: str, output_dir: str) -> dict`

生成并保存结果到文件。

**参数**:
- `image_path` (str): 输入图片路径
- `image_type` (str): 图片类型
- `output_dir` (str): 输出目录（默认: "output"）

**返回**:
```python
{
    'grid_base64': List[str],
    'original_base64': List[str],
    'files': List[str]  # 保存的文件路径列表
}
```

## 常见问题

### Q: 如何处理非1:1的图片？

A: 生成器会自动调整图片大小并保持宽高比，然后居中放置在目标尺寸内。

### Q: 支持哪些图片格式？

A: 支持PIL/Pillow支持的所有格式（PNG, JPG, BMP, GIF等）。输出始终为PNG格式以保持透明度。

### Q: Base64数据如何使用？

A: 可以直接在HTML中使用：
```html
<img src="data:image/png;base64,{base64_string}" />
```

或在其他应用中解码：
```python
import base64
img_data = base64.b64decode(base64_string)
```

### Q: 如何修改网格大小？

A: 修改类常量：
```python
generator = ImageGridGenerator()
generator.GRID_SIZE = 2048
generator.CELL_SIZE = 1024
```

## 许可证

MIT License

## 更新日志

### v1.0.0 (2025-12-21)
- ✅ 初始版本
- ✅ 支持 plant, building, other 类型
- ✅ Base64编码输出
- ✅ 文件保存功能
