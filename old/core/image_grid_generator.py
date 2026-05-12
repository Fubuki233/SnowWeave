"""
图片网格生成器
支持多种图片类型的处理和网格生成
"""
import os
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List


class ImageGridGenerator:
    """图片网格生成器类"""
    
    # 网格配置
    GRID_TEMPLATE_PATH = Path(__file__).parent / "grid_1024_512.png"
    GRID_SIZE = 1024
    CELL_SIZE = 512
    CELLS_PER_ROW = 2
    CELLS_PER_COL = 2
    
    def __init__(self):
        """初始化生成器"""
        if not self.GRID_TEMPLATE_PATH.exists():
            # 如果模板不存在，创建一个空白网格
            self._create_default_grid()
    
    def _create_default_grid(self):
        """创建默认的网格模板"""
        grid = Image.new('RGBA', (self.GRID_SIZE, self.GRID_SIZE), (255, 255, 255, 0))
        os.makedirs(self.GRID_TEMPLATE_PATH.parent, exist_ok=True)
        grid.save(self.GRID_TEMPLATE_PATH)
    
    def _load_and_resize_image(self, image_path: str, size: Tuple[int, int]) -> Image.Image:
        """
        加载并调整图片大小
        
        Args:
            image_path: 图片路径
            size: 目标尺寸 (width, height)
            
        Returns:
            调整后的图片
        """
        img = Image.open(image_path)
        
        # 转换为RGBA模式以支持透明度
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # 保持宽高比，调整到目标尺寸
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        # 创建目标尺寸的图片，居中放置
        result = Image.new('RGBA', size, (255, 255, 255, 0))
        offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
        result.paste(img, offset, img if img.mode == 'RGBA' else None)
        
        return result
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """
        将PIL图片转换为Base64字符串
        
        Args:
            image: PIL图片对象
            
        Returns:
            Base64编码的字符串
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def _create_plant_grid(self, image_path: str) -> Image.Image:
        """
        创建plant类型的网格图
        将输入图片按原尺寸放入1024x1024画布的4个格子中
        
        Args:
            image_path: 输入图片路径
            
        Returns:
            生成的网格图（1024x1024）
        """
        # 加载输入图片
        img = Image.open(image_path)
        
        # 转换为RGBA模式
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # 创建1024x1024白色画布
        grid = Image.new('RGBA', (self.GRID_SIZE, self.GRID_SIZE), (255, 255, 255, 255))
        
        # 填充到4个网格位置（每个格子512x512）
        positions = [
            (0, 0),                      # 左上
            (self.CELL_SIZE, 0),         # 右上
            (0, self.CELL_SIZE),         # 左下
            (self.CELL_SIZE, self.CELL_SIZE)  # 右下
        ]
        
        for x, y in positions:
            grid.paste(img, (x, y), img)
        
        return grid
    
    def _create_building_grid(self, image_path: str) -> Image.Image:
        """
        创建building类型的网格图
        将输入图片放缩到512x512并填充到第一个网格中
        
        Args:
            image_path: 输入图片路径
            
        Returns:
            生成的网格图
        """
        # 加载模板
        grid = Image.open(self.GRID_TEMPLATE_PATH).convert('RGBA')
        
        # 加载并调整输入图片
        resized_img = self._load_and_resize_image(image_path, (self.CELL_SIZE, self.CELL_SIZE))
        
        # 填充到第一个网格位置（左上）
        grid.paste(resized_img, (0, 0), resized_img)
        
        return grid
    
    def generate(self, image_path: str, image_type: str) -> Tuple[List[str], List[str]]:
        """
        生成图片网格
        
        Args:
            image_path: 输入图片路径
            image_type: 图片类型 ('plant', 'building', 等)
            
        Returns:
            元组 (grid_base64_array, original_base64_array)
            - plant类型: ([grid_base64], [])
            - 其他类型: ([grid_base64], [resized_original_base64])
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        image_type = image_type.lower()
        
        if image_type == 'plant':
            # Plant类型: 创建填充所有网格的图片
            grid_img = self._create_plant_grid(image_path)
            grid_base64 = self._image_to_base64(grid_img)
            return [grid_base64], []
        
        elif image_type == 'building':
            # Building类型: 只填充第一个网格
            grid_img = self._create_building_grid(image_path)
            grid_base64 = self._image_to_base64(grid_img)
            
            # 原图缩放到512x512
            resized_original = self._load_and_resize_image(image_path, (self.CELL_SIZE, self.CELL_SIZE))
            original_base64 = self._image_to_base64(resized_original)
            
            return [grid_base64], [original_base64]
        
        else:
            # 其他类型: 返回网格图和缩放后的原图
            # 默认行为：创建带第一个网格的图片
            grid_img = Image.open(self.GRID_TEMPLATE_PATH).convert('RGBA')
            resized_img = self._load_and_resize_image(image_path, (self.CELL_SIZE, self.CELL_SIZE))
            grid_img.paste(resized_img, (0, 0), resized_img)
            
            grid_base64 = self._image_to_base64(grid_img)
            original_base64 = self._image_to_base64(resized_img)
            
            return [grid_base64], [original_base64]
    
    def split_grid(self, grid_image: Image.Image) -> List[Image.Image]:
        """
        将网格图切割为4个独立的图片
        
        Args:
            grid_image: 1024x1024的网格图
            
        Returns:
            4个512x512的图片列表 [左上, 右上, 左下, 右下]
        """
        cells = []
        positions = [
            (0, 0),                                      # 左上 - Stage 1
            (self.CELL_SIZE, 0),                         # 右上 - Stage 2
            (0, self.CELL_SIZE),                         # 左下 - Stage 3
            (self.CELL_SIZE, self.CELL_SIZE)             # 右下 - Stage 4
        ]
        
        for x, y in positions:
            cell = grid_image.crop((x, y, x + self.CELL_SIZE, y + self.CELL_SIZE))
            cells.append(cell)
        
        return cells
    
    def split_grid_from_path(self, grid_path: str) -> List[Image.Image]:
        """
        从文件路径加载并切割网格图
        
        Args:
            grid_path: 网格图文件路径
            
        Returns:
            4个512x512的图片列表
        """
        grid_img = Image.open(grid_path).convert('RGBA')
        return self.split_grid(grid_img)
    
    def process_plant_stages(
        self, 
        image_path: str, 
        plant_id: str, 
        output_dir: str,
        frames_per_stage: int = 1
    ) -> dict:
        """
        处理植物图片：生成网格图并切割为4个阶段
        
        Args:
            image_path: 输入图片路径（1:1图片）
            plant_id: 植物ID
            output_dir: 输出目录
            frames_per_stage: 每个阶段的帧数（默认1）
            
        Returns:
            包含所有生成文件信息的字典
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成plant类型的网格图
        grid_img = self._create_plant_grid(image_path)
        
        # 保存网格图
        grid_path = os.path.join(output_dir, f"{plant_id}_grid.png")
        grid_img.save(grid_path)
        
        # 切割为4个阶段
        stage_cells = self.split_grid(grid_img)
        
        # 保存每个阶段的图片
        stage_files = []
        for stage_idx, cell_img in enumerate(stage_cells, start=1):
            # 为每个阶段创建目录
            stage_dir = os.path.join(output_dir, f"stage_{stage_idx}")
            idle_dir = os.path.join(stage_dir, "idle_frames")
            os.makedirs(idle_dir, exist_ok=True)
            
            # 保存帧（支持多帧，但通常是1帧）
            stage_frame_files = []
            for frame_idx in range(1, frames_per_stage + 1):
                frame_filename = f"{plant_id}-stage{stage_idx}-frame{frame_idx}.png"
                frame_path = os.path.join(idle_dir, frame_filename)
                cell_img.save(frame_path)
                stage_frame_files.append(frame_path)
            
            stage_files.append({
                'stage': stage_idx,
                'frames': stage_frame_files,
                'cell_image': cell_img
            })
        
        return {
            'grid_path': grid_path,
            'grid_image': grid_img,
            'stages': stage_files,
            'plant_id': plant_id,
            'total_stages': 4
        }
    
    def save_result(self, image_path: str, image_type: str, output_dir: str = "output") -> dict:
        """
        生成并保存结果到文件
        
        Args:
            image_path: 输入图片路径
            image_type: 图片类型
            output_dir: 输出目录
            
        Returns:
            包含文件路径和Base64数据的字典
        """
        os.makedirs(output_dir, exist_ok=True)
        
        grid_base64_array, original_base64_array = self.generate(image_path, image_type)
        
        result = {
            'grid_base64': grid_base64_array,
            'original_base64': original_base64_array,
            'files': []
        }
        
        # 保存网格图
        if grid_base64_array:
            grid_data = base64.b64decode(grid_base64_array[0])
            grid_path = os.path.join(output_dir, f"{image_type}_grid.png")
            with open(grid_path, 'wb') as f:
                f.write(grid_data)
            result['files'].append(grid_path)
        
        # 保存原图缩放版本
        if original_base64_array:
            original_data = base64.b64decode(original_base64_array[0])
            original_path = os.path.join(output_dir, f"{image_type}_original_512.png")
            with open(original_path, 'wb') as f:
                f.write(original_data)
            result['files'].append(original_path)
        
        return result
    
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


def main():
    """示例用法"""
    generator = ImageGridGenerator()
    
    # 示例1: Plant类型
    print("生成Plant类型网格...")
    try:
        grid_b64, orig_b64 = generator.generate("input_plant.png", "plant")
        print(f"Plant - 网格数组长度: {len(grid_b64)}, 原图数组长度: {len(orig_b64)}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 示例2: Building类型
    print("\n生成Building类型网格...")
    try:
        grid_b64, orig_b64 = generator.generate("input_building.png", "building")
        print(f"Building - 网格数组长度: {len(grid_b64)}, 原图数组长度: {len(orig_b64)}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 示例3: 其他类型
    print("\n生成Other类型网格...")
    try:
        grid_b64, orig_b64 = generator.generate("input_other.png", "other")
        print(f"Other - 网格数组长度: {len(grid_b64)}, 原图数组长度: {len(orig_b64)}")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
