"""
简化的绿幕背景剔除工具
自动检测绿色背景并移除

使用方法:
    python remove_background.py <图片路径> [颜色容差]
    
示例:
    python remove_background.py extracted_frames/frame_000.png
    python remove_background.py extracted_frames/frame_000.png 30
"""

import os
import sys
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count

def detect_background_color(image):
    """
    自动检测背景颜色（取四角平均值，带异常值过滤）
    
    参数:
        image: PIL Image对象
    
    返回:
        背景色 (R, G, B)
    """
    img_array = np.array(image.convert('RGB'))
    h, w = img_array.shape[:2]
    
    # 采样四角区域（10x10像素）
    corner_size = 10
    corners = [
        img_array[0:corner_size, 0:corner_size],                    # 左上
        img_array[0:corner_size, w-corner_size:w],                  # 右上
        img_array[h-corner_size:h, 0:corner_size],                  # 左下
        img_array[h-corner_size:h, w-corner_size:w],                # 右下
    ]
    
    # 计算每个角的平均颜色
    corner_colors = [np.mean(corner.reshape(-1, 3), axis=0) for corner in corners]
    
    # 计算所有角颜色的中位数（更鲁棒，避免异常值影响）
    all_pixels = np.concatenate([corner.reshape(-1, 3) for corner in corners])
    bg_color = tuple(np.median(all_pixels, axis=0).astype(int))
    
    print(f"检测到背景色: RGB{bg_color}")
    return bg_color


def is_white_background(bg_color, threshold=230):
    """
    检测是否为白色/浅色背景
    
    参数:
        bg_color: 背景色 (R, G, B)
        threshold: 白色判定阈值
    
    返回:
        bool
    """
    return all(c >= threshold for c in bg_color)

def color_distance(color1, color2):
    """计算两个颜色的欧氏距离"""
    return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))

def detect_and_remove_black_borders(image, black_threshold=30):
    """
    检测并移除图片左右的黑边 (保持上下完整)
    
    参数:
        image: PIL Image对象
        black_threshold: 判定为黑色的阈值 (0-255)
    
    返回:
        裁剪后的PIL Image对象
    """
    img_array = np.array(image.convert('RGB'))
    h, w = img_array.shape[:2]
    
    # 检查是否是1280x720尺寸 (允许小幅度误差)
    if not (1270 <= w <= 1290 and 710 <= h <= 770):
        print(f"图片尺寸 {w}x{h} 不在目标范围内，跳过黑边检测")
        return image
    
    print(f"检测到目标尺寸 {w}x{h}，开始检测左右黑边...")
    
    # 只检测左右两侧
    # 计算每列的平均亮度
    col_brightness = np.mean(img_array, axis=(0, 2))
    
    # 找到非黑色区域的左右边界
    non_black_cols = np.where(col_brightness > black_threshold)[0]
    
    if len(non_black_cols) == 0:
        print("警告: 整张图片都是黑色，保持原图")
        return image
    
    # 计算裁剪边界 (只裁剪左右,保持上下完整)
    left = non_black_cols[0]
    right = non_black_cols[-1] + 1
    
    # 如果左右都没有黑边,保持原图
    if left == 0 and right == w:
        print("未检测到左右黑边，保持原图")
        return image
    
    # 裁剪图片 (只裁剪左右)
    cropped = image.crop((left, 0, right, h))
    
    old_size = image.size
    new_size = cropped.size
    
    print(f"✓ 黑边裁剪完成: {old_size} → {new_size}")
    print(f"  - 移除边距: 左{left}px, 右{w-right}px (保持上下完整)")
    
    return cropped

def remove_background(image, bg_color, tolerance=30):
    """
    智能背景移除：针对白色背景优化
    
    参数:
        image: PIL Image对象
        bg_color: 背景色 (R, G, B)
        tolerance: 颜色容差
    
    返回:
        带透明通道的PIL Image对象
    """
    img_array = np.array(image.convert('RGB')).astype(np.float32)
    h, w = img_array.shape[:2]
    
    # 创建alpha通道
    alpha = np.ones((h, w), dtype=np.uint8) * 255
    
    # 检测是否为白色背景
    white_bg = is_white_background(bg_color)
    
    if white_bg:
        print(f"检测到白色背景，使用优化算法 (容差: {tolerance})...")
        
        # 白色背景优化策略:
        # 1. 计算像素亮度 (V 值)
        # 2. 计算像素饱和度 (S 值) - 白色背景饱和度低
        # 3. 综合判断
        
        # 计算亮度 (取最大RGB值作为V)
        brightness = np.max(img_array, axis=2)
        
        # 计算饱和度: S = (max - min) / max
        max_rgb = np.max(img_array, axis=2)
        min_rgb = np.min(img_array, axis=2)
        # 避免除零
        saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
        
        # 白色背景特征: 高亮度 + 低饱和度
        # 容差越大，允许的饱和度越高
        brightness_threshold = 255 - tolerance  # 容差30 -> 亮度阈值225
        saturation_threshold = tolerance / 255.0 * 1.5  # 容差30 -> 饱和度阈值约0.18
        
        # 背景掩码: 高亮度且低饱和度
        mask = (brightness >= brightness_threshold) & (saturation <= saturation_threshold)
        
        # 边缘优化: 使用颜色距离作为补充判断
        bg_array = np.array(bg_color, dtype=np.float32)
        distances = np.sqrt(np.sum((img_array - bg_array) ** 2, axis=2))
        color_mask = distances <= tolerance * 1.5
        
        # 合并两种掩码 (取并集，确保白色区域都被移除)
        mask = mask | color_mask
        
    else:
        print(f"开始移除背景 (容差: {tolerance})...")
        # 非白色背景: 使用传统颜色距离方法
        bg_array = np.array(bg_color, dtype=np.float32)
        distances = np.sqrt(np.sum((img_array - bg_array) ** 2, axis=2))
        mask = distances <= tolerance
    
    # 标记背景像素为透明
    alpha[mask] = 0
    
    removed_count = np.sum(mask)
    print(f"✓ 移除背景像素: {removed_count} ({removed_count/(h*w)*100:.1f}%)")
    
    # 合并RGB和Alpha
    result = np.dstack((img_array.astype(np.uint8), alpha))
    result_image = Image.fromarray(result, 'RGBA')
    
    # 对白色背景应用边缘羽化，消除白边
    if white_bg:
        result_image = refine_edges(result_image, bg_color, feather_radius=1)
    
    return result_image


def refine_edges(image, bg_color, feather_radius=1):
    """
    边缘精细化处理：消除白边/杂边 (向量化实现)
    
    参数:
        image: PIL Image对象 (RGBA)
        bg_color: 背景色 (R, G, B)
        feather_radius: 羽化半径
    
    返回:
        处理后的PIL Image对象
    """
    try:
        from scipy import ndimage
    except ImportError:
        print("⚠ scipy未安装，跳过边缘精细化")
        return image
    
    img_array = np.array(image)
    alpha = img_array[:, :, 3].astype(np.float32)
    rgb = img_array[:, :, :3].astype(np.float32)
    
    # 找到边缘像素 (alpha从0到255的过渡区)
    # 膨胀操作找到边缘外围
    alpha_binary = alpha > 0
    dilated = ndimage.binary_dilation(alpha_binary, iterations=feather_radius + 1)
    eroded = ndimage.binary_erosion(alpha_binary, iterations=max(1, feather_radius))
    edge_mask = dilated & ~eroded
    
    # 对边缘像素检查是否接近背景色 (向量化)
    bg_array = np.array(bg_color, dtype=np.float32)
    
    # 计算所有像素与背景色的距离
    distances = np.sqrt(np.sum((rgb - bg_array) ** 2, axis=2))
    
    # 对边缘区域中接近背景色的像素调整透明度
    # 距离阈值: 越接近背景色，透明度越低
    edge_threshold = 60
    adjustment_mask = edge_mask & (alpha > 0) & (distances < edge_threshold)
    
    # 根据距离计算新的透明度
    # 距离为0时透明度为0，距离为threshold时保持原透明度
    alpha_factor = np.clip(distances / edge_threshold, 0, 1)
    alpha[adjustment_mask] = alpha[adjustment_mask] * alpha_factor[adjustment_mask]
    
    adjusted_count = np.sum(adjustment_mask)
    if adjusted_count > 0:
        print(f"  边缘精细化: 调整了 {adjusted_count} 个边缘像素")
    
    # 合并结果
    result = np.dstack((rgb.astype(np.uint8), alpha.astype(np.uint8)))
    return Image.fromarray(result, 'RGBA')

def auto_crop_transparent(image, padding=0):
    """
    自动裁剪透明边缘，让主体"顶天立地"
    
    参数:
        image: PIL Image对象 (RGBA)
        padding: 保留的边距像素数
    
    返回:
        裁剪后的PIL Image对象
    """
    img_array = np.array(image)
    
    # 获取alpha通道
    if img_array.shape[2] == 4:
        alpha = img_array[:, :, 3]
    else:
        # 如果没有alpha通道，直接返回
        return image
    
    # 找到非透明像素的边界
    non_transparent = np.where(alpha > 0)
    
    if len(non_transparent[0]) == 0:
        # 完全透明，返回原图
        print("⚠ 警告: 图片完全透明，跳过裁剪")
        return image
    
    # 计算边界框
    y_min = max(0, non_transparent[0].min() - padding)
    y_max = min(img_array.shape[0], non_transparent[0].max() + 1 + padding)
    x_min = max(0, non_transparent[1].min() - padding)
    x_max = min(img_array.shape[1], non_transparent[1].max() + 1 + padding)
    
    # 裁剪
    cropped = image.crop((x_min, y_min, x_max, y_max))
    
    old_size = image.size
    new_size = cropped.size
    
    print(f"✓ 自动裁剪: {old_size} → {new_size} (节省 {(1 - new_size[0]*new_size[1]/(old_size[0]*old_size[1]))*100:.1f}% 空间)")
    
    return cropped

def normalize_width(images, target_width=None):
    """
    统一所有图片的宽度为最宽图片的宽度
    
    参数:
        images: PIL Image对象列表
        target_width: 目标宽度，如果为None则使用最宽图片的宽度
    
    返回:
        处理后的PIL Image对象列表
    """
    if not images:
        return images
    
    # 找到最宽的图片
    if target_width is None:
        target_width = max(img.width for img in images)
    
    print(f"统一宽度至: {target_width}px")
    
    normalized = []
    for img in images:
        if img.width < target_width:
            # 需要扩展宽度，居中放置
            new_height = img.height
            new_img = Image.new('RGBA', (target_width, new_height), (0, 0, 0, 0))
            
            # 计算居中位置
            x_offset = (target_width - img.width) // 2
            new_img.paste(img, (x_offset, 0), img if img.mode == 'RGBA' else None)
            
            normalized.append(new_img)
        else:
            normalized.append(img)
    
    return normalized

def process_image(input_path, output_path=None, tolerance=30, auto_crop=True, crop_padding=0):
    """
    处理单张图片
    
    参数:
        input_path: 输入图片路径
        output_path: 输出图片路径（如果为None，自动生成）
        tolerance: 颜色容差
        auto_crop: 是否自动裁剪透明边缘
        crop_padding: 裁剪时保留的边距
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到文件: {input_path}")
    
    # 加载图片
    print(f"加载图片: {input_path}")
    image = Image.open(input_path)
    print(f"图片大小: {image.size}")
    
    # 1. 先检测并移除黑边 (针对1280x720尺寸)
    image = detect_and_remove_black_borders(image)
    
    # 2. 检测背景色
    bg_color = detect_background_color(image)
    
    # 3. 移除背景
    result = remove_background(image, bg_color, tolerance)
    
    # 4. 自动裁剪透明边缘
    if auto_crop:
        result = auto_crop_transparent(result, padding=crop_padding)
    
    # 生成输出路径
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_nobg.png"
    
    # 保存结果
    result.save(output_path)
    print(f"✓ 已保存: {output_path}")
    
    return output_path

def process_single_image_wrapper(args):
    """
    包装函数用于多进程处理
    
    参数:
        args: (input_path, output_path, tolerance, auto_crop, crop_padding) 元组
    
    返回:
        (filename, success, error_msg)
    """
    input_path, output_path, tolerance, auto_crop, crop_padding = args
    filename = os.path.basename(input_path)
    
    try:
        process_image(input_path, output_path, tolerance, auto_crop, crop_padding)
        return (filename, True, None)
    except Exception as e:
        return (filename, False, str(e))

def process_directory(input_dir, output_dir=None, tolerance=30, num_workers=None, auto_crop=True, crop_padding=0):
    """
    批量处理目录中的所有图片（多核心并行）
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录（如果为None，创建 *_nobg 目录）
        tolerance: 颜色容差
        num_workers: 工作进程数（默认为CPU核心数）
        auto_crop: 是否自动裁剪透明边缘
        crop_padding: 裁剪时保留的边距
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"找不到目录: {input_dir}")
    
    # 生成输出目录
    if output_dir is None:
        output_dir = f"{input_dir}_nobg"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图片格式
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_dir) 
                   if os.path.splitext(f)[1].lower() in image_exts]
    
    if not image_files:
        print(f" 在 {input_dir} 中没有找到图片文件")
        return
    
    # 确定工作进程数
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"找到 {len(image_files)} 个图片文件")
    print(f"输出目录: {output_dir}")
    print(f"自动裁剪: {'开启' if auto_crop else '关闭'}")
    print(f"使用 {num_workers} 个进程并行处理\n")
    
    # 第一步: 处理所有图片（黑边检测 + 去背景 + 裁剪）
    processed_images = []
    image_filenames = []
    
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        try:
            print(f"处理: {filename}")
            image = Image.open(input_path)
            
            # 1. 先检测并移除黑边 (针对1280x720尺寸)
            image = detect_and_remove_black_borders(image)
            
            # 2. 检测背景色
            bg_color = detect_background_color(image)
            
            # 3. 移除背景
            result = remove_background(image, bg_color, tolerance)
            
            # 4. 自动裁剪透明边缘
            if auto_crop:
                result = auto_crop_transparent(result, padding=crop_padding)
            
            processed_images.append(result)
            image_filenames.append(filename)
            
        except Exception as e:
            print(f" 处理失败 {filename}: {e}")
    
    if not processed_images:
        print(" 没有成功处理的图片")
        return
    
    # 第二步: 统一宽度
    print(f"\n统一所有图片宽度...")
    normalized_images = normalize_width(processed_images)
    
    # 第三步: 保存结果
    print(f"\n保存处理后的图片...")
    success_count = 0
    fail_count = 0
    
    for filename, image in zip(image_filenames, normalized_images):
        try:
            output_path = os.path.join(output_dir, filename)
            image.save(output_path)
            print(f"✓ 已保存: {filename}")
            success_count += 1
        except Exception as e:
            print(f" 保存失败 {filename}: {e}")
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"批量处理完成!")
    print(f"  - 成功: {success_count}")
    print(f"  - 失败: {fail_count}")
    print(f"  - 输出目录: {output_dir}")
    print(f"{'='*60}")
