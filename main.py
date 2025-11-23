"""
完整的sprite动画生成流水线
自动化整个流程: 生成动画 → 提取帧 → 去除背景 → 自动裁剪

使用方法:
    python main.py <输入文件> [动作描述] [选项]
    
示例:
    python main.py character.png
    python main.py video.mp4 --only-extract
    python main.py frames/ --only-remove-bg
    
环境变量:
    GEMINI_API_KEY: Gemini API密钥（必需）
"""

import os
import sys
import time
from datetime import datetime

# 导入各模块的功能
from generate_sprite_animation import (
    load_reference_image,
    generate_animation_video,
    client as gemini_client
)
from extract_sprite_frames import (
    extract_frames_from_video_segment,
    create_sprite_sheet,
    save_individual_frames
)
from remove_background import (
    process_directory
)

def print_banner(text):
    """打印美化的横幅"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def print_step(step_num, total_steps, description):
    """打印步骤信息"""
    print(f"\n{'─'*70}")
    print(f"📍 步骤 {step_num}/{total_steps}: {description}")
    print(f"{'─'*70}\n")

def cleanup_temp_files(*file_paths):
    """清理临时文件"""
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"  🗑️  清理临时文件: {file_path}")
            except Exception as e:
                print(f"  ⚠️  无法删除 {file_path}: {e}")

def main():
    # 解析命令行参数
    if len(sys.argv) < 2:
        print("用法: python main.py <输入文件> [动作描述] [选项]")
        print("\n参数说明:")
        print("  输入文件: 必需，根据模式不同:")
        print("    - 完整流程: 角色参考图片")
        print("    - 仅视频切片: 视频文件路径")
        print("    - 仅去除背景: 帧图片目录")
        print("  动作描述: 可选，默认 'walking animation' (仅生成视频时使用)")
        print("\n模式选项 (互斥，只能选择一个):")
        print("  --only-generate: 仅生成动画视频，不进行后续处理")
        print("  --only-extract: 仅从视频提取帧，输入为视频文件")
        print("  --only-remove-bg: 仅去除背景，输入为帧图片目录")
        print("\n其他选项:")
        print("  --start-time N: 视频提取开始时间（秒），默认 2.0")
        print("  --end-time N: 视频提取结束时间（秒），默认 3.0")
        print("  --max-frames N: 最大提取帧数，默认 8")
        print("  --tolerance N: 背景颜色容差，默认 30")
        print("  --no-crop: 禁用自动裁剪")
        print("  --padding N: 裁剪边距（像素），默认 0")
        print("  --keep-temp: 保留临时文件")
        print("  --output DIR: 指定输出目录")
        print("\n示例:")
        print('  # 完整流程')
        print('  python main.py character.png')
        print('  python main.py goblin.png "running animation"')
        print('')
        print('  # 仅生成视频')
        print('  python main.py character.png --only-generate')
        print('  python main.py warrior.png "attack animation" --only-generate')
        print('')
        print('  # 仅提取帧')
        print('  python main.py video.mp4 --only-extract')
        print('  python main.py video.mp4 --only-extract --start-time 1.5 --end-time 2.5')
        print('')
        print('  # 仅去除背景')
        print('  python main.py extracted_frames/ --only-remove-bg')
        print('  python main.py frames/ --only-remove-bg --tolerance 40 --no-crop')
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # 解析参数
    action = "The character keeps walking in place from a side view"
    start_time = 0
    end_time = 5.0
    max_frames = 8
    tolerance = 30
    auto_crop = True
    crop_padding = 0
    keep_temp = False
    output_dir = None
    
    # 模式选择
    only_generate = False
    only_extract = False
    only_remove_bg = False
    
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == '--only-generate':
            only_generate = True
            i += 1
        elif arg == '--only-extract':
            only_extract = True
            i += 1
        elif arg == '--only-remove-bg':
            only_remove_bg = True
            i += 1
        elif arg == '--output':
            output_dir = sys.argv[i + 1]
            i += 2
        elif arg == '--start-time':
            start_time = float(sys.argv[i + 1])
            i += 2
        elif arg == '--end-time':
            end_time = float(sys.argv[i + 1])
            i += 2
        elif arg == '--max-frames':
            max_frames = int(sys.argv[i + 1])
            i += 2
        elif arg == '--tolerance':
            tolerance = int(sys.argv[i + 1])
            i += 2
        elif arg == '--no-crop':
            auto_crop = False
            i += 1
        elif arg == '--padding':
            crop_padding = int(sys.argv[i + 1])
            i += 2
        elif arg == '--keep-temp':
            keep_temp = True
            i += 1
        elif arg.startswith('--'):
            print(f"× 错误: 未知选项 {arg}")
            sys.exit(1)
        else:
            # 第一个非选项参数是动作描述
            if i == 2:
                action = arg
            i += 1
    
    # 检查模式互斥性
    mode_count = sum([only_generate, only_extract, only_remove_bg])
    if mode_count > 1:
        print("× 错误: 只能选择一个模式 (--only-generate, --only-extract, --only-remove-bg)")
        sys.exit(1)
    
    # 检查输入文件
    if not os.path.exists(input_path):
        print(f"× 错误: 找不到文件/目录 {input_path}")
        sys.exit(1)
    
    # 开始流水线
    start_overall = time.time()
    
    # 生成输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if only_extract:
            output_dir = f"extracted_{timestamp}"
        elif only_remove_bg:
            output_dir = f"nobg_{timestamp}"
        else:
            output_dir = f"output_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # ========== 模式1: 仅去除背景 ==========
        if only_remove_bg:
            print_banner("🎨 仅去除背景模式")
            print(f"📋 配置:")
            print(f"  - 输入目录: {input_path}")
            print(f"  - 输出目录: {output_dir}")
            print(f"  - 背景容差: {tolerance}")
            print(f"  - 自动裁剪: {'是' if auto_crop else '否'}")
            if auto_crop and crop_padding > 0:
                print(f"  - 裁剪边距: {crop_padding}px")
            
            if not os.path.isdir(input_path):
                print(f"× 错误: {input_path} 不是目录")
                sys.exit(1)
            
            # 去除背景
            nobg_dir = os.path.join(output_dir, "frames")
            process_directory(
                input_path,
                output_dir=nobg_dir,
                tolerance=tolerance,
                num_workers=None,
                auto_crop=auto_crop,
                crop_padding=crop_padding
            )
            
            # 创建sprite sheet
            from PIL import Image
            nobg_files = sorted([f for f in os.listdir(nobg_dir) if f.endswith('.png')])
            if nobg_files:
                final_frames = [Image.open(os.path.join(nobg_dir, f)) for f in nobg_files]
                final_sheet, _ = create_sprite_sheet(final_frames, frame_size=None)
                final_sheet_path = os.path.join(output_dir, "sprite_sheet.png")
                final_sheet.save(final_sheet_path)
                print(f"\n✓ Sprite sheet 已保存: {final_sheet_path}")
            
            print_banner("✅ 背景去除完成!")
            print(f"⏱️  耗时: {time.time() - start_overall:.1f} 秒")
            print(f"📁 输出目录: {output_dir}/")
        
        # ========== 模式2: 仅提取帧 ==========
        elif only_extract:
            print_banner("✂️ 仅视频切片模式")
            print(f"📋 配置:")
            print(f"  - 输入视频: {input_path}")
            print(f"  - 输出目录: {output_dir}")
            print(f"  - 提取时间段: {start_time}s - {end_time}s")
            print(f"  - 最大帧数: {max_frames}")
            
            if not os.path.isfile(input_path):
                print(f"× 错误: {input_path} 不是文件")
                sys.exit(1)
            
            # 提取帧
            frames = extract_frames_from_video_segment(input_path, start_time, end_time, max_frames)
            
            if not frames:
                raise ValueError("没有提取到任何帧")
            
            # 保存帧
            frames_dir = os.path.join(output_dir, "frames")
            save_individual_frames(frames, output_dir=frames_dir)
            
            # 创建sprite sheet
            sprite_sheet, _ = create_sprite_sheet(frames, frame_size=None)
            sheet_path = os.path.join(output_dir, "sprite_sheet.png")
            sprite_sheet.save(sheet_path)
            print(f"\n✓ Sprite sheet 已保存: {sheet_path}")
            
            print_banner("✅ 视频切片完成!")
            print(f"⏱️  耗时: {time.time() - start_overall:.1f} 秒")
            print(f"📁 输出目录: {output_dir}/")
            print(f"🎞️  提取帧数: {len(frames)}")
        
        # ========== 模式3: 仅生成视频 ==========
        elif only_generate:
            print_banner("🎬 仅生成视频模式")
            print(f"📋 配置:")
            print(f"  - 角色图片: {input_path}")
            print(f"  - 动作: {action}")
            print(f"  - 输出目录: {output_dir}")
            
            if not os.path.isfile(input_path):
                print(f"× 错误: {input_path} 不是文件")
                sys.exit(1)
            
            # 加载参考图片
            reference_image = load_reference_image(input_path)
            print(f"✓ 图片已加载: {reference_image.size}")
            
            # 生成动画视频
            full_prompt = f"""
Create a smooth sprite animation of the character {action} IN PLACE (not moving across screen).

CRITICAL REQUIREMENTS:
- START IMMEDIATELY with the character visible - NO fade in effect
- Character STAYS IN THE CENTER, does NOT move left or right across the screen
- Only the character's body/limbs animate, position remains FIXED
- Smooth, fluid animation with natural motion
- Complete {action} cycle IN PLACE
- Keep the exact same character design, colors, and art style
- Loop-able animation cycle

VISUAL STYLE REQUIREMENTS:
- NO physics effects (no particles, debris, dust, etc.)
- NO lighting effects (no shadows, highlights, glows, reflections)
- NO post-processing effects (no blur, bloom, color grading)
- Flat, clean animation with solid colors only
- Simple sprite animation style without any special effects
- Background MUST be PURE CHROMA GREEN (#00FF00, RGB 0,255,0)

Style: Clean pixel art / 2D game sprite animation with smooth motion, no effects
Camera: Fixed, character stays in center and animates in place
Background: Pure chroma green (#00FF00) for entire duration - FOR POST-PRODUCTION EDITING
Transitions: None - instant start, instant character removal at end, green background constant
Effects: NONE - no physics, lighting, or post-processing effects
"""
            
            video = generate_animation_video(reference_image, full_prompt)
            
            # 下载视频
            video_path = os.path.join(output_dir, "animation.mp4")
            print(f"正在下载视频到 {video_path}...")
            video_data = gemini_client.files.download(file=video.video)
            with open(video_path, "wb") as f:
                f.write(video_data)
            print(f"✓ 视频已保存: {video_path}")
            
            print_banner("✅ 视频生成完成!")
            print(f"⏱️  耗时: {time.time() - start_overall:.1f} 秒")
            print(f"📹 视频文件: {video_path}")
        
        # ========== 模式4: 完整流程 ==========
        else:
            print_banner("🎬 Sprite动画生成流水线")
            
            print(f"📋 配置:")
            print(f"  - 角色图片: {input_path}")
            print(f"  - 动作: {action}")
            print(f"  - 提取时间段: {start_time}s - {end_time}s")
            print(f"  - 最大帧数: {max_frames}")
            print(f"  - 背景容差: {tolerance}")
            print(f"  - 自动裁剪: {'是' if auto_crop else '否'}")
            if auto_crop and crop_padding > 0:
                print(f"  - 裁剪边距: {crop_padding}px")
            
            if not os.path.isfile(input_path):
                print(f"× 错误: {input_path} 不是文件")
                sys.exit(1)
            
            # ========== 步骤 1: 加载参考图片 ==========
            print_step(1, 5, "加载角色参考图片")
            reference_image = load_reference_image(input_path)
            print(f"✓ 图片已加载: {reference_image.size}")
            
            # ========== 步骤 2: 生成动画视频 ==========
            print_step(2, 5, "使用 Gemini Veo 生成动画视频")
            
            full_prompt = f"""
Create a smooth sprite animation of the character {action} IN PLACE (not moving across screen).

CRITICAL REQUIREMENTS:
- START IMMEDIATELY with the character visible - NO fade in effect
- Character STAYS IN THE CENTER, does NOT move left or right across the screen
- Only the character's body/limbs animate, position remains FIXED
- Smooth, fluid animation with natural motion
- Complete {action} cycle IN PLACE
- Keep the exact same character design, colors, and art style
- Loop-able animation cycle

VISUAL STYLE REQUIREMENTS:
- NO physics effects (no particles, debris, dust, etc.)
- NO lighting effects (no shadows, highlights, glows, reflections)
- NO post-processing effects (no blur, bloom, color grading)
- Flat, clean animation with solid colors only
- Simple sprite animation style without any special effects
- Background MUST be PURE CHROMA GREEN (#00FF00, RGB 0,255,0)

Style: Clean pixel art / 2D game sprite animation with smooth motion, no effects
Camera: Fixed, character stays in center and animates in place
Background: Pure chroma green (#00FF00) for entire duration - FOR POST-PRODUCTION EDITING
Transitions: None - instant start, instant character removal at end, green background constant
Effects: NONE - no physics, lighting, or post-processing effects
"""
            
            video = generate_animation_video(reference_image, full_prompt)
            
            # 下载视频
            temp_video_path = os.path.join(output_dir, "temp_animation.mp4")
            print(f"正在下载视频到 {temp_video_path}...")
            video_data = gemini_client.files.download(file=video.video)
            with open(temp_video_path, "wb") as f:
                f.write(video_data)
            print("✓ 视频已下载")
            
            # ========== 步骤 3: 提取帧 ==========
            print_step(3, 5, "从视频中提取Sprite帧")
            frames = extract_frames_from_video_segment(temp_video_path, start_time, end_time, max_frames)
            
            if not frames:
                raise ValueError("没有提取到任何帧")
            
            # 保存原始提取的帧
            extracted_dir = os.path.join(output_dir, "1_extracted_frames")
            save_individual_frames(frames, output_dir=extracted_dir)
            
            # 创建原始sprite sheet
            sprite_sheet, _ = create_sprite_sheet(frames, frame_size=None)
            original_sheet_path = os.path.join(output_dir, "1_original_sprite_sheet.png")
            sprite_sheet.save(original_sheet_path)
            print(f"✓ 原始 Sprite sheet 已保存: {original_sheet_path}")
            
            # ========== 步骤 4: 去除背景 ==========
            print_step(4, 5, "去除背景")
            nobg_dir = os.path.join(output_dir, "2_nobg_frames")
            process_directory(
                extracted_dir,
                output_dir=nobg_dir,
                tolerance=tolerance,
                num_workers=None,
                auto_crop=auto_crop,
                crop_padding=crop_padding
            )
            
            # ========== 步骤 5: 创建最终Sprite Sheet ==========
            print_step(5, 5, "生成最终Sprite Sheet")
            
            # 读取处理后的帧
            from PIL import Image
            nobg_files = sorted([f for f in os.listdir(nobg_dir) if f.endswith('.png')])
            final_frames = [Image.open(os.path.join(nobg_dir, f)) for f in nobg_files]
            
            # 创建最终sprite sheet
            final_sheet, _ = create_sprite_sheet(final_frames, frame_size=None)
            final_sheet_path = os.path.join(output_dir, "3_final_sprite_sheet.png")
            final_sheet.save(final_sheet_path)
            print(f"✓ 最终 Sprite sheet 已保存: {final_sheet_path}")
            
            # 清理临时文件
            if not keep_temp:
                print(f"\n{'─'*70}")
                print("🗑️  清理临时文件")
                print(f"{'─'*70}\n")
                cleanup_temp_files(temp_video_path)
            
            # 输出总结
            print_banner("✅ 流水线执行完成!")
            
            print(f"⏱️  总耗时: {time.time() - start_overall:.1f} 秒")
            print(f"\n📁 输出目录: {output_dir}/")
            print(f"\n生成的文件:")
            print(f"  1️⃣  原始提取帧: {extracted_dir}/")
            print(f"  2️⃣  去背景帧: {nobg_dir}/")
            print(f"  3️⃣  原始Sprite Sheet: {original_sheet_path}")
            print(f"  4️⃣  最终Sprite Sheet: {final_sheet_path}")
            
            if keep_temp:
                print(f"  📹 视频文件: {temp_video_path}")
            
            print(f"\n🎮 可直接在游戏引擎中使用:")
            print(f"  - 导入: {final_sheet_path}")
            print(f"  - 帧数: {len(final_frames)}")
            print(f"  - 单帧尺寸: {final_frames[0].size if final_frames else 'N/A'}")
            
            print("\n" + "="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
