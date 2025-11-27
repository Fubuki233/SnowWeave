import os
import time
from datetime import datetime
from PIL import Image
import tempfile
import shutil
from google import genai
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path

from lib.generate_sprite_animation import (
    load_reference_image,
    generate_animation_video
)
from lib.extract_sprite_frames import (
    extract_frames_from_video_segment,
    create_sprite_sheet,
    save_individual_frames
)
from lib.remove_background import (
    process_directory,
    process_image,
    detect_background_color
)


class SnowWeaveAPI:
    
    AVAILABLE_MODELS = {
        "veo-3.1-generate-preview": "Veo 3.1",
        "veo-3.1-fast-generate-preview": "Veo 3.1 Fast",
        "veo-3.0-generate-001": "Veo 3.0",
        "veo-3.0-fast-generate-001": "Veo 3.0 Fast",
        "veo-2.0-generate-001": "Veo 2.0",
    }
    DEFAULT_MODEL = "veo-3.1-generate-preview"
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "api_outputs"):
        self.gemini_client = None
        self.api_key = api_key
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        if api_key:
            self.initialize_api(api_key)
    
    def initialize_api(self, api_key: str) -> bool:
        try:
            self.gemini_client = genai.Client(api_key=api_key)
            self.api_key = api_key
            return True
        except Exception as e:
            raise Exception(f"API key initialization failed: {str(e)}")
    
    def generate_video(
        self,
        image_path: str,
        action: str = "walking animation, side view, loop",
        model_name: str = None,
        duration: int = 6,
        output_subdir: Optional[str] = None
    ) -> Dict[str, str]:
        
        if self.gemini_client is None:
            raise Exception("API not initialized. Call initialize_api() first.")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        model_name = model_name or self.DEFAULT_MODEL
        
        reference_image = load_reference_image(image_path)
        img_width, img_height = reference_image.size
        
        full_prompt = f"""
Create a smooth sprite animation of a STYLIZED, NON-REALISTIC game character performing {action} IN PLACE.

IMPORTANT - CHARACTER STYLE:
- This is a FICTIONAL GAME CHARACTER, not a real person
- Use CARTOON/PIXEL ART style with simplified features
- ABSTRACT or STYLIZED representation only
- NO photorealistic human features
- Game sprite aesthetic

CRITICAL REQUIREMENTS:
- START IMMEDIATELY with the character visible - NO fade in effect No Irrelevant actions
- Character STAYS IN THE CENTER, does NOT move left or right across the screen
- Only the character's body/limbs animate, position remains FIXED
- Smooth, fluid animation with natural motion
- Complete {action} cycle IN PLACE
- Keep the exact same character design, colors, and art style
- Loop-able animation cycle
- Video dimensions MUST match the reference image: {img_width}x{img_height} pixels
- Maintain exact aspect ratio of {img_width}:{img_height}

VISUAL STYLE REQUIREMENTS:
- NO physics effects (no particles, debris, dust, etc.)
- NO lighting effects (no shadows, highlights, glows, reflections)
- NO post-processing effects (no blur, bloom, color grading)
- Flat, clean animation with solid colors only
- Simple sprite animation style without any special effects
- Background MUST be PURE CHROMA GREEN (#00FF00, RGB 0,255,0)

Style: Clean pixel art / 2D game sprite animation with smooth motion, no effects
Camera: Fixed, character stays in center and animates in place
Background: Pure chroma green (#00FF00) for entire duration
Resolution: {img_width}x{img_height} (match reference image exactly)
Effects: NONE - no physics, lighting, or post-processing effects
"""
        
        video = generate_animation_video(reference_image, full_prompt, self.gemini_client, model_name, duration)
        
        if video is None:
            raise Exception("Video generation failed: API returned empty result")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_subdir:
            output_dir = os.path.join(self.output_dir, output_subdir)
        else:
            output_dir = os.path.join(self.output_dir, f"video_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "animation.mp4")
        video_data = self.gemini_client.files.download(file=video.video)
        with open(output_path, "wb") as f:
            f.write(video_data)
        
        reference_path = os.path.join(output_dir, "reference_image.png")
        reference_image.save(reference_path)
        
        metadata_path = os.path.join(output_dir, "metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(f"Generation Time: {timestamp}\n")
            f.write(f"Action Description: {action}\n")
            f.write(f"Model Used: {model_name}\n")
            f.write(f"Video File: animation.mp4\n")
            f.write(f"Reference Image: reference_image.png\n")
        
        return {
            "video_path": os.path.abspath(output_path),
            "reference_path": os.path.abspath(reference_path),
            "metadata_path": os.path.abspath(metadata_path),
            "output_dir": os.path.abspath(output_dir)
        }
    
    def extract_frames(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float = 0.0,
        max_frames: int = 24,
        output_subdir: Optional[str] = None
    ) -> Dict[str, Any]:
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        frames = extract_frames_from_video_segment(
            video_path,
            float(start_time),
            float(end_time),
            int(max_frames)
        )
        
        if not frames:
            raise Exception("No frames extracted")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_subdir:
            output_dir = os.path.join(self.output_dir, output_subdir)
        else:
            output_dir = os.path.join(self.output_dir, f"frames_{timestamp}")
        
        frames_dir = os.path.join(output_dir, "frames")
        save_individual_frames(frames, output_dir=frames_dir)
        
        sprite_sheet, _ = create_sprite_sheet(frames, frame_size=None)
        sheet_path = os.path.join(output_dir, "sprite_sheet.png")
        sprite_sheet.save(sheet_path)
        
        return {
            "sprite_sheet_path": os.path.abspath(sheet_path),
            "frames_dir": os.path.abspath(frames_dir),
            "frame_count": len(frames),
            "output_dir": os.path.abspath(output_dir),
            "frames": frames
        }
    
    def remove_background(
        self,
        input_path: str,
        tolerance: int = 180,
        auto_crop: bool = False,
        crop_padding: int = 0,
        output_subdir: Optional[str] = None
    ) -> Dict[str, Any]:
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input not found: {input_path}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_subdir:
            output_dir = os.path.join(self.output_dir, output_subdir)
        else:
            output_dir = os.path.join(self.output_dir, f"nobg_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        if os.path.isdir(input_path):
            nobg_dir = os.path.join(output_dir, "frames")
            process_directory(
                input_path,
                output_dir=nobg_dir,
                tolerance=tolerance,
                num_workers=None,
                auto_crop=auto_crop,
                crop_padding=crop_padding
            )
            
            nobg_files = sorted([f for f in os.listdir(nobg_dir) if f.endswith('.png')])
            if nobg_files:
                processed_images = [Image.open(os.path.join(nobg_dir, f)) for f in nobg_files]
                final_sheet, _ = create_sprite_sheet(processed_images, frame_size=None)
                sheet_path = os.path.join(output_dir, "sprite_sheet.png")
                final_sheet.save(sheet_path)
            else:
                sheet_path = None
            
            return {
                "sprite_sheet_path": os.path.abspath(sheet_path) if sheet_path else None,
                "frames_dir": os.path.abspath(nobg_dir),
                "frame_count": len(nobg_files),
                "output_dir": os.path.abspath(output_dir)
            }
        else:
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_dir, filename)
            
            process_image(
                input_path,
                output_path=output_path,
                tolerance=tolerance,
                auto_crop=auto_crop,
                crop_padding=crop_padding
            )
            
            return {
                "output_path": os.path.abspath(output_path),
                "output_dir": os.path.abspath(output_dir)
            }
    
    def full_pipeline(
        self,
        image_path: str,
        action: str = "walking animation, side view, loop",
        start_time: float = 0.0,
        end_time: float = 0.0,
        max_frames: int = 24,
        tolerance: int = 180,
        auto_crop: bool = False,
        crop_padding: int = 0,
        model_name: str = None,
        duration: int = 6,
        output_subdir: Optional[str] = None
    ) -> Dict[str, Any]:
        
        if self.gemini_client is None:
            raise Exception("API not initialized. Call initialize_api() first.")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        model_name = model_name or self.DEFAULT_MODEL
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_subdir:
            output_base = os.path.join(self.output_dir, output_subdir)
        else:
            output_base = os.path.join(self.output_dir, f"full_{timestamp}")
        os.makedirs(output_base, exist_ok=True)
        
        reference_image = load_reference_image(image_path)
        img_width, img_height = reference_image.size
        
        full_prompt = f"""
Create a smooth sprite animation of a STYLIZED, NON-REALISTIC game character performing {action} IN PLACE.

IMPORTANT - CHARACTER STYLE:
- This is a FICTIONAL GAME CHARACTER, not a real person
- Use CARTOON/PIXEL ART style with simplified features
- ABSTRACT or STYLIZED representation only
- NO photorealistic human features
- Game sprite aesthetic

CRITICAL REQUIREMENTS:
- START IMMEDIATELY with the character visible - NO fade in effect
- Character STAYS IN THE CENTER, does NOT move left or right across the screen
- Only the character's body/limbs animate, position remains FIXED
- Smooth, fluid animation with natural motion
- Complete {action} cycle IN PLACE
- Keep the exact same character design, colors, and art style
- Loop-able animation cycle
- Video dimensions MUST match the reference image: {img_width}x{img_height} pixels
- Maintain exact aspect ratio of {img_width}:{img_height}

VISUAL STYLE REQUIREMENTS:
- NO physics effects (no particles, debris, dust, etc.)
- NO lighting effects (no shadows, highlights, glows, reflections)
- NO post-processing effects (no blur, bloom, color grading)
- Flat, clean animation with solid colors only
- Simple sprite animation style without any special effects
- Background MUST be PURE CHROMA GREEN (#00FF00, RGB 0,255,0)

Style: Clean pixel art / 2D game sprite animation with smooth motion, no effects
Camera: Fixed, character stays in center and animates in place
Background: Pure chroma green (#00FF00) for entire duration
Resolution: {img_width}x{img_height} (match reference image exactly)
Effects: NONE
"""
        
        video = generate_animation_video(reference_image, full_prompt, self.gemini_client, model_name, duration)
        
        if video is None:
            raise Exception("Video generation failed: API returned empty result")
        
        video_path = os.path.join(output_base, "animation.mp4")
        video_data = self.gemini_client.files.download(file=video.video)
        with open(video_path, "wb") as f:
            f.write(video_data)
        
        reference_path = os.path.join(output_base, "reference_image.png")
        reference_image.save(reference_path)
        
        metadata_path = os.path.join(output_base, "metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(f"=== SnowWeave Full Pipeline Output ===\n\n")
            f.write(f"Generation Time: {timestamp}\n")
            f.write(f"Action Description: {action}\n")
            f.write(f"Model Used: {model_name}\n\n")
            f.write(f"=== Video Generation Parameters ===\n")
            f.write(f"Extraction Time Range: {start_time}s - {end_time}s\n")
            f.write(f"Max Frames: {max_frames}\n\n")
            f.write(f"=== Background Removal Parameters ===\n")
            f.write(f"Color Tolerance: {tolerance}\n")
            f.write(f"Auto Crop: {auto_crop}\n")
            f.write(f"Crop Padding: {crop_padding}px\n\n")
            f.write(f"=== Output Files ===\n")
            f.write(f"Video: animation.mp4\n")
            f.write(f"Reference Image: reference_image.png\n")
            f.write(f"Original Extracted Frames: 1_extracted_frames/\n")
            f.write(f"No-Background Frames: 2_nobg_frames/\n")
            f.write(f"Original Sprite Sheet: 1_original_sprite_sheet.png\n")
            f.write(f"Final Sprite Sheet: 3_final_sprite_sheet.png\n")
        
        frames = extract_frames_from_video_segment(
            video_path,
            float(start_time),
            float(end_time),
            int(max_frames)
        )
        
        frames_dir = os.path.join(output_base, "1_extracted_frames")
        save_individual_frames(frames, output_dir=frames_dir)
        
        original_sheet, _ = create_sprite_sheet(frames, frame_size=None)
        original_sheet_path = os.path.join(output_base, "1_original_sprite_sheet.png")
        original_sheet.save(original_sheet_path)
        
        nobg_dir = os.path.join(output_base, "2_nobg_frames")
        process_directory(
            frames_dir,
            output_dir=nobg_dir,
            tolerance=int(tolerance),
            num_workers=None,
            auto_crop=auto_crop,
            crop_padding=int(crop_padding)
        )
        
        nobg_files = sorted([f for f in os.listdir(nobg_dir) if f.endswith('.png')])
        final_frames = [Image.open(os.path.join(nobg_dir, f)) for f in nobg_files]
        
        final_sheet, _ = create_sprite_sheet(final_frames, frame_size=None)
        final_sheet_path = os.path.join(output_base, "3_final_sprite_sheet.png")
        final_sheet.save(final_sheet_path)
        
        return {
            "video_path": os.path.abspath(video_path),
            "reference_path": os.path.abspath(reference_path),
            "metadata_path": os.path.abspath(metadata_path),
            "original_frames_dir": os.path.abspath(frames_dir),
            "nobg_frames_dir": os.path.abspath(nobg_dir),
            "original_sprite_sheet": os.path.abspath(original_sheet_path),
            "final_sprite_sheet": os.path.abspath(final_sheet_path),
            "output_dir": os.path.abspath(output_base),
            "frame_count": len(frames),
            "final_frame_count": len(final_frames)
        }
    
    def clean_old_outputs(self, pattern: str = None):
        if pattern:
            pattern = f"{pattern}_*"
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                if pattern is None or item.startswith(pattern.split('_')[0]):
                    shutil.rmtree(item_path)
