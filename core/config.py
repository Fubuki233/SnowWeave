"""
SnowWeave Configuration Module
配置管理模块 - 包含常量、翻译和配置管理
"""

import os

# ============ 目录配置 ============
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "gradio_outputs")
DEFAULT_IMAGE_PATH = os.path.join(SCRIPT_DIR, "test.png")
DEFAULT_DIRT_IMAGE_PATH = os.path.join(SCRIPT_DIR, "dirt_E.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ 后端配置 ============
AVAILABLE_BACKENDS = {
    "gemini": "Google Gemini Veo (国际 / International)",
    "seedance": "字节跳动 Seedance (国内 / China)",
}

# ============ 模型配置 ============
# Gemini Veo 模型
GEMINI_MODELS = {
    "veo-3.1-generate-preview": "Veo 3.1 (Preview, Latest / 预览版，最新)",
    "veo-3.1-fast-generate-preview": "Veo 3.1 Fast (Preview, Fast / 预览版，快速)",
    "veo-3.0-generate-001": "Veo 3.0 (Stable / 稳定版)",
    "veo-3.0-fast-generate-001": "Veo 3.0 Fast (Stable, Fast / 稳定版，快速)",
    "veo-2.0-generate-001": "Veo 2.0 (Legacy / 旧版)",
}

# Seedance 模型
SEEDANCE_MODELS = {
    "doubao-seedance-1-0-pro-250528": "Seedance 1.0 Pro (Stable, 稳定版)",
    "doubao-seedance-1-0-pro-fast-251015": "Seedance 1.0 Pro Fast (快速版)",
    "doubao-seedance-1-0-lite-t2v-250428": "Seedance 1.0 Lite T2V (文生视频)",
    "doubao-seedance-1-0-lite-i2v-250428": "Seedance 1.0 Lite I2V (图生视频)",
}

DEFAULT_MODEL = "veo-3.1-generate-preview"

# ============ 语言配置 ============
LANGUAGES = {
    "zh": "中文",
    "en": "English"
}

# ============ 翻译字典 ============
TRANSLATIONS = {
    # API 相关
    "api_success": {"zh": "API密钥验证成功！", "en": "API key verified successfully!"},
    "api_failed": {"zh": "API密钥验证失败", "en": "API key verification failed"},
    "api_required": {"zh": "请先在设置中配置API密钥", "en": "Please configure API key in settings first"},
    
    # 输入验证
    "upload_image": {"zh": "请先上传图片", "en": "Please upload an image first"},
    "upload_video": {"zh": "请先上传视频", "en": "Please upload a video first"},
    
    # 模型相关
    "model_switched": {"zh": "已切换到模型", "en": "Switched to model"},
    
    # 视频生成
    "loading_image": {"zh": "正在加载图片...", "en": "Loading image..."},
    "generating_video": {"zh": "正在生成动画视频 (这可能需要几分钟)...", "en": "Generating animation video (this may take several minutes)..."},
    "downloading_video": {"zh": "正在下载视频...", "en": "Downloading video..."},
    "video_complete": {"zh": "视频生成完成!", "en": "Video generation complete!"},
    "video_failed": {"zh": "视频生成失败: API 返回空结果", "en": "Video generation failed: API returned empty result"},
    
    # 帧提取
    "extracting_frames": {"zh": "正在提取帧...", "en": "Extracting frames..."},
    "saving_frames": {"zh": "正在保存 {} 帧...", "en": "Saving {} frames..."},
    "no_frames": {"zh": "没有提取到帧", "en": "No frames extracted"},
    "extract_complete": {"zh": "提取完成!", "en": "Extraction complete!"},
    
    # 图片处理
    "processing_images": {"zh": "开始处理...", "en": "Starting processing..."},
    "processing_n_images": {"zh": "处理 {} 张图片...", "en": "Processing {} images..."},
    "processing_single": {"zh": "处理单张图片...", "en": "Processing single image..."},
    "creating_sprite": {"zh": "创建sprite sheet...", "en": "Creating sprite sheet..."},
    "complete": {"zh": "完成!", "en": "Complete!"},
    "bg_complete": {"zh": "背景去除完成!", "en": "Background removal complete!"},
    
    # 完整流程
    "step_generating": {"zh": "步骤1/4: 生成动画视频...", "en": "Step 1/4: Generating animation video..."},
    "step_extracting": {"zh": "步骤2/4: 提取帧...", "en": "Step 2/4: Extracting frames..."},
    "step_removing_bg": {"zh": "步骤3/4: 去除背景...", "en": "Step 3/4: Removing background..."},
    "step_final_sheet": {"zh": "步骤4/4: 生成最终Sprite Sheet...", "en": "Step 4/4: Generating final Sprite Sheet..."},
    "pipeline_complete": {"zh": "完整流程执行完成!", "en": "Full pipeline execution complete!"},
    
    # 植物生成相关
    "plant_generating_stage": {"zh": "正在生成阶段 {}/{} ...", "en": "Generating stage {}/{}..."},
    "plant_processing_stage": {"zh": "正在处理阶段 {} 帧...", "en": "Processing stage {} frames..."},
    "plant_resizing": {"zh": "正在缩放帧到 {}px 宽度...", "en": "Resizing frames to {}px width..."},
    "plant_complete": {"zh": "植物生成完成!", "en": "Plant generation complete!"},
    "plant_stage_failed": {"zh": "阶段 {} 生成失败", "en": "Stage {} generation failed"},
    "plant_need_input": {"zh": "请提供参考图片或描述提示词", "en": "Please provide reference image or description prompt"},
    "plant_need_id": {"zh": "请输入植物ID", "en": "Please enter plant ID"},
    
    # 通用
    "error": {"zh": "错误", "en": "Error"},
}


class Config:
    """全局配置管理类"""
    
    _instance = None
    _current_language = "zh"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def language(self):
        return self._current_language
    
    @language.setter
    def language(self, lang):
        if lang in LANGUAGES:
            self._current_language = lang
    
    @staticmethod
    def get_models_for_backend(backend: str) -> dict:
        """根据后端获取可用模型列表"""
        if backend == "seedance":
            return SEEDANCE_MODELS
        return GEMINI_MODELS


# 全局配置实例
_config = Config()


def t(key: str, *args) -> str:
    """
    获取翻译文本
    Get translated text
    
    Args:
        key: 翻译键
        *args: 格式化参数
    
    Returns:
        翻译后的文本
    """
    text = TRANSLATIONS.get(key, {}).get(_config.language, key)
    if args:
        return text.format(*args)
    return text


def set_language(lang: str) -> str:
    """
    设置界面语言
    Set interface language
    
    Args:
        lang: 语言代码 ('zh' 或 'en')
    
    Returns:
        状态消息
    """
    _config.language = lang
    if lang == "zh":
        return "已切换到中文"
    else:
        return "Switched to English"


def get_current_language() -> str:
    """获取当前语言"""
    return _config.language
