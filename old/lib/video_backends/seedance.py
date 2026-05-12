"""
ByteDance Seedance (Doubao) Video Backend
字节跳动豆包 Seedance 视频生成后端
"""

import os
import time
import base64
import tempfile
import requests
from io import BytesIO
from typing import Optional
from PIL import Image

from .base import VideoBackend, VideoResult


class SeedanceBackend(VideoBackend):
    """字节跳动豆包 Seedance 视频生成后端"""
    
    name = "seedance"
    
    # API 端点
    API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
    
    available_models = {
    "doubao-seedance-1-5-pro-251215": "Seedance 1.5 Pro (最新，推荐)",
    "doubao-seedance-1-0-pro-250528": "Seedance 1.0 Pro (Stable, 稳定版)",
    "doubao-seedance-1-0-pro-fast-251015": "Seedance 1.0 Pro Fast (快速版)",
    "doubao-seedance-1-0-lite-t2v-250428": "Seedance 1.0 Lite T2V (文生视频)",
    "doubao-seedance-1-0-lite-i2v-250428": "Seedance 1.0 Lite I2V (图生视频)",
}

    
    # 支持的分辨率
    RESOLUTIONS = ["480p", "720p", "1080p"]
    
    # 支持的时长
    DURATIONS = [2, 10]
    
    def __init__(self, api_key: str, **kwargs):
        """
        初始化 Seedance 后端
        
        Args:
            api_key: 火山引擎 ARK API Key
            **kwargs: 其他配置
                - resolution: 分辨率 ("480p", "720p", "1080p")
                - watermark: 是否添加水印
                - draft: 是否使用预览模式（仅 1.5 pro）
                - generate_audio: 是否生成音频（仅 1.5 pro）
        """
        super().__init__(api_key, **kwargs)
        self.resolution = kwargs.get("resolution", "720p")
        self.watermark = kwargs.get("watermark", False)
        self.camera_fixed = kwargs.get("camera_fixed", True)
        self.draft = kwargs.get("draft", True)  # 默认使用预览模式
        self.generate_audio = kwargs.get("generate_audio", False)  # 默认关闭音频
    
    def is_available(self) -> bool:
        """检查后端是否可用"""
        return bool(self.api_key)
    
    def _get_headers(self) -> dict:
        """获取 API 请求头"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _image_to_base64_url(self, image: Image.Image) -> str:
        """将 PIL Image 转换为 base64 data URL"""
        img_bytes = BytesIO()
        # 确保是 RGB 或 RGBA
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')
        image.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        b64_str = base64.b64encode(img_data).decode('utf-8')
        return f"data:image/png;base64,{b64_str}"
    
    def _create_task(
        self,
        prompt: str,
        image: Image.Image,
        model_name: str,
        duration: int,
        resolution: str = "480p",
        reference_images: list = None,
        draft: bool = True,
        generate_audio: bool = False
    ) -> str:
        """
        创建视频生成任务
        
        Args:
            prompt: 提示词
            image: 主参考图片（首帧/起始帧）
            model_name: 模型名称
            duration: 视频时长
            resolution: 分辨率
            reference_images: 额外参考图片列表（仅 lite-i2v 模型支持）
            draft: 是否使用预览模式（仅 1.5 pro 支持）
            generate_audio: 是否生成音频（仅 1.5 pro 支持）
        
        Returns:
            task_id: 任务ID
        """
        url = f"{self.API_BASE}/contents/generations/tasks"
        
        # 检测是否是 1.5 pro 模型
        is_1_5_pro = "1-5-pro" in model_name.lower() or "1.5-pro" in model_name.lower()
        
        # 构建 content 数组 - 只包含提示词，参数通过 request body 传递
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        
        # 检查是否是 lite-i2v 模型（需要指定图片 role）
        is_lite_i2v = "lite-i2v" in model_name.lower()
        print(f"[SeedanceBackend] model_name={model_name}, is_lite_i2v={is_lite_i2v}")
        
        if is_lite_i2v:
            # lite-i2v 模型：所有图片都必须指定 role
            # 主图作为 first_frame
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": self._image_to_base64_url(image)
                },
                "role": "first_frame"  # 首帧/起始帧
            })
            
            # 添加额外参考图（如果有）
            if reference_images:
                for ref_img in reference_images:
                    if isinstance(ref_img, str):
                        # URL 或文件路径
                        if ref_img.startswith(("http://", "https://")):
                            img_url = ref_img
                        else:
                            # 本地文件，转 base64
                            ref_pil = Image.open(ref_img)
                            img_url = self._image_to_base64_url(ref_pil)
                    elif isinstance(ref_img, Image.Image):
                        img_url = self._image_to_base64_url(ref_img)
                    else:
                        continue
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": img_url
                        },
                        "role": "reference_image"  # 参考图
                    })
                
                print(f"[SeedanceBackend] Using {len(reference_images)} additional reference images")
        else:
            # 其他模型：只用主图（无 role）
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": self._image_to_base64_url(image)
                }
            })
        
        # 构建请求体 - 使用新的参数格式（在 request body 中直接传参）
        payload = {
            "model": model_name,
            "content": content,
            # 视频参数
            "ratio": "adaptive",  # 自适应宽高比
            "duration": duration,
            "camera_fixed": self.camera_fixed,
            "watermark": self.watermark,
        }
        
        # 分辨率参数（draft 模式下强制使用 480p）
        if is_1_5_pro and draft:
            payload["resolution"] = "480p"  # draft 模式只支持 480p
        else:
            payload["resolution"] = resolution
        
        # 1.5 pro 特有参数
        if is_1_5_pro:
            payload["draft"] = draft  # 预览模式
            payload["generate_audio"] = generate_audio  # 关闭音频
            print(f"[SeedanceBackend] Seedance 1.5 Pro mode: draft={draft}, generate_audio={generate_audio}")
        
        print(f"[SeedanceBackend] Creating task...")
        print(f"[SeedanceBackend] Model: {model_name}")
        print(f"[SeedanceBackend] Duration: {duration}s, Resolution: {payload.get('resolution')}")
        print(f"[SeedanceBackend] Is 1.5 Pro: {is_1_5_pro}")
        print(f"[SeedanceBackend] Content structure (without image data):")
        for i, item in enumerate(content):
            if item.get("type") == "image_url":
                print(f"  [{i}] type=image_url, role={item.get('role', 'NOT SET')}, keys={list(item.keys())}")
            else:
                print(f"  [{i}] type={item.get('type')}")
        
        # 打印完整 payload 结构（不含图片数据）
        import json
        debug_payload = {k: v for k, v in payload.items() if k != "content"}
        debug_payload["content"] = []
        for item in content:
            if item.get("type") == "image_url":
                debug_item = {k: (v if k != "image_url" else {"url": "BASE64_DATA..."}) for k, v in item.items()}
                debug_payload["content"].append(debug_item)
            else:
                debug_payload["content"].append(item)
        print(f"[SeedanceBackend] Payload structure:\n{json.dumps(debug_payload, indent=2, ensure_ascii=False)}")
        
        response = requests.post(url, headers=self._get_headers(), json=payload, timeout=90)
        
        if response.status_code != 200:
            error_detail = response.text
            raise RuntimeError(f"Failed to create task: HTTP {response.status_code} - {error_detail}")
        
        result = response.json()
        
        # 获取任务 ID
        task_id = result.get("id")
        if not task_id:
            raise RuntimeError(f"No task ID in response: {result}")
        
        print(f"[SeedanceBackend] Task created: {task_id}")
        return task_id
    
    def _query_task(self, task_id: str, max_retries: int = 3) -> dict:
        """
        查询任务状态
        
        Args:
            task_id: 任务ID
            max_retries: 最大重试次数
        
        Returns:
            任务状态信息
        """
        url = f"{self.API_BASE}/contents/generations/tasks/{task_id}"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self._get_headers(), timeout=60)
                
                if response.status_code != 200:
                    raise RuntimeError(f"Failed to query task: HTTP {response.status_code}")
                
                return response.json()
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"\n[SeedanceBackend] 查询超时，重试 {attempt + 1}/{max_retries}...", end="", flush=True)
                    time.sleep(2)
                else:
                    raise TimeoutError(f"Task query timed out after {max_retries} attempts")
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\n[SeedanceBackend] 查询失败: {e}，重试 {attempt + 1}/{max_retries}...", end="", flush=True)
                    time.sleep(2)
                else:
                    raise
    
    def _wait_for_completion(self, task_id: str, timeout: int = 900, poll_interval: int = 5) -> dict:
        """
        等待任务完成
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）
            poll_interval: 轮询间隔（秒）
        
        Returns:
            任务结果
        """
        start_time = time.time()
        print(f"[SeedanceBackend] Waiting for task {task_id}...", end="", flush=True)
        
        last_status = None
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
            
            result = self._query_task(task_id)
            status = result.get("status", "")
            
            # 状态变化时打印详细信息
            if status != last_status:
                print(f"\n[SeedanceBackend] Status: {status}", end="", flush=True)
                last_status = status
                
                # 如果有额外信息，打印出来
                if "message" in result:
                    print(f" - {result['message']}", end="", flush=True)
            
            if status == "succeeded":
                print(f"\n[SeedanceBackend] Task completed!")
                return result
            elif status == "failed":
                error = result.get("error", {})
                error_msg = result.get("message", str(error))
                print(f"\n[SeedanceBackend] Task failed: {error_msg}")
                raise RuntimeError(f"Task failed: {error_msg}")
            elif status in ("pending", "running"):
                print(".", end="", flush=True)
                time.sleep(poll_interval)
            else:
                print(f"\n[SeedanceBackend] Unknown status: {status}, result: {result}")
                print(".", end="", flush=True)
                time.sleep(poll_interval)
    
    def _download_video(self, video_url: str, max_retries: int = 3) -> bytes:
        """下载视频"""
        print(f"[SeedanceBackend] Downloading video...")
        
        for attempt in range(max_retries):
            try:
                response = requests.get(video_url, timeout=180)
                if response.status_code != 200:
                    raise RuntimeError(f"Failed to download video: HTTP {response.status_code}")
                return response.content
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"[SeedanceBackend] 下载超时，重试 {attempt + 1}/{max_retries}...")
                    time.sleep(2)
                else:
                    raise TimeoutError(f"Video download timed out after {max_retries} attempts")
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[SeedanceBackend] 下载失败: {e}，重试 {attempt + 1}/{max_retries}...")
                    time.sleep(2)
                else:
                    raise
    
    def generate_video(
        self,
        reference_image: Image.Image,
        prompt: str,
        model_name: Optional[str] = None,
        duration: int = 4,
        **kwargs
    ) -> VideoResult:
        """
        使用 Seedance 生成视频
        
        Args:
            reference_image: 参考图片（主图/首帧）
            prompt: 提示词
            model_name: 模型名称
            duration: 视频时长（秒），1.5 pro 支持 4-12
            **kwargs:
                - resolution: 分辨率 ("480p", "720p", "1080p")
                - timeout: 超时时间（秒）
                - reference_images: 额外参考图片列表（仅 lite-i2v 模型支持）
                  可以是 PIL.Image、文件路径或 URL 的列表
                - draft: 是否使用预览模式（仅 1.5 pro）
                - generate_audio: 是否生成音频（仅 1.5 pro）
        
        Returns:
            VideoResult: 视频生成结果
        """
        if not self.is_available():
            raise RuntimeError("Seedance API key not provided")
        
        model_name = model_name or self.get_default_model()
        resolution = kwargs.get("resolution", self.resolution)
        timeout = kwargs.get("timeout", 600)
        reference_images = kwargs.get("reference_images", None)
        draft = kwargs.get("draft", self.draft)  # 默认使用预览模式
        generate_audio = kwargs.get("generate_audio", self.generate_audio)  # 默认关闭音频
        
        img_width, img_height = reference_image.size
        print(f"[SeedanceBackend] Image size: {img_width}x{img_height}")
        
        # 创建任务
        task_id = self._create_task(
            prompt=prompt,
            image=reference_image,
            model_name=model_name,
            duration=duration,
            resolution=resolution,
            reference_images=reference_images,
            draft=draft,
            generate_audio=generate_audio
        )
        
        # 等待完成
        result = self._wait_for_completion(task_id, timeout=timeout)
        
        # 获取视频 URL
        content = result.get("content", {})
        video_url = content.get("video_url")
        
        if not video_url:
            # 尝试其他字段
            videos = content.get("videos", [])
            if videos:
                video_url = videos[0].get("url")
        
        if not video_url:
            raise RuntimeError(f"No video URL in result: {result}")
        
        # 下载视频
        video_data = self._download_video(video_url)
        
        # 保存到临时文件
        temp_path = os.path.join(tempfile.gettempdir(), f"seedance_video_{int(time.time())}.mp4")
        with open(temp_path, "wb") as f:
            f.write(video_data)
        
        print(f"[SeedanceBackend] Video saved to: {temp_path}")
        
        return VideoResult(
            video_path=temp_path,
            video_data=video_data,
            duration=duration,
            width=img_width,
            height=img_height,
            backend=self.name,
            raw_response=result
        )
    
    def get_default_model(self) -> str:
        """获取默认模型"""
        return "doubao-seedance-1-5-pro-251215"
