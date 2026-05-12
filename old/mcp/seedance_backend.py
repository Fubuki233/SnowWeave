"""
Seedance Video Generation Backend (MCP Standalone)
豆包 Seedance 1.5 视频生成后端（MCP 独立版）

用于生成角色动画视频
"""

import os
import time
import base64
import tempfile
import requests
from io import BytesIO
from typing import Optional, List
from PIL import Image
from dataclasses import dataclass


@dataclass
class VideoResult:
    """视频生成结果"""
    video_path: str  # 视频文件路径
    video_data: bytes  # 视频原始数据
    duration: int  # 视频时长
    width: int  # 视频宽度
    height: int  # 视频高度
    raw_response: dict = None  # 原始响应


class SeedanceBackend:
    """豆包 Seedance 1.5 视频生成后端"""
    
    API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
    MODEL_ID = "doubao-seedance-1-5-pro-251215"
    
    def __init__(self, api_key: str, **kwargs):
        """
        初始化 Seedance 后端
        
        Args:
            api_key: 火山引擎 ARK API Key
            **kwargs: 其他配置
                - resolution: 分辨率 ("480p", "720p", "1080p")
                - watermark: 是否添加水印
                - draft: 是否使用预览模式
                - generate_audio: 是否生成音频
                - camera_fixed: 是否固定相机
        """
        self.api_key = api_key
        self.resolution = kwargs.get("resolution", "480p")
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
        duration: int,
        resolution: str = "480p",
        draft: bool = True,
        generate_audio: bool = False
    ) -> str:
        """
        创建视频生成任务
        
        Args:
            prompt: 提示词
            image: 首帧图片
            duration: 视频时长
            resolution: 分辨率
            draft: 是否使用预览模式
            generate_audio: 是否生成音频
        
        Returns:
            task_id: 任务ID
        """
        url = f"{self.API_BASE}/contents/generations/tasks"
        
        # 构建 content 数组
        content = [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": self._image_to_base64_url(image)
                }
            }
        ]
        
        # 构建请求体 - 使用新的参数格式
        payload = {
            "model": self.MODEL_ID,
            "content": content,
            "ratio": "adaptive",
            "duration": duration,
            "camera_fixed": self.camera_fixed,
            "watermark": self.watermark,
        }
        
        # 分辨率参数（draft 模式下强制使用 480p）
        if draft:
            payload["resolution"] = "480p"
        else:
            payload["resolution"] = resolution
        
        # 1.5 pro 特有参数
        payload["draft"] = draft
        payload["generate_audio"] = generate_audio
        
        print(f"[SeedanceBackend] Creating task...")
        print(f"[SeedanceBackend] Model: {self.MODEL_ID}")
        print(f"[SeedanceBackend] Duration: {duration}s, Resolution: {payload.get('resolution')}")
        print(f"[SeedanceBackend] Draft mode: {draft}, Generate audio: {generate_audio}")
        
        response = requests.post(url, headers=self._get_headers(), json=payload, timeout=90)
        
        if response.status_code != 200:
            error_detail = response.text
            raise RuntimeError(f"Failed to create task: HTTP {response.status_code} - {error_detail}")
        
        result = response.json()
        
        task_id = result.get("id")
        if not task_id:
            raise RuntimeError(f"No task ID in response: {result}")
        
        print(f"[SeedanceBackend] Task created: {task_id}")
        return task_id
    
    def _query_task(self, task_id: str, max_retries: int = 3) -> dict:
        """查询任务状态"""
        url = f"{self.API_BASE}/contents/generations/tasks/{task_id}"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self._get_headers(), timeout=60)
                
                if response.status_code != 200:
                    raise RuntimeError(f"Failed to query task: HTTP {response.status_code}")
                
                return response.json()
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"\n[SeedanceBackend] Query timeout, retry {attempt + 1}/{max_retries}...", end="", flush=True)
                    time.sleep(2)
                else:
                    raise TimeoutError(f"Task query timed out after {max_retries} attempts")
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\n[SeedanceBackend] Query failed: {e}, retry {attempt + 1}/{max_retries}...", end="", flush=True)
                    time.sleep(2)
                else:
                    raise
    
    def _wait_for_completion(self, task_id: str, timeout: int = 600, poll_interval: int = 5) -> dict:
        """等待任务完成"""
        start_time = time.time()
        print(f"[SeedanceBackend] Waiting for task {task_id}...", end="", flush=True)
        
        last_status = None
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
            
            result = self._query_task(task_id)
            status = result.get("status", "")
            
            if status != last_status:
                print(f"\n[SeedanceBackend] Status: {status}", end="", flush=True)
                last_status = status
            
            if status == "succeeded":
                print(f"\n[SeedanceBackend] Task completed!")
                return result
            elif status == "failed":
                error = result.get("error", {})
                error_msg = result.get("message", str(error))
                print(f"\n[SeedanceBackend] Task failed: {error_msg}")
                raise RuntimeError(f"Task failed: {error_msg}")
            elif status in ("pending", "running", "queued"):
                print(".", end="", flush=True)
                time.sleep(poll_interval)
            else:
                print(f"\n[SeedanceBackend] Unknown status: {status}")
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
                    print(f"[SeedanceBackend] Download timeout, retry {attempt + 1}/{max_retries}...")
                    time.sleep(2)
                else:
                    raise TimeoutError(f"Video download timed out after {max_retries} attempts")
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[SeedanceBackend] Download failed: {e}, retry {attempt + 1}/{max_retries}...")
                    time.sleep(2)
                else:
                    raise
    
    def generate_video(
        self,
        reference_image: Image.Image,
        prompt: str,
        duration: int = 4,
        **kwargs
    ) -> VideoResult:
        """
        使用 Seedance 生成视频
        
        Args:
            reference_image: 参考图片（首帧）
            prompt: 提示词
            duration: 视频时长（秒），1.5 pro 支持 4-12
            **kwargs:
                - resolution: 分辨率
                - timeout: 超时时间
                - draft: 是否使用预览模式
                - generate_audio: 是否生成音频
        
        Returns:
            VideoResult: 视频生成结果
        """
        if not self.is_available():
            raise RuntimeError("Seedance API key not provided")
        
        resolution = kwargs.get("resolution", self.resolution)
        timeout = kwargs.get("timeout", 600)
        draft = kwargs.get("draft", self.draft)
        generate_audio = kwargs.get("generate_audio", self.generate_audio)
        
        img_width, img_height = reference_image.size
        print(f"[SeedanceBackend] Image size: {img_width}x{img_height}")
        
        # 创建任务
        task_id = self._create_task(
            prompt=prompt,
            image=reference_image,
            duration=duration,
            resolution=resolution,
            draft=draft,
            generate_audio=generate_audio
        )
        
        # 等待完成
        result = self._wait_for_completion(task_id, timeout=timeout)
        
        # 获取视频 URL
        content = result.get("content", {})
        video_url = content.get("video_url")
        
        if not video_url:
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
            raw_response=result
        )
