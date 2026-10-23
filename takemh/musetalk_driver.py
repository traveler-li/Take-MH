"""面部驱动模块：MuseTalk 官方推理封装，音频流 -> 口型对齐的 BGR 视频帧。"""

from __future__ import annotations

import logging
import os
import queue
import tempfile
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from takemh.audio_io import audio_to_wav_bytes
from takemh.config import MuseTalkConfig

from queue import Empty as _QueueEmpty

logger = logging.getLogger(__name__)

_SENTINEL = object()


class MuseTalkDriver:
    """MuseTalk 实时帧生成驱动。

    官方 realtime_inference.py 的链路是"整段音频 -> 帧序列"，并不支持逐帧实时；
    本类按官方接口实现最小可行的流式循环：把输入音频分块（chunk_seconds），
    每块走一遍 特征提取 -> get_whisper_chunk -> UNet -> VAE 解码 -> 混合 的官方流程，
    产出带时间戳的 BGR 帧供渲染线程按音频进度播放，实现音画对齐。

    - push_audio(pcm16_bytes): 输入 16k int16 PCM
    - get_frame(timeout): 取出下一帧 BGR ndarray
    - get_frame_with_timestamp(timeout): 取出 (预计播放时刻秒, BGR 帧)
    """

    def __init__(self, cfg: MuseTalkConfig):
        self.cfg = cfg
        self._frame_queue: "queue.Queue" = queue.Queue(maxsize=cfg.frame_queue_size)
        self._audio_queue: "queue.Queue" = queue.Queue(maxsize=cfg.audio_queue_size)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._consumed_seconds = 0.0
        self._consumed_lock = threading.Lock()

        # 模型与基准照片材料（load() 后填充）
        self._vae = None
        self._unet = None
        self._pe = None
        self._whisper = None
        self._audio_processor = None
        self._device = None
        self._timesteps = None
        self._latents_cycle = None
        self._bbox = None
        self._ori_frame = None
        self._mask = None
        self._mask_crop_box = None

    def load(self) -> None:
        """加载 MuseTalk 全部模型并按基准照片准备静态材料（官方接口）。"""
        # 确保官方仓库路径在 sys.path 中
        _repo_root = Path(__file__).resolve().parents[1]  # 项目根目录 D:/GitHub/Take-MH
        _repos = [
            _repo_root.parent / "CosyVoice",
            _repo_root.parent / "MuseTalk",
            _repo_root.parent / "MuseTalk" / "musetalk" / "utils",
        ]
        import sys
        for _p in _repos:
            _s = str(_p.resolve())
            if _p.is_dir() and _s not in sys.path:
                sys.path.insert(0, _s)

        torch_cache = Path(__file__).resolve().parents[1] / ".torch_cache"
        torch_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(torch_cache))

        import torch
        import cv2
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.blending import get_image_prepare_material
        from musetalk.utils.face_parsing import FaceParsing
        from musetalk.utils.preprocessing import get_landmark_and_bbox
        from musetalk.utils.utils import load_all_model
        from transformers import WhisperModel

        if not os.path.exists(self.cfg.base_image):
            raise FileNotFoundError(f"基准照片不存在: {self.cfg.base_image}")

        self._device = torch.device(self.cfg.device)
        vae, unet, pe = load_all_model(
            unet_model_path=self.cfg.unet_model_path,
            vae_type=self.cfg.vae_type,
            unet_config=self.cfg.unet_config,
            device=self._device,
        )
        self._vae, self._unet, self._pe = vae, unet, pe
        self._timesteps = torch.tensor([0], device=self._device)

        # 官方实时脚本中的半精度优化
        pe = pe.half().to(self._device)
        vae.vae = vae.vae.half().to(self._device)
        unet.model = unet.model.half().to(self._device)

        self._audio_processor = AudioProcessor(
            feature_extractor_path=self.cfg.whisper_dir
        )
        weight_dtype = unet.model.dtype
        whisper = WhisperModel.from_pretrained(self.cfg.whisper_dir)
        whisper = whisper.to(device=self._device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False)
        self._whisper = whisper

        if self.cfg.version == "v15":
            fp = FaceParsing(
                left_cheek_width=self.cfg.left_cheek_width,
                right_cheek_width=self.cfg.right_cheek_width,
            )
        else:
            fp = FaceParsing()

        # 基准照片：人脸框 + 256 裁剪 latents + 混合 mask（只计算一次）
        coords, frames = get_landmark_and_bbox([self.cfg.base_image], self.cfg.bbox_shift)
        self._bbox = tuple(int(v) for v in coords[0])
        self._ori_frame = frames[0]
        x1, y1, x2, y2 = self._bbox
        crop_frame = self._ori_frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)  # [1, 8, 32, 32]
        self._latents_cycle = [latents]
        mode = "jaw" if self.cfg.version == "v15" else "raw"
        self._mask, self._mask_crop_box = get_image_prepare_material(
            self._ori_frame, self._bbox, fp=fp, mode=mode
        )

    def start(self) -> None:
        """启动帧生成线程。"""
        if self._thread is not None and self._thread.is_alive():
            return
        if self._vae is None:
            self.load()
        self._running = True
        self._thread = threading.Thread(target=self._generate_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        try:
            self._audio_queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    def push_audio(self, pcm16_bytes: bytes) -> None:
        """输入 16k int16 PCM 音频块（由播放线程按播放顺序推送）。

        非阻塞：队列满时丢弃最旧的音频块，避免播放线程被 push 阻塞导致声音断续；
        被丢弃的音频仍计入已消费时间，保证帧时间轴连续。
        """
        if not pcm16_bytes:
            return
        try:
            self._audio_queue.put_nowait(pcm16_bytes)
        except queue.Full:
            try:
                dropped = self._audio_queue.get_nowait()
            except _QueueEmpty:
                dropped = None
            if dropped is not None and dropped is not _SENTINEL:
                self._get_and_add_consumed(len(dropped) / 2 / 16000)
            try:
                self._audio_queue.put_nowait(pcm16_bytes)
            except queue.Full:
                self._get_and_add_consumed(len(pcm16_bytes) / 2 / 16000)

    def get_frame(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """取出下一帧 BGR ndarray；队列空且超时返回 None。"""
        try:
            item = self._frame_queue.get(timeout=timeout)
        except _QueueEmpty:
            return None
        return None if item is _SENTINEL else item[1]

    def get_frame_with_timestamp(
        self, timeout: Optional[float] = None
    ) -> Optional[Tuple[float, np.ndarray]]:
        """取出 (预计播放时刻秒, BGR 帧)，渲染线程据此做音画对齐。"""
        try:
            item = self._frame_queue.get(timeout=timeout)
        except _QueueEmpty:
            return None
        return None if item is _SENTINEL else item

    def _put_frame(self, item: Tuple[float, np.ndarray]) -> bool:
        """写入帧队列；线程停止时立即返回，避免关闭时死锁。"""
        while self._running:
            try:
                self._frame_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _generate_loop(self) -> None:
        """帧生成线程：按块消费音频，逐块走官方推理链路。"""
        # 每个处理块固定为 chunk_seconds 秒（16k 采样），不足一块的音频先缓存
        target_samples = int(self.cfg.chunk_seconds * 16000)
        buffer = bytearray()
        while self._running:
            try:
                item = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                break
            buffer.extend(item)
            # 积满一块才处理，避免把 TTS 不定长片段直接喂给模型
            while len(buffer) >= target_samples:
                block = bytes(buffer[:target_samples])
                del buffer[:target_samples]
                try:
                    self._process_audio_block(block)
                except Exception as exc:  # noqa: BLE001
                    logger.error("MuseTalk 帧生成失败: %s", exc)
        # 处理残余不足一块的音频
        if buffer:
            try:
                self._process_audio_block(bytes(buffer))
            except Exception as exc:  # noqa: BLE001
                logger.error("MuseTalk 收尾帧生成失败: %s", exc)
        try:
            self._frame_queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass

    def _process_audio_block(self, pcm16_bytes: bytes) -> None:
        """处理一个音频块：特征提取 -> UNet 推理 -> VAE 解码 -> 混合 -> 帧队列。"""
        import cv2
        from musetalk.utils.utils import datagen

        sample_rate = 16000
        audio_np = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if audio_np.size == 0:
            return

        # 官方 get_audio_feature 需要 wav 文件路径
        wav_bytes = audio_to_wav_bytes(audio_np, sample_rate)
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(wav_bytes)
            weight_dtype = self._unet.model.dtype
            features, librosa_length = self._audio_processor.get_audio_feature(
                wav_path, weight_dtype=weight_dtype
            )
            if features is None or librosa_length == 0:
                return
            whisper_chunks = self._audio_processor.get_whisper_chunk(
                features,
                self._device,
                weight_dtype,
                self._whisper,
                librosa_length,
                fps=self.cfg.fps,
                audio_padding_length_left=self.cfg.audio_padding_length_left,
                audio_padding_length_right=self.cfg.audio_padding_length_right,
            )
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

        block_start = self._get_and_add_consumed(len(audio_np) / sample_rate)
        gen = datagen(
            whisper_chunks,
            self._latents_cycle,
            batch_size=self.cfg.batch_size,
            device=self.cfg.device,
        )
        frame_index = 0
        for whisper_batch, latent_batch in gen:
            audio_feature_batch = self._pe(whisper_batch.to(self._device))
            latent_batch = latent_batch.to(device=self._device, dtype=self._unet.model.dtype)
            pred_latents = self._unet.model(
                latent_batch,
                self._timesteps,
                encoder_hidden_states=audio_feature_batch,
            ).sample
            pred_latents = pred_latents.to(device=self._device, dtype=self._vae.vae.dtype)
            recon = self._vae.decode_latents(pred_latents)  # uint8 BGR [B, 256, 256, 3]
            for raw_frame in recon:
                timestamp = block_start + frame_index / self.cfg.fps
                frame_index += 1
                bgr_frame = self._post_process(raw_frame, cv2)
                if not self._put_frame((timestamp, bgr_frame)):
                    return

    def _post_process(self, res_frame: np.ndarray, cv2) -> np.ndarray:
        """按官方 process_frames 逻辑：resize 到人脸框尺寸并与原图混合。"""
        x1, y1, x2, y2 = self._bbox
        resized = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        from musetalk.utils.blending import get_image_blending

        return get_image_blending(
            self._ori_frame,
            resized,
            self._bbox,
            self._mask,
            self._mask_crop_box,
        )

    def _get_and_add_consumed(self, seconds: float) -> float:
        """返回该音频块在输出流中的起始时刻（秒）并累加已消费时长。

        帧时间戳以"已消费音频的累计秒数"为基准，与播放线程的播放进度对齐；
        丢弃的音频块也计入这里，保证时间轴连续。
        """
        with self._consumed_lock:
            start = self._consumed_seconds
            self._consumed_seconds += seconds
            return start

    def release(self) -> None:
        self.stop()
        self._vae = None
        self._unet = None
        self._pe = None
        self._whisper = None
        self._audio_processor = None
