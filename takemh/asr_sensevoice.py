"""ASR 模块：SenseVoice-small + fsmn-vad(DynamicStreamingVAD)，强制 CPU，麦克风实时语音转文字。"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, Optional

import numpy as np
import torch

from takemh.audio_io import AudioInputStream, pcm16_to_float32
from takemh.config import AsrConfig

logger = logging.getLogger(__name__)


class SenseVoiceASR:
    """SenseVoice 实时语音识别。

    线程循环从麦克风读取 16k 单声道 PCM，交给 FunASR 官方 DynamicStreamingVAD
    做流式端点检测；检测到完整语音段后从音频缓冲切出该段，交给 SenseVoice-small
    识别，并通过回调输出文本。

    回调：
        on_final(text)   - 一句话识别完成的最终文本
        on_interim(info) - 中间状态（{"state": "speech", "duration_ms": ...}）
    """

    def __init__(
        self,
        cfg: AsrConfig,
        on_final: Optional[Callable[[str], None]] = None,
        on_interim: Optional[Callable[[Dict[str, object]], None]] = None,
        audio_source: Optional[object] = None,
    ):
        self.cfg = cfg
        self.on_final = on_final
        self.on_interim = on_interim
        self._audio_source = audio_source

        self._asr_model = None
        self._vad_stream = None  # DynamicStreamingVAD（load() 后填充）
        self._mic: Optional[AudioInputStream] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # 音频缓冲与 VAD 段切分
        self._audio_buffer = np.array([], dtype=np.float32)
        self._buffer_start_ms = 0.0

    @property
    def sample_rate(self) -> int:
        return self.cfg.sample_rate

    def load(self) -> None:
        """加载 SenseVoice-small 与 fsmn-vad，并创建官方 DynamicStreamingVAD。"""
        from funasr import AutoModel
        from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD

        vad_model = AutoModel(
            model=self.cfg.vad_model,
            device=self.cfg.device,
            disable_update=True,
        )
        self._vad_stream = DynamicStreamingVAD(
            vad_model,
            chunk_size_ms=60,
            sample_rate=self.cfg.sample_rate,
            silence_schedule=[(float("inf"), self.cfg.vad_silence_ms)],
        )
        self._asr_model = AutoModel(
            model=self.cfg.model,
            trust_remote_code=True,
            device=self.cfg.device,  # 强制 CPU
            vad_model=self.cfg.vad_model,
            vad_kwargs={"max_single_segment_time": self.cfg.max_single_segment_ms},
            disable_update=True,
        )

    def start(self) -> None:
        """启动麦克风采集与识别子线程。"""
        if self._thread is not None and self._thread.is_alive():
            return
        if self._asr_model is None:
            self.load()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止采集线程；残留语音段会被识别并回调。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._flush_remaining()
        if self._mic is not None:
            self._mic.close()
            self._mic = None

    def _capture_loop(self) -> None:
        """采集线程：读取麦克风块并送入 VAD 流水线。"""
        if self._audio_source is not None:
            self._mic = self._audio_source
        else:
            self._mic = AudioInputStream(
                sample_rate=self.cfg.sample_rate,
                channels=self.cfg.channels,
                chunk_size=self.cfg.sample_rate * self.cfg.chunk_ms // 1000,
            )
        self._mic.open()
        while self._running:
            try:
                pcm = self._mic.read_chunk()
            except EOFError:
                self._running = False
                break
            except Exception as exc:  # noqa: BLE001 - 输入设备异常需要显式记录
                logger.exception("音频输入读取失败: %s", exc)
                break
            try:
                self.feed_pcm(pcm)
            except Exception as exc:  # noqa: BLE001 - 单块失败不应静默结束采集线程
                logger.error("ASR 处理音频块失败: %s", exc)
                break
        self._flush_remaining()

    def feed_pcm(self, pcm: bytes) -> None:
        """对外暴露的单块 PCM 输入（16k int16），供采集线程与测试调用。"""
        audio = pcm16_to_float32(pcm)
        self._feed_float32(audio)

    def _feed_float32(self, audio: np.ndarray) -> None:
        """把一块 float32 音频送入流式 VAD，并处理确认的语音段。"""
        if len(audio) == 0:
            return
        self._audio_buffer = np.concatenate([self._audio_buffer, audio])
        segments = self._vad_stream.feed(torch.from_numpy(audio))
        for start_ms, end_ms in segments:
            self._flush_segment_ms(start_ms, end_ms)
        # 裁剪缓冲，防止长时间运行内存无限增长
        max_buf = int(self.cfg.sample_rate * self.cfg.max_buffer_seconds)
        if len(self._audio_buffer) > max_buf:
            drop = len(self._audio_buffer) - max_buf
            self._buffer_start_ms += drop / self.cfg.sample_rate * 1000.0
            self._audio_buffer = self._audio_buffer[drop:]

    def _flush_remaining(self) -> None:
        """流结束：让 VAD 输出剩余语音段。

        DynamicStreamingVAD.finalize() 内部只传入 10ms 空块，在 fsmn-vad
        处理进行中的语音段时可能触发 `stack expects a non-empty TensorList`；
        这里改传一个完整静音块并标记 is_final=True，让 VAD 正常结束当前段。
        """
        if self._vad_stream is None:
            return
        try:
            silence = torch.zeros(
                self.cfg.sample_rate * self.cfg.chunk_ms // 1000,
                dtype=torch.float32,
            )
            segments = self._vad_stream.feed(silence, is_final=True)
        except Exception as exc:  # noqa: BLE001
            logger.error("VAD 收尾失败: %s", exc)
            return
        for start_ms, end_ms in segments:
            self._flush_segment_ms(start_ms, end_ms)

    def _flush_segment_ms(self, start_ms: int, end_ms: int) -> None:
        """从音频缓冲切出 VAD 确认的语音段并识别（统一换算为毫秒）。"""
        unit_ms = max(1, int(self.cfg.vad_time_unit_ms))
        start_ms = int(start_ms * unit_ms)
        end_ms = int(end_ms * unit_ms)
        sr = self.cfg.sample_rate
        buf_start_s = self._buffer_start_ms / 1000.0
        rel_start = max(0, int((start_ms / 1000.0 - buf_start_s) * sr))
        rel_end = min(len(self._audio_buffer), int((end_ms / 1000.0 - buf_start_s) * sr))
        if rel_end - rel_start < int(sr * 0.2):  # 不足 200ms 视为噪声，忽略
            self._audio_buffer = self._audio_buffer[rel_end:]
            self._buffer_start_ms = end_ms
            return
        # 先切出语音段，再释放已消费的缓冲（顺序不能颠倒）
        segment = self._audio_buffer[rel_start:rel_end]
        self._audio_buffer = self._audio_buffer[rel_end:]
        self._buffer_start_ms = end_ms
        text = self._recognize(segment)
        if not text:
            return
        self._emit_interim("speech", end_ms - start_ms)
        if self.on_final:
            self.on_final(text)

    def _recognize(self, audio: np.ndarray) -> str:
        """调用官方 SenseVoice 接口识别一段语音。"""
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        res = self._asr_model.generate(
            input=audio,
            cache={},
            language=self.cfg.language,
            use_itn=self.cfg.use_itn,
            batch_size_s=self.cfg.batch_size_s,
            merge_vad=self.cfg.merge_vad,
            merge_length_s=self.cfg.merge_length_s,
        )
        if not res or "text" not in res[0]:
            return ""
        return rich_transcription_postprocess(res[0]["text"]).strip()

    def _emit_interim(self, state: str, duration_ms: int = 0) -> None:
        if self.on_interim:
            self.on_interim({"state": state, "duration_ms": int(duration_ms)})

    def release(self) -> None:
        """释放模型引用（模型资源由 FunASR 管理，这里仅清引用）。"""
        self.stop()
        self._asr_model = None
        self._vad_stream = None
