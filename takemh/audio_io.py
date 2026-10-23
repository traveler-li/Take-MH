"""音频输入输出与 PCM 工具：麦克风采集、格式转换、重采样。"""

from __future__ import annotations

import io
import wave
from typing import Optional

import numpy as np


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """int16 PCM bytes -> float32 numpy（范围 [-1, 1]）。"""
    data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return data / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """float32 numpy -> int16 PCM bytes，供 pyaudio 播放。"""
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16).tobytes()


def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """使用 scipy 将音频重采样到目标采样率，输入输出均为 float32。"""
    if src_rate == dst_rate:
        return audio
    from scipy.signal import resample_poly

    gcd = int(np.gcd(src_rate, dst_rate))
    up = dst_rate // gcd
    down = src_rate // gcd
    return resample_poly(audio, up=up, down=down).astype(np.float32)


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """float32 numpy -> WAV 文件字节（16bit PCM）。"""
    buf = io.BytesIO()
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class AudioInputStream:
    """基于 PyAudio 的麦克风输入流封装。

    提供 open/read_chunk/close 原语，线程循环由调用方（如 SenseVoiceASR）负责。
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 5120,
        device_index: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index
        self._pa = None
        self._stream = None

    def open(self) -> None:
        import pyaudio

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.device_index,
        )

    def read_chunk(self) -> bytes:
        """阻塞读取一块 int16 PCM bytes。"""
        if self._stream is None:
            raise RuntimeError("AudioInputStream 尚未 open()")
        return self._stream.read(self.chunk_size, exception_on_overflow=False)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

