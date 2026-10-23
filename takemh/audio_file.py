"""文件音频输入流：从 WAV 文件逐块读取 16k 单声道 PCM。"""

from __future__ import annotations

from typing import Optional

import numpy as np
import soundfile as sf


class WavFileInputStream:
    """模拟麦克风输入接口，从 WAV 文件读取 16k 单声道 int16 PCM。

    非 16k 采样率的单声道 WAV 会被自动重采样到 16k；多声道文件不支持。
    """

    def __init__(self, path: str, chunk_size: int = 5120):
        self.path = path
        self.chunk_size = chunk_size
        self._reader: Optional[sf.SoundFile] = None
        self._sample_rate: Optional[int] = None
        # 非 16k 时用 float32 缓冲重采样
        self._float_buffer: Optional[np.ndarray] = None
        self._float_pos = 0

    def open(self) -> None:
        self._reader = sf.SoundFile(self.path, mode="r")
        self._sample_rate = self._reader.samplerate
        if self._reader.channels != 1:
            raise ValueError(f"需要单声道 WAV，实际 {self._reader.channels} 声道")
        if self._sample_rate != 16000:
            # 用 scipy 重采样（librosa 首次调用会触发 numba JIT 编译，可能卡数十秒）
            from scipy.signal import resample_poly

            data = self._reader.read(dtype="float32")
            self._reader.close()
            self._reader = None
            gcd = int(np.gcd(self._sample_rate, 16000))
            up = 16000 // gcd
            down = self._sample_rate // gcd
            self._float_buffer = resample_poly(data, up=up, down=down).astype(np.float32)
            self._float_pos = 0

    def read_chunk(self) -> bytes:
        if self._float_buffer is not None:
            chunk = self._float_buffer[self._float_pos:self._float_pos + self.chunk_size]
            self._float_pos += len(chunk)
            if len(chunk) == 0:
                raise EOFError("WAV 文件读取完毕")
            return (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        if self._reader is None:
            raise RuntimeError("WavFileInputStream 尚未 open()")
        data = self._reader.read(self.chunk_size, dtype="int16")
        if len(data) == 0:
            raise EOFError("WAV 文件读取完毕")
        return data.tobytes()

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self._float_buffer = None
