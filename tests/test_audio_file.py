"""WAV 文件输入流测试：验证读取与资源关闭，不依赖音频设备。"""

from __future__ import annotations

import wave

from takemh.audio_file import WavFileInputStream


def test_wav_file_input_stream_reads_and_closes(tmp_path):
    path = tmp_path / "input.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)

    stream = WavFileInputStream(str(path), chunk_size=800)
    stream.open()
    try:
        assert len(stream.read_chunk()) == 1600
        assert len(stream.read_chunk()) == 1600
    finally:
        stream.close()

    assert stream._reader is None
