"""CosyVoiceTTS 最小独立测试：mock CosyVoice2，验证 16k PCM 输出与参考音频校验。"""

from __future__ import annotations

import types

import numpy as np
import pytest

from takemh.config import TtsConfig
from takemh.tts_cosyvoice import CosyVoiceTTS


class FakeSpeech:
    """模拟 torch.Tensor 的最小对象。"""

    def __init__(self, arr):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self._arr

    def reshape(self, *shape):
        return self._arr.reshape(*shape)


class FakeFrontend:
    """模拟官方 CosyVoice frontend 的关键方法。"""

    def text_normalize(self, text, split=True, text_frontend=True):
        parts = [s for s in text.split("。") if s.strip()]
        return parts or [text]

    def _extract_text_token(self, text):
        return ("tok", np.array([1], dtype=np.int32))

    def _extract_speech_feat(self, wav):
        return (np.zeros((1, 8, 80), dtype=np.float32), np.array([8], dtype=np.int32))

    def _extract_speech_token(self, wav):
        return (np.zeros((1, 4), dtype=np.int32), np.array([4], dtype=np.int32))

    def _extract_spk_embedding(self, wav):
        return np.zeros((1, 192), dtype=np.float32)


class FakeModel:
    """模拟 CosyVoice model.tts 流式输出。"""

    def tts(self, **kwargs):
        assert kwargs.get("stream") is True
        speech = np.ones((1, 2400), dtype=np.float32) * 0.1  # 0.1s @ 24k
        yield {"tts_speech": FakeSpeech(speech)}


class FakeEngine:
    sample_rate = 24000
    frontend = FakeFrontend()
    model = FakeModel()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_wav, stream):
        assert prompt_wav is not None
        assert stream is True
        speech = np.ones((1, 2400), dtype=np.float32) * 0.1  # 0.1s @ 24k
        yield {"tts_speech": FakeSpeech(speech)}


def test_push_text_produces_16k_pcm():
    cfg = TtsConfig(
        prompt_wav="assets/prompt.wav",
        prompt_text="参考文本",
        output_sample_rate=16000,
        inter_sentence_silence_seconds=0.0,
    )
    tts = CosyVoiceTTS(cfg)
    tts._engine = FakeEngine()
    tts._model_sample_rate = 24000
    tts._prompt_wav = "assets/prompt.wav"  # 路径字符串，FakeFrontend 不真正读取

    tts.start()
    tts.push_text("你好。")
    chunk = tts.get_audio_chunk(timeout=15.0)
    tts.stop()

    assert chunk is not None and len(chunk) > 0
    # 0.1s @24k 重采样到 16k -> 1600 samples -> 3200 bytes int16
    assert len(chunk) == 1600 * 2


def test_prompt_audio_short_raises(monkeypatch):
    fake_librosa = types.ModuleType("librosa")
    fake_librosa.load = lambda path, sr: (
        np.zeros(int(sr * 1.0), dtype=np.float32),
        sr,
    )  # 仅 1 秒
    monkeypatch.setitem(__import__("sys").modules, "librosa", fake_librosa)

    cfg = TtsConfig(
        prompt_wav="assets/prompt.wav",
        prompt_text="参考文本",
        min_prompt_seconds=3.0,
        max_prompt_seconds=10.0,
    )
    tts = CosyVoiceTTS(cfg)
    with pytest.raises(ValueError, match="参考音频时长"):
        tts._check_prompt_audio()

