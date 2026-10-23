"""SenseVoiceASR 最小独立测试：mock FunASR 与麦克风，验证 VAD 段切分与识别回调。"""

from __future__ import annotations

import types

import numpy as np

from takemh.asr_sensevoice import SenseVoiceASR
from takemh.config import AsrConfig


class FakeVadStream:
    """模拟官方 DynamicStreamingVAD：按调用顺序返回确认段。"""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def feed(self, audio_chunk, is_final=False):
        self.calls.append(audio_chunk)
        return self._results.pop(0) if self._results else []

    def finalize(self):
        return self._results.pop(0) if self._results else []


class FakeAsr:
    def __init__(self, text="<|zh|>你好"):
        self.text = text
        self.inputs = []

    def generate(self, **kwargs):
        self.inputs.append(kwargs)
        return [{"text": self.text}]


def _install_fake_funasr(monkeypatch):
    """向 sys.modules 注入最小 funasr 模块树，供延迟导入使用。"""
    sys_mods = __import__("sys").modules
    fake_funasr = types.ModuleType("funasr")
    fake_utils = types.ModuleType("funasr.utils")
    fake_pp = types.ModuleType("funasr.utils.postprocess_utils")
    fake_pp.rich_transcription_postprocess = lambda text: text.split(">")[-1].strip()
    fake_utils.postprocess_utils = fake_pp
    fake_funasr.utils = fake_utils

    fake_models = types.ModuleType("funasr.models")
    fake_stream = types.ModuleType("funasr.models.fsmn_vad_streaming")
    fake_dvad = types.ModuleType("funasr.models.fsmn_vad_streaming.dynamic_vad")
    fake_dvad.DynamicStreamingVAD = FakeVadStream
    fake_stream.dynamic_vad = fake_dvad
    fake_models.fsmn_vad_streaming = fake_stream
    fake_funasr.models = fake_models

    monkeypatch.setitem(sys_mods, "funasr", fake_funasr)
    monkeypatch.setitem(sys_mods, "funasr.utils", fake_utils)
    monkeypatch.setitem(sys_mods, "funasr.utils.postprocess_utils", fake_pp)
    monkeypatch.setitem(sys_mods, "funasr.models", fake_models)
    monkeypatch.setitem(sys_mods, "funasr.models.fsmn_vad_streaming", fake_stream)
    monkeypatch.setitem(
        sys_mods, "funasr.models.fsmn_vad_streaming.dynamic_vad", fake_dvad
    )


def test_vad_segment_triggers_final_callback(monkeypatch):
    _install_fake_funasr(monkeypatch)
    finals = []
    interim = []
    cfg = AsrConfig(sample_rate=16000, chunk_ms=320, vad_silence_ms=600)
    asr = SenseVoiceASR(cfg, on_final=finals.append, on_interim=interim.append)
    # 前 3 块无确认，第 4 块（缓冲达 2 秒）才确认 500~1500ms 段
    asr._vad_stream = FakeVadStream(results=[[], [], [], [[500, 1500]], []])
    asr._asr_model = FakeAsr(text="<|zh|>你好")

    # 喂入 4 块 320ms = 2 秒音频，覆盖 500~1500ms 语音段
    chunk = np.zeros(cfg.sample_rate * cfg.chunk_ms // 1000, dtype=np.float32)
    for _ in range(4):
        asr._feed_float32(chunk)

    assert finals == ["你好"], f"最终回调应为识别文本，实际 {finals}"
    assert interim, "应产生中间状态"
    assert asr._audio_buffer.size == 0, "段切出后缓冲应被释放"


def test_short_segment_ignored(monkeypatch):
    _install_fake_funasr(monkeypatch)
    finals = []
    cfg = AsrConfig(sample_rate=16000, chunk_ms=320, vad_silence_ms=600)
    asr = SenseVoiceASR(cfg, on_final=finals.append)
    asr._vad_stream = FakeVadStream(results=[[[0, 100]], []])  # 100ms 过短
    asr._asr_model = FakeAsr()
    asr._feed_float32(np.zeros(5120, dtype=np.float32))
    assert finals == []


def test_finalize_flushes_remaining(monkeypatch):
    _install_fake_funasr(monkeypatch)
    finals = []
    cfg = AsrConfig(sample_rate=16000, chunk_ms=320, vad_silence_ms=600)
    asr = SenseVoiceASR(cfg, on_final=finals.append)
    asr._vad_stream = FakeVadStream(results=[[], [[0, 800]]])  # feed 无段，finalize 给一段
    asr._asr_model = FakeAsr(text="<|zh|>结束")
    asr._feed_float32(np.zeros(5120, dtype=np.float32))
    asr._flush_remaining()
    assert finals == ["结束"]


def test_recognize_uses_official_generate_params(monkeypatch):
    _install_fake_funasr(monkeypatch)
    cfg = AsrConfig(sample_rate=16000)
    asr = SenseVoiceASR(cfg)
    asr._asr_model = FakeAsr(text="<|zh|>你好")
    text = asr._recognize(np.zeros(16000, dtype=np.float32))
    assert text == "你好"
    kwargs = asr._asr_model.inputs[0]
    assert kwargs["language"] == cfg.language
    assert kwargs["use_itn"] is True
    assert kwargs["cache"] == {}


def test_vad_timestamp_unit_is_applied(monkeypatch):
    _install_fake_funasr(monkeypatch)
    finals = []
    cfg = AsrConfig(
        sample_rate=16000,
        chunk_ms=320,
        vad_silence_ms=600,
        vad_time_unit_ms=10,
    )
    asr = SenseVoiceASR(cfg, on_final=finals.append)
    asr._vad_stream = FakeVadStream(results=[[], [], [], [[50, 150]], []])
    asr._asr_model = FakeAsr(text="<|zh|>你好")

    chunk = np.zeros(cfg.sample_rate * cfg.chunk_ms // 1000, dtype=np.float32)
    for _ in range(4):
        asr._feed_float32(chunk)

    assert finals == ["你好"]
    assert asr._buffer_start_ms == 1500
