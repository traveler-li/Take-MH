"""MuseTalkDriver 最小独立测试：mock 官方接口，验证音频块 -> BGR 帧。"""

from __future__ import annotations

import types

import numpy as np

from takemh.config import MuseTalkConfig
from takemh.musetalk_driver import MuseTalkDriver


class FakeTensor:
    dtype = "half"

    def __init__(self, shape=(1, 8, 32, 32)):
        self.shape = shape

    def to(self, *args, **kwargs):
        return self


class FakePE:
    def __call__(self, batch):
        return FakeTensor((1, 10, 5, 384))


class FakeUnetModel:
    dtype = "half"

    def __call__(self, latent_batch, timesteps, encoder_hidden_states):
        return types.SimpleNamespace(sample=FakeTensor((1, 4, 32, 32)))


class FakeVAE:
    vae = FakeTensor()

    def decode_latents(self, pred):
        return np.zeros((1, 256, 256, 3), dtype=np.uint8)


class FakeAudioProcessor:
    def get_audio_feature(self, wav_path, weight_dtype=None):
        return [FakeTensor((1, 80, 3000))], 16000

    def get_whisper_chunk(self, features, device, weight_dtype, whisper, length, **kwargs):
        return [FakeTensor((10, 5, 384))]


def _install_fakes(monkeypatch):
    """注入 fake musetalk 模块树与 cv2，避免真实依赖。"""
    sys = __import__("sys")
    fake_mt = types.ModuleType("musetalk")
    fake_utils = types.ModuleType("musetalk.utils")
    fake_utils_mod = types.ModuleType("musetalk.utils.utils")
    fake_blend = types.ModuleType("musetalk.utils.blending")

    def fake_datagen(whisper_chunks, vae_encode_latents, batch_size, device):
        yield FakeTensor((1, 10, 5, 384)), FakeTensor((1, 8, 32, 32))

    def fake_blending(image, face, face_box, mask_array, crop_box):
        return face

    fake_utils_mod.datagen = fake_datagen
    fake_blend.get_image_blending = fake_blending
    fake_utils.utils = fake_utils_mod
    fake_utils.blending = fake_blend
    fake_mt.utils = fake_utils
    monkeypatch.setitem(sys.modules, "musetalk", fake_mt)
    monkeypatch.setitem(sys.modules, "musetalk.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "musetalk.utils.utils", fake_utils_mod)
    monkeypatch.setitem(sys.modules, "musetalk.utils.blending", fake_blend)

    class FakeCv2:
        @staticmethod
        def resize(src, dsize, interpolation=None):
            h, w = dsize
            return np.zeros((h, w, 3), dtype=np.uint8)

    monkeypatch.setitem(sys.modules, "cv2", FakeCv2())


def _driver():
    cfg = MuseTalkConfig(
        fps=25,
        batch_size=2,
        chunk_seconds=1.0,
        frame_queue_size=64,
    )
    driver = MuseTalkDriver(cfg)
    driver._running = True
    driver._device = "cuda:0"
    driver._timesteps = "ts"
    driver._bbox = (10, 20, 90, 120)
    driver._ori_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    driver._mask = np.zeros((200, 200), dtype=np.uint8)
    driver._mask_crop_box = (0, 0, 200, 200)
    driver._latents_cycle = [FakeTensor()]
    driver._pe = FakePE()
    driver._unet = types.SimpleNamespace(model=FakeUnetModel(), dtype="half")
    driver._vae = FakeVAE()
    driver._audio_processor = FakeAudioProcessor()
    driver._whisper = object()
    return driver


def test_process_audio_block_produces_bgr_frame(monkeypatch):
    _install_fakes(monkeypatch)
    driver = _driver()
    one_second_pcm = bytes(16000 * 2)  # 1 秒 16k int16 静音
    driver._process_audio_block(one_second_pcm)
    frame = driver.get_frame(timeout=1.0)
    assert frame is not None
    assert frame.ndim == 3
    assert frame.shape[2] == 3  # BGR


def test_audio_queue_and_consumed_tracking(monkeypatch):
    _install_fakes(monkeypatch)
    driver = _driver()
    start = driver._get_and_add_consumed(1.0)
    assert start == 0.0
    start = driver._get_and_add_consumed(0.5)
    assert start == 1.0
    with driver._consumed_lock:
        assert driver._consumed_seconds == 1.5
