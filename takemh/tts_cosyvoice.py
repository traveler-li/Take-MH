"""TTS 模块：CosyVoice 2 0.5B 官方 SDK 封装，流式文本合成 16k PCM 音频流。"""

from __future__ import annotations

import logging
from pathlib import Path
import queue
import threading
from typing import Optional

import numpy as np

from takemh.audio_io import float32_to_pcm16, resample_audio
from takemh.config import TtsConfig

logger = logging.getLogger(__name__)

_SENTINEL = object()


class CosyVoiceTTS:
    """CosyVoice2 流式 TTS。

    使用官方 CosyVoice2.inference_zero_shot 进行零样本音色克隆（3~10 秒参考音频），
    流式输出每个句子的音频片段，统一重采样为 16k int16 PCM 后放入音频队列。

    - push_text(text): 推入文本，内部按句合成
    - get_audio_chunk(timeout): 取出 PCM bytes
    """

    def __init__(self, cfg: TtsConfig):
        self.cfg = cfg
        self._engine = None
        self._prompt_wav: Optional[np.ndarray] = None
        self._text_queue: "queue.Queue" = queue.Queue(maxsize=cfg.text_queue_size)
        self._audio_queue: "queue.Queue" = queue.Queue(maxsize=cfg.audio_queue_size)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._prompt_features: Optional[dict] = None

    def load(self) -> None:
        """校验参考音频并加载 CosyVoice2 模型（官方接口）。"""
        # 确保 CosyVoice 仓库路径在 sys.path 中
        _repo_root = Path(__file__).resolve().parents[1]
        _cosyvoice = _repo_root.parent / "CosyVoice"
        import sys
        if _cosyvoice.is_dir() and str(_cosyvoice) not in sys.path:
            sys.path.insert(0, str(_cosyvoice))

        self._check_prompt_audio()
        from cosyvoice.cli.cosyvoice import CosyVoice2

        self._engine = CosyVoice2(
            self.cfg.model_dir,
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=self.cfg.fp16,
        )
        self._model_sample_rate = int(self._engine.sample_rate)

    def _check_prompt_audio(self) -> None:
        """参考音频时长必须落在 3~10 秒区间。

        CosyVoice 官方 inference_zero_shot 的 prompt_wav 接受文件路径或
        torch.Tensor（见官方 example.py / prepare_reject_sample.py）；这里保留
        文件路径字符串，由官方 load_wav 内部加载，避免传入 numpy/Tensor 时
        torchaudio.load 报 Invalid file。
        """
        import librosa

        wav, _ = librosa.load(self.cfg.prompt_wav, sr=16000)
        duration = len(wav) / 16000.0
        if not (self.cfg.min_prompt_seconds <= duration <= self.cfg.max_prompt_seconds):
            raise ValueError(
                f"参考音频时长 {duration:.2f}s 超出 [{self.cfg.min_prompt_seconds}, "
                f"{self.cfg.max_prompt_seconds}]s，请更换 3~10 秒的参考音频"
            )
        self._prompt_wav = self.cfg.prompt_wav

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if self._engine is None:
            self.load()
        self._running = True
        self._thread = threading.Thread(target=self._synthesis_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止合成线程并清空队列。"""
        self._running = False
        try:
            self._text_queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        self._drain(self._text_queue)
        self._drain(self._audio_queue)

    @staticmethod
    def _drain(q: "queue.Queue") -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def push_text(self, text: str, timeout: float = 1.0) -> bool:
        """把一段完整文本推入合成队列；超时或线程停止时返回 False。"""
        if not (text or "").strip():
            return False
        try:
            self._text_queue.put(text.strip(), timeout=timeout)
        except queue.Full:
            logger.warning("TTS 文本队列已满，丢弃文本: %s", text.strip())
            return False
        return True

    def get_audio_chunk(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """取出下一块 16k int16 PCM；超时返回 None。"""
        try:
            item = self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        return None if item is _SENTINEL else item

    def _synthesis_loop(self) -> None:
        """合成线程：逐句调用官方推理接口并产出 PCM。"""
        while self._running:
            try:
                text = self._text_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if text is _SENTINEL:
                break
            try:
                # 逐句合成：每句输出多个 tts_speech 片段，随后补一句间静音。
                # 音频队列写入带停止检查，避免关闭时永久卡在 put()。
                for chunk in self._synthesize_sentence(text):
                    if chunk and not self._put_audio(chunk):
                        break
                else:
                    if self._put_silence():
                        continue
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("TTS 合成失败: %s", exc)
        # 通知播放侧结束（_SENTINEL 标记流终止）
        try:
            self._audio_queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass

    def _get_prompt_features(self) -> dict:
        """提取并缓存 prompt 特征（官方 frontend 方法）。

        官方 inference_zero_shot 对每个句子都重新提取 prompt 的文本 token、
        speech token / feat / 音色 embedding，这些对同一基准音频完全相同；
        这里只提取一次并缓存，多句回复时每句可省 1~2 秒。
        """
        if self._prompt_features is not None:
            return dict(self._prompt_features)
        frontend = self._engine.frontend
        prompt_text_token, prompt_text_token_len = frontend._extract_text_token(
            self.cfg.prompt_text
        )
        speech_feat, speech_feat_len = frontend._extract_speech_feat(self._prompt_wav)
        speech_token, speech_token_len = frontend._extract_speech_token(self._prompt_wav)
        if self._model_sample_rate == 24000:
            # 与官方 frontend_zero_shot 一致：force speech_feat % speech_token = 2
            token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat = speech_feat[:, : 2 * token_len]
            speech_feat_len[:] = 2 * token_len
            speech_token = speech_token[:, :token_len]
            speech_token_len[:] = token_len
        embedding = frontend._extract_spk_embedding(self._prompt_wav)
        self._prompt_features = {
            "prompt_text": prompt_text_token,
            "prompt_text_len": prompt_text_token_len,
            "llm_prompt_speech_token": speech_token,
            "llm_prompt_speech_token_len": speech_token_len,
            "flow_prompt_speech_token": speech_token,
            "flow_prompt_speech_token_len": speech_token_len,
            "prompt_speech_feat": speech_feat,
            "prompt_speech_feat_len": speech_feat_len,
            "llm_embedding": embedding,
            "flow_embedding": embedding,
        }
        return dict(self._prompt_features)

    def _synthesize_sentence(self, text: str):
        """按官方 inference_zero_shot 链路合成，但 prompt 特征只提取一次。"""
        frontend = self._engine.frontend
        prompt_features = self._get_prompt_features()
        for sentence in frontend.text_normalize(text, split=True, text_frontend=True):
            text_token, text_token_len = frontend._extract_text_token(sentence)
            model_input = dict(prompt_features)
            model_input["text"] = text_token
            model_input["text_len"] = text_token_len
            for model_output in self._engine.model.tts(
                **model_input, stream=self.cfg.stream, speed=1.0
            ):
                speech = model_output["tts_speech"]  # torch.Tensor [1, N]
                yield self._to_pcm16(speech)

    def _to_pcm16(self, speech) -> bytes:
        """模型输出 -> 16k int16 PCM bytes（带响度归一化）。

        参考音频或模型输出音量可能偏低（如 -29dBFS），统一把 RMS 提升到
        约 -16.5dBFS，让播放更清晰；音量本已足够的音频不会被压低。
        """
        audio = speech.detach().cpu().float().numpy().reshape(-1)
        audio = resample_audio(audio, self._model_sample_rate, self.cfg.output_sample_rate)
        target_rms = 0.15  # 约 -16.5 dBFS
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms > 1e-6 and rms < target_rms:
            audio = audio * (target_rms / rms)
            audio = np.clip(audio, -0.99, 0.99)
        return float32_to_pcm16(audio)

    def _put_audio(self, chunk: bytes) -> bool:
        """写入音频队列；线程停止时立即返回，避免阻塞关闭流程。"""
        while self._running:
            try:
                self._audio_queue.put(chunk, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _put_silence(self) -> bool:
        """句间静音，避免句子拼接生硬。"""
        silence_len = int(self.cfg.output_sample_rate * self.cfg.inter_sentence_silence_seconds)
        if silence_len <= 0:
            return True
        return self._put_audio(
            float32_to_pcm16(np.zeros(silence_len, dtype=np.float32))
        )

    def release(self) -> None:
        self.stop()
        self._engine = None
