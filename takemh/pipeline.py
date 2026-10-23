"""主程序：VirtualAvatarApp，多线程流式流水线串联 ASR -> LLM -> TTS -> MuseTalk。"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Optional

import numpy as np

from takemh.asr_sensevoice import SenseVoiceASR
from takemh.audio_file import WavFileInputStream
from takemh.config import Config
from takemh.llm_openai import LLMApiClient
from takemh.musetalk_driver import MuseTalkDriver
from takemh.text_splitting import SentenceSplitter
from takemh.tts_cosyvoice import CosyVoiceTTS

logger = logging.getLogger(__name__)


class VirtualAvatarApp:
    """实时照片虚拟人对话应用。

    线程布局：
      - ASR 线程（SenseVoiceASR 内部）：麦克风 -> 文字
      - LLM 线程：文字 -> 流式 token -> 句子切分 -> TTS
      - TTS 线程（CosyVoiceTTS 内部）：句子 -> 16k PCM
      - 播放线程：PCM 播放（pyaudio），同时按顺序推送给 MuseTalk
      - MuseTalk 线程（MuseTalkDriver 内部）：音频块 -> BGR 帧
      - 主线程：按音频进度渲染帧，实现音画同步
    """

    def __init__(self, cfg: Config, test_audio_path: Optional[str] = None):
        self.cfg = cfg
        self._test_audio_path = test_audio_path
        # ASR 识别文本队列：ASR 线程写入，LLM 线程消费
        self._user_text_queue: "queue.Queue" = queue.Queue(maxsize=16)
        self._splitter = SentenceSplitter()
        # 已播放音频的累计秒数（播放线程维护，主线程读取做音画对齐）
        self._played_seconds = 0.0
        self._played_lock = threading.Lock()
        self._last_lag_log_ts = 0.0  # "渲染落后帧"日志限流时间戳
        self._status = "Loading models..."  # 窗口标题状态（英文，避免乱码）
        self._status_lock = threading.Lock()

        asr_kwargs = {}
        if test_audio_path:
            chunk_size = cfg.asr.sample_rate * cfg.asr.chunk_ms // 1000
            asr_kwargs["audio_source"] = WavFileInputStream(
                path=test_audio_path, chunk_size=chunk_size
            )

        self.asr = SenseVoiceASR(
            cfg.asr,
            on_final=self._on_asr_final,
            on_interim=self._on_asr_interim,
            **asr_kwargs,
        )
        self.llm = LLMApiClient(cfg.llm)
        self.tts = CosyVoiceTTS(cfg.tts)
        self.musetalk = MuseTalkDriver(cfg.musetalk)

        self._llm_thread: Optional[threading.Thread] = None
        self._play_thread: Optional[threading.Thread] = None
        self._running = False

    # ---------- 启动与退出 ----------

    def start(self) -> None:
        """加载模型并启动全部线程。"""
        if self._test_audio_path and not os.path.exists(self._test_audio_path):
            raise FileNotFoundError(f"测试音频文件不存在: {self._test_audio_path}")
        self._running = True
        # 按"先占用 GPU 的 MuseTalk -> TTS -> 纯 CPU 的 ASR -> 轻量 LLM 客户端"顺序加载，
        # 避免多个模型同时初始化造成显存/内存峰值叠加
        logger.info("加载 MuseTalk 模型（首次较慢）...")
        self.musetalk.load()
        self.musetalk.start()
        logger.info("加载 CosyVoice 模型...")
        self.tts.load()
        self.tts.start()
        logger.info("加载 SenseVoice 模型...")
        self.asr.load()
        self.asr.start()
        self.llm.load()

        self._llm_thread = threading.Thread(target=self._llm_loop, daemon=True)
        self._llm_thread.start()
        self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self._play_thread.start()
        logger.info("VirtualAvatarApp 已启动，对话中...")
        self._set_status("Waiting for speech...")
        if self._test_audio_path:
            logger.info("测试模式：从 %s 读取音频", self._test_audio_path)

    def _set_status(self, text: str) -> None:
        """更新窗口标题状态（线程安全），供用户感知系统当前阶段。"""
        with self._status_lock:
            self._status = text

    def run(self) -> None:
        """主线程渲染循环；按 Esc / q / 窗口关闭退出。"""
        import cv2

        # 先显示基准照片作为启动画面，让用户立即看到窗口
        splash = cv2.imread(self.cfg.musetalk.base_image)
        if splash is not None:
            scale = self.cfg.app.frame_display_scale
            if abs(scale - 1.0) > 1e-6:
                splash = cv2.resize(
                    splash,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_LINEAR,
                )
            cv2.imshow("Virtual Avatar", splash)
            cv2.setWindowTitle("Virtual Avatar", "Take-MH - " + self._status)
            cv2.waitKey(1)

        try:
            while self._running:
                item = self.musetalk.get_frame_with_timestamp(timeout=0.1)
                if item is None:
                    # 无帧时继续处理窗口事件，保持窗口响应
                    key = cv2.waitKey(10) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if cv2.getWindowProperty("Virtual Avatar", cv2.WND_PROP_VISIBLE) < 1:
                        break
                    continue
                ts, frame = item
                with self._played_lock:
                    played = self._played_seconds
                # 跳过队列中明显落后于播放进度的旧帧，优先渲染接近当前播放进度的帧；
                # 若队列已空则渲染最新可用帧，保证画面持续更新（MuseTalk 慢于实时时不会静止）
                while ts < played - 0.15:
                    newer = self.musetalk.get_frame_with_timestamp(timeout=0)
                    if newer is None:
                        now = time.time()
                        if now - self._last_lag_log_ts >= 1.0:
                            logger.debug(
                                "渲染落后帧 ts=%.2f played=%.2f（MuseTalk 滞后）", ts, played
                            )
                            self._last_lag_log_ts = now
                        break
                    item = newer
                    ts, frame = item
                scale = self.cfg.app.frame_display_scale
                if abs(scale - 1.0) > 1e-6:
                    frame = cv2.resize(
                        frame,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_LINEAR,
                    )
                cv2.imshow("Virtual Avatar", frame)
                cv2.setWindowTitle("Virtual Avatar", "Take-MH - " + self._status)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if cv2.getWindowProperty("Virtual Avatar", cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.stop()

    def stop(self) -> None:
        """停止所有线程并释放资源。"""
        if not self._running:
            return
        self._running = False

        self.asr.stop()
        self._stop_thread(self._llm_thread)
        self._llm_thread = None
        self.tts.stop()
        self._stop_thread(self._play_thread)
        self._play_thread = None
        self.musetalk.stop()

        import cv2

        cv2.destroyAllWindows()
        logger.info("VirtualAvatarApp 已退出")

    @staticmethod
    def _stop_thread(thread: Optional[threading.Thread]) -> None:
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

    def release(self) -> None:
        """彻底释放模型与设备资源。"""
        self.stop()
        self.asr.release()
        self.llm.release()
        self.tts.release()
        self.musetalk.release()

    # ---------- 模块间回调 ----------

    def _on_asr_final(self, text: str) -> None:
        if not text:
            return
        logger.info("[ASR] %s", text)
        self._set_status("Replying...")
        try:
            self._user_text_queue.put_nowait(text)
        except queue.Full:
            logger.warning("用户输入队列已满，丢弃: %s", text)

    def _on_asr_interim(self, info) -> None:
        if self.cfg.app.debug:
            logger.info("[ASR] 中间状态: %s", info)

    def _llm_loop(self) -> None:
        """LLM 线程：消费用户文本，流式输出并切句送入 TTS。"""
        while self._running:
            try:
                user_text = self._user_text_queue.get(timeout=0.2)
            except queue.Empty:
                self._set_status("Waiting for speech...")
                continue
            self._set_status("Replying...")
            self._splitter.reset()
            for token in self.llm.stream_chat(user_text):
                for sentence in self._splitter.push(token):
                    self._set_status("Synthesizing...")
                    if not self.tts.push_text(sentence):
                        logger.warning("TTS 文本队列繁忙，已丢弃句子: %s", sentence)
            rest = self._splitter.flush()
            if rest and not self.tts.push_text(rest):
                logger.warning("TTS 文本队列繁忙，已丢弃残余文本: %s", rest)

    def _play_loop(self) -> None:
        """播放线程：播放 TTS PCM，并按顺序喂给 MuseTalk。"""
        import pyaudio

        rate = self.cfg.tts.output_sample_rate
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            output=True,
            output_device_index=self.cfg.app.audio_output_device_index,
        )
        try:
            while self._running:
                chunk = self.tts.get_audio_chunk(timeout=0.2)
                if chunk is None:
                    continue
                # 先推给 MuseTalk 再播放，保证驱动端看到的是"即将播放"的音频
                self.musetalk.push_audio(chunk)
                self._set_status("Speaking...")
                # 阻塞写声卡；写完后累加播放进度，供主线程按进度对齐帧
                stream.write(chunk)
                with self._played_lock:
                    self._played_seconds += len(chunk) / 2 / rate
                    played_now = self._played_seconds
                logger.debug("播放音频块 %.2fs，累计 %.2fs", len(chunk) / 2 / rate, played_now)
        except Exception:
            logger.exception("音频播放线程异常，正在停止应用")
            self._running = False
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
