"""LLM 模块：基于官方 openai SDK 的 OpenAI 兼容对话客户端。"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterator, List, Optional

from takemh.config import LlmConfig

logger = logging.getLogger(__name__)


class LLMApiClient:
    """OpenAI 兼容流式对话客户端。

    - 维护多轮上下文历史，可手动清空
    - stream_chat() 逐字返回 token
    - 创建请求失败时自动重试（默认 2 次），最终失败返回兜底提示
    """

    FALLBACK_REPLY = "抱歉，我现在无法连接模型服务，请稍后再试。"

    def __init__(self, cfg: LlmConfig):
        self.cfg = cfg
        self._client = None
        self._system_prompt = cfg.system_prompt
        self._messages: List[dict] = [{"role": "system", "content": self._system_prompt}]

    def load(self) -> None:
        """创建 openai 客户端（官方 SDK）。"""
        from openai import OpenAI

        api_key = (
            self.cfg.api_key
            or os.getenv("TAKE_MH_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.cfg.base_url,
            timeout=self.cfg.timeout_seconds,
        )

    def _reset_messages(self) -> None:
        self._messages = [{"role": "system", "content": self._system_prompt}]

    def set_system_prompt(self, prompt: str) -> None:
        """更新系统提示词并重建历史。"""
        self._system_prompt = prompt
        self._reset_messages()

    def clear_history(self) -> None:
        """清空多轮对话历史，仅保留系统提示词。"""
        self._reset_messages()

    def get_history(self) -> List[dict]:
        return list(self._messages)

    def _append_history(self, role: str, content: str) -> None:
        """追加一条消息，超出上限时裁剪：保留系统提示词 + 最近的 N-1 条。"""
        self._messages.append({"role": role, "content": content})
        max_msgs = max(2, self.cfg.max_history_messages)
        if len(self._messages) > max_msgs:
            self._messages = [self._messages[0]] + self._messages[-(max_msgs - 1):]

    def _create_stream(self, messages: List[dict]) -> Iterator:
        """创建流式请求；失败时按配置重试，最终抛异常给调用方。"""
        last_error: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                return self._client.chat.completions.create(
                    model=self.cfg.model_name,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    stream=True,
                )
            except Exception as exc:  # noqa: BLE001 - 兼容 SDK 各类网络/API 异常
                last_error = exc
                logger.warning("LLM 请求失败(第 %d 次): %s", attempt + 1, exc)
                if attempt < self.cfg.max_retries:
                    time.sleep(0.8 * (attempt + 1))
        raise RuntimeError(f"LLM 请求重试 {self.cfg.max_retries} 次后仍失败: {last_error}")

    def stream_chat(self, user_text: str) -> Iterator[str]:
        """发送用户文本并逐字返回模型回复；自动追加到多轮历史。"""
        user_text = (user_text or "").strip()
        if not user_text:
            return

        self._append_history("user", user_text)
        messages = self.get_history()
        stream = None
        try:
            stream = self._create_stream(messages)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM 最终失败: %s", exc)
            self._append_history("assistant", self.FALLBACK_REPLY)
            yield self.FALLBACK_REPLY
            return

        collected: List[str] = []
        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                piece = getattr(delta, "content", None)
                if piece:
                    collected.append(piece)
                    yield piece
        except Exception as exc:  # noqa: BLE001 - 流中断时给出兜底提示
            logger.error("LLM 流式输出中断: %s", exc)
            self._append_history("assistant", self.FALLBACK_REPLY)
            yield self.FALLBACK_REPLY
            return

        reply = "".join(collected).strip()
        self._append_history("assistant", reply or self.FALLBACK_REPLY)

    def release(self) -> None:
        self._client = None
