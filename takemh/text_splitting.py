"""文本切分：把流式 LLM token 按句子边界切分为 TTS 可消费的完整句子。"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"[。！？；!?;…\n]+")


class SentenceSplitter:
    """按中文标点切分句子，push() 返回完整句子，flush() 返回残余文本。"""

    def __init__(self, min_chars: int = 2):
        self.min_chars = min_chars
        self._buffer = ""

    def push(self, text: str) -> list:
        """输入新增文本，返回可消费的完整句子列表。"""
        self._buffer += text or ""
        sentences: list = []
        parts = _SENTENCE_END.split(self._buffer)
        delimiters = _SENTENCE_END.findall(self._buffer)
        # 最后一个 part 是不完整句子，留在缓冲；其余句子保留终止标点。
        if len(parts) > 1:
            for index, part in enumerate(parts[:-1]):
                content = part.strip()
                if content:
                    sentence = content + delimiters[index].strip()
                    sentences.append(sentence)
        self._buffer = parts[-1]
        # 缓冲区积累过长且无标点，按配置切分（防御性，避免 LLM 不输出标点）
        if len(self._buffer) >= max(80, self.min_chars * 40):
            sentences.append(self._buffer)
            self._buffer = ""
        return sentences

    def flush(self) -> str:
        """返回并清空残余文本（流结束时调用）。"""
        rest = self._buffer.strip()
        self._buffer = ""
        return rest

    def reset(self) -> None:
        self._buffer = ""

