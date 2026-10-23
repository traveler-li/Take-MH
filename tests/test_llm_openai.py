"""LLMApiClient 最小独立测试：mock openai 客户端，验证流式、历史、重试与兜底。"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from takemh.config import LlmConfig
from takemh.llm_openai import LLMApiClient


def _chunk(content):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


class FakeCompletions:
    """按预设序列返回流 chunk 或抛异常的 create()。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return iter(resp)


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def _client(responses):
    cfg = LlmConfig(
        base_url="http://example.test/v1",
        api_key="test-key",
        model_name="test-model",
        max_retries=2,
        system_prompt="你是测试助手。",
    )
    c = LLMApiClient(cfg)
    c._client = FakeClient(responses)
    return c


def test_stream_chat_yields_tokens_and_updates_history():
    client = _client([[_chunk("你"), _chunk("好"), _chunk("世界")]])
    tokens = list(client.stream_chat("早上好"))
    assert "".join(tokens) == "你好世界"
    history = client.get_history()
    assert [m["role"] for m in history] == ["system", "user", "assistant"]
    assert history[-1]["content"] == "你好世界"
    # 第二次对话应携带上一轮历史
    client2 = _client([[_chunk("嗯")]])
    client2._client = FakeClient([[_chunk("嗯")]])
    client2._append_history("user", "上一轮问题")
    client2._append_history("assistant", "上一轮回答")
    list(client2.stream_chat("再问"))
    sent = client2._client.chat.completions.calls[0]
    assert len(sent["messages"]) == 4


def test_clear_history_keeps_only_system_prompt():
    client = _client([[_chunk("好")]])
    list(client.stream_chat("hi"))
    client.clear_history()
    assert len(client.get_history()) == 1
    assert client.get_history()[0]["role"] == "system"


def test_retries_then_succeeds():
    client = _client([RuntimeError("timeout"), [_chunk("重试成功")]])
    tokens = list(client.stream_chat("问题"))
    assert "".join(tokens) == "重试成功"
    assert client._client.chat.completions.calls[0]["stream"] is True
    assert client._client.chat.completions.calls[0]["model"] == "test-model"


def test_all_failures_return_fallback():
    client = _client([RuntimeError("a"), RuntimeError("b"), RuntimeError("c")])
    tokens = list(client.stream_chat("问题"))
    assert tokens == [LLMApiClient.FALLBACK_REPLY]


def test_empty_input_yields_nothing():
    client = _client([])
    assert list(client.stream_chat("   ")) == []


def test_load_uses_environment_api_key_when_config_empty(monkeypatch):
    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("TAKE_MH_LLM_API_KEY", "env-key")

    client = LLMApiClient(LlmConfig(api_key=""))
    client.load()

    assert client._client.kwargs["api_key"] == "env-key"
