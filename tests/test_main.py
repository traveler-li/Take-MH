"""命令行入口测试：验证配置写法与 API Key 解析，不加载模型。"""

from __future__ import annotations

import pytest

from main import parse_args, resolve_api_key


def test_parse_args_accepts_positional_config():
    args = parse_args(["config.yaml"])
    assert args.config_path == "config.yaml"
    assert args.config_option is None


def test_parse_args_accepts_config_option():
    args = parse_args(["--config", "config.yaml"])
    assert args.config_option == "config.yaml"
    assert args.config_path is None


def test_parse_args_rejects_both_config_forms():
    with pytest.raises(SystemExit):
        parse_args(["config.yaml", "--config", "config.yaml"])


def test_resolve_api_key_prefers_cli(monkeypatch):
    monkeypatch.setenv("TAKE_MH_LLM_API_KEY", "env-key")
    assert resolve_api_key("cli-key", "config-key") == "cli-key"


def test_resolve_api_key_uses_environment(monkeypatch):
    monkeypatch.setenv("TAKE_MH_LLM_API_KEY", "env-key")
    assert resolve_api_key(None, "config-key") == "env-key"


def test_resolve_api_key_falls_back_to_legacy_config(monkeypatch):
    monkeypatch.delenv("TAKE_MH_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve_api_key(None, "config-key") == "config-key"


def test_resolve_api_key_prompts_without_echo(monkeypatch):
    monkeypatch.delenv("TAKE_MH_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    prompts = []

    def fake_prompt(prompt):
        prompts.append(prompt)
        return "  prompted-key  "

    assert resolve_api_key(None, None, fake_prompt) == "prompted-key"
    assert prompts and "不回显" in prompts[0]
