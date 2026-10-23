"""启动前预检工具测试：不依赖真实模型、权重或网络。"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

from takemh.config import Config


def _load_preflight():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"
    spec = spec_from_file_location("preflight", script_path)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_preflight_warns_without_api_key(monkeypatch):
    preflight = _load_preflight()
    monkeypatch.setattr(preflight, "_module_missing", lambda name: False)
    monkeypatch.setattr(preflight.os.path, "exists", lambda path: True)

    cfg = Config()
    results = preflight.check_runtime_config(cfg)

    assert any(
        result.level == "warning" and "api_key" in result.message
        for result in results
    )
    assert not any(result.level == "error" for result in results)


def test_preflight_reports_missing_local_paths(monkeypatch):
    preflight = _load_preflight()
    monkeypatch.setattr(preflight, "_module_missing", lambda name: False)
    monkeypatch.setattr(preflight.os.path, "exists", lambda path: False)

    results = preflight.check_runtime_config(Config())

    assert any("TTS 参考音频" in result.message for result in results)
    assert any("MuseTalk 基准照片" in result.message for result in results)


def test_preflight_accepts_environment_api_key(monkeypatch):
    preflight = _load_preflight()
    monkeypatch.setattr(preflight, "_module_missing", lambda name: False)
    monkeypatch.setattr(preflight.os.path, "exists", lambda path: True)
    monkeypatch.setenv("TAKE_MH_LLM_API_KEY", "env-key")

    results = preflight.check_runtime_config(Config())

    assert not any("api_key" in result.message for result in results)
