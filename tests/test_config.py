"""配置加载测试：验证 YAML 映射到类型化配置对象。"""

from __future__ import annotations

import os
import tempfile

from takemh.config import Config, load_config


def test_load_config_reads_known_sections():
    tmpdir = tempfile.mkdtemp()
    config_path = os.path.join(tmpdir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("""
app:
  frame_display_scale: 0.5
  debug: true
asr:
  chunk_ms: 640
llm:
  model_name: "test-model"
  api_key: "test-key"
tts:
  output_sample_rate: 24000
musetalk:
  fps: 30
""")

    cfg = load_config(str(config_path))

    assert cfg.app.frame_display_scale == 0.5
    assert cfg.app.debug is True
    assert cfg.asr.chunk_ms == 640
    assert cfg.llm.model_name == "test-model"
    assert cfg.llm.api_key == "test-key"
    assert cfg.tts.output_sample_rate == 24000
    assert cfg.musetalk.fps == 30


def test_load_config_ignores_unknown_fields_and_uses_defaults():
    tmpdir = tempfile.mkdtemp()
    config_path = os.path.join(tmpdir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("""
app:
  unknown: 1
asr:
  unknown_model: "x"
""")

    cfg = load_config(str(config_path))

    assert isinstance(cfg, Config)
    assert cfg.app.frame_display_scale == 0.75
    assert cfg.asr.model == "iic/SenseVoiceSmall"