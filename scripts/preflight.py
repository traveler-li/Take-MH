"""Take-MH 启动前检查：验证配置、本地素材/模型路径与关键依赖。"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Auto-add official repo paths for module detection
_REPO_PATHS = [
    ROOT.parent / "CosyVoice",
    ROOT.parent / "MuseTalk",
    ROOT.parent / "MuseTalk" / "musetalk" / "utils",
]
for _p in _REPO_PATHS:
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from takemh.config import Config, load_config  # noqa: E402

REMOTE_MODEL_PREFIXES = ("iic/", "modelscope/", "http://", "https://")


@dataclass
class CheckResult:
    level: str
    message: str


def _module_missing(name: str) -> bool:
    """判断 Python 模块是否缺失（用于关键依赖检查）。"""
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError):
        return True


def _is_remote(path: str) -> bool:
    """路径是否为远程模型标识（iic/、modelscope/、http 等）。"""
    return bool(path) and path.startswith(REMOTE_MODEL_PREFIXES)


def _add_local_path(results: List[CheckResult], label: str, path: str) -> None:
    """检查本地路径是否存在，远程标识或未配置时跳过/记录错误。"""
    if not path:
        results.append(CheckResult("error", f"{label} 未配置"))
        return
    if not os.path.isabs(path):
        path = os.path.join(str(ROOT), path)
    if _is_remote(path):
        return
    if not os.path.exists(path):
        results.append(CheckResult("error", f"{label} 不存在: {path}"))


def check_runtime_config(cfg: Config) -> List[CheckResult]:
    """核对 Python 版本、LLM 配置、本地素材/模型路径与关键依赖。

    返回 CheckResult 列表；error 表示阻断启动，warning 表示建议关注。
    """
    results: List[CheckResult] = []

    if sys.version_info[:2] < (3, 10):
        results.append(CheckResult("error", "需要 Python 3.10+"))

    if not cfg.llm.base_url:
        results.append(CheckResult("error", "llm.base_url 未配置"))
    llm_api_key = (
        cfg.llm.api_key
        or os.getenv("TAKE_MH_LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not llm_api_key:
        results.append(CheckResult("warning", "llm.api_key 为空，真实请求需要 API 密钥"))
    if not cfg.llm.model_name:
        results.append(CheckResult("error", "llm.model_name 未配置"))
    if cfg.llm.api_key:
        results.append(
            CheckResult("warning", "llm.api_key 不建议写入 config.yaml，请改用命令行或环境变量")
        )

    if not cfg.asr.model:
        results.append(CheckResult("error", "asr.model 未配置"))
    if not cfg.asr.vad_model:
        results.append(CheckResult("error", "asr.vad_model 未配置"))
    if cfg.asr.device.lower() != "cpu":
        results.append(CheckResult("error", "asr.device 必须为 cpu"))
    if cfg.asr.vad_time_unit_ms <= 0:
        results.append(CheckResult("error", "asr.vad_time_unit_ms 必须大于 0"))
    if cfg.tts.output_sample_rate != 16000:
        results.append(CheckResult("warning", "tts.output_sample_rate 建议保持 16000"))

    _add_local_path(results, "TTS 参考音频", cfg.tts.prompt_wav)
    _add_local_path(results, "MuseTalk 基准照片", cfg.musetalk.base_image)
    _add_local_path(results, "MuseTalk UNet 权重", cfg.musetalk.unet_model_path)
    _add_local_path(results, "MuseTalk UNet 配置", cfg.musetalk.unet_config)
    _add_local_path(results, "MuseTalk Whisper 目录", cfg.musetalk.whisper_dir)

    required_modules = [
        "numpy",
        "yaml",
        "scipy",
        "pyaudio",
        "librosa",
        "openai",
        "cv2",
        "torch",
        "transformers",
        "funasr",
    ]
    for module in required_modules:
        if _module_missing(module):
            results.append(CheckResult("error", f"缺少关键依赖: {module}"))

    optional_modules = [
        ("cosyvoice", "CosyVoice 官方仓库/PYTHONPATH"),
        ("musetalk", "MuseTalk 官方仓库/PYTHONPATH"),
    ]
    for module, hint in optional_modules:
        if _module_missing(module):
            results.append(
                CheckResult("warning", f"缺少官方仓库包: {module}（{hint}）")
            )

    return results


def run_preflight(config_path: str) -> int:
    try:
        cfg = load_config(config_path)
    except Exception as exc:  # noqa: BLE001 - 命令行应直接展示配置读取错误
        print(f"[ERROR ] 无法加载配置 {config_path}: {exc}")
        return 1

    results = check_runtime_config(cfg)
    errors = sum(result.level == "error" for result in results)
    warnings = sum(result.level == "warning" for result in results)

    print(f"Take-MH 启动前检查（{config_path}）")
    for result in results:
        print(f"[{result.level.upper():<7}] {result.message}")
    print(f"检查完成: 共 {len(results)} 项，错误 {errors}，警告 {warnings}")
    return 1 if errors else 0


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Take-MH 启动前环境与配置检查")
    parser.add_argument(
        "--config",
        default=str(ROOT / "config.yaml"),
        help="运行配置路径，默认项目根目录下的 config.yaml",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    return run_preflight(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
