"""Take-MH 入口：加载配置并运行实时照片虚拟人对话系统。"""

from __future__ import annotations

import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from getpass import getpass
from typing import Callable, Optional, Sequence

from takemh.config import load_config

_API_KEY_ENV_VARS = ("TAKE_MH_LLM_API_KEY", "OPENAI_API_KEY")


def build_parser() -> ArgumentParser:
    """构造命令行解析器。"""
    parser = ArgumentParser(description="Take-MH 实时照片虚拟人对话系统")
    # 同时支持 `main.py config.yaml` 与 `main.py --config config.yaml`，兼容旧用法。
    parser.add_argument("config_path", nargs="?", help="配置文件路径（兼容位置参数写法）")
    parser.add_argument(
        "-c",
        "--config",
        dest="config_option",
        help="配置文件路径，默认 config.yaml",
    )
    parser.add_argument(
        "--api-key",
        help=(
            "LLM API 密钥，优先级最高；会出现在进程参数和命令历史中。"
            "建议留空，改用环境变量或启动时的隐藏输入。"
        ),
    )
    parser.add_argument("--test-audio", help="从 WAV 文件读取音频替代麦克风")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> Namespace:
    """解析并校验命令行参数。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config_option and args.config_path:
        parser.error("配置路径不能同时使用位置参数和 --config")
    return args


def _config_path(args: Namespace) -> str:
    return args.config_option or args.config_path or "config.yaml"


def resolve_api_key(
    cli_api_key: Optional[str],
    config_api_key: Optional[str] = None,
    prompt_func: Optional[Callable[[str], str]] = None,
) -> str:
    """按 CLI、环境变量、旧配置、交互输入的优先级解析 API 密钥。

    ``--api-key`` 会暴露在命令行历史中，因此交互式隐藏输入是默认推荐方式。
    标准输入不是终端时不会阻塞等待输入，而是返回空字符串交由调用方报错。
    """
    for candidate in (cli_api_key, *(os.getenv(name) for name in _API_KEY_ENV_VARS)):
        value = (candidate or "").strip()
        if value:
            return value

    # 兼容旧配置，但 README 和默认配置不再建议把密钥写入 YAML。
    legacy_value = (config_api_key or "").strip()
    if legacy_value:
        return legacy_value

    if prompt_func is None:
        if not sys.stdin.isatty():
            return ""
        prompt_func = getpass
    try:
        return (prompt_func("LLM API Key（输入不回显）: ") or "").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口：解析参数、加载配置、启动并运行虚拟人应用。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = parse_args(argv)
    cfg = load_config(_config_path(args))
    logging.basicConfig(
        level=logging.DEBUG if cfg.app.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if cfg.llm.api_key:
        logging.warning("检测到 config.yaml 中的明文 api_key，建议删除并改用命令行安全输入或环境变量")

    api_key = resolve_api_key(args.api_key, cfg.llm.api_key)
    if not api_key:
        logging.error(
            "缺少 LLM API Key。可设置 TAKE_MH_LLM_API_KEY / OPENAI_API_KEY，"
            "或在终端中按提示输入，也可使用 --api-key（不推荐共享机器）"
        )
        return 2
    cfg.llm.api_key = api_key

    # 延迟导入重型模型模块，保证 --help 和参数错误无需加载 NumPy/Torch 等依赖。
    from takemh.pipeline import VirtualAvatarApp

    app = VirtualAvatarApp(cfg, test_audio_path=args.test_audio)
    try:
        app.start()
        app.run()
    except KeyboardInterrupt:
        logging.info("收到中断信号，正在退出...")
    except Exception as exc:  # noqa: BLE001 - 顶层兜底，保证资源释放
        logging.exception("程序异常: %s", exc)
        return 1
    finally:
        app.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
