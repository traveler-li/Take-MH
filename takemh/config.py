"""配置加载：读取 config.yaml 并转换为类型化配置对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class AppConfig:
    """应用级配置：音频输出设备、画面缩放与调试开关。"""
    audio_output_device_index: Optional[int] = None  # 音频输出设备索引，None 表示系统默认设备
    frame_display_scale: float = 0.75  # 画面显示缩放比例（1.0 为原始尺寸）
    debug: bool = False  # 调试模式，开启后打印更多模块状态日志


@dataclass
class AsrConfig:
    """ASR 配置：SenseVoice-small + fsmn-vad，强制 CPU。"""
    model: str = "iic/SenseVoiceSmall"  # SenseVoice-small 模型标识（FunASR 官方模型名）
    device: str = "cpu"  # 推理设备（ASR 固定 CPU，不使用 CUDA）
    vad_model: str = "fsmn-vad"  # VAD 模型名（fsmn-vad）
    sample_rate: int = 16000  # 麦克风采样率（16k）
    channels: int = 1  # 采集声道数（单声道）
    chunk_ms: int = 320  # 每次采集的音频块时长（毫秒）
    vad_time_unit_ms: int = 1  # VAD 返回时间戳单位（毫秒；部分版本为 10ms 帧号）
    vad_silence_ms: int = 600  # 语音结束后的静音阈值（毫秒），越小响应越快但可能切碎短句
    max_single_segment_ms: int = 15000  # 单段语音超过该时长强制切分（毫秒）
    max_buffer_seconds: float = 30.0  # ASR 音频缓冲保留时长（秒），防止长时间运行内存增长
    language: str = "auto"  # 识别语言（auto/zh/en/ja/ko/yue）
    use_itn: bool = True  # 是否启用标点/数字格式化
    batch_size_s: int = 60  # 识别批处理时长（秒）
    merge_vad: bool = True  # 是否合并 VAD 段
    merge_length_s: int = 15  # VAD 段合并长度（秒）


@dataclass
class LlmConfig:
    """LLM 配置：OpenAI 兼容接口、模型名、上下文历史。"""
    base_url: str = "https://api.deepseek.com/v1"  # OpenAI 兼容接口地址
    api_key: str = ""  # API 密钥（也可通过环境变量提供）
    model_name: str = "deepseek-chat"  # 对话模型名称
    temperature: float = 0.7  # 生成温度
    system_prompt: str = "你是一个友好、自然的语音助手，请用简洁口语化的中文回答。"  # 系统提示词
    timeout_seconds: float = 30.0  # 单次请求超时（秒）
    max_retries: int = 2  # 创建请求失败时的重试次数
    max_history_messages: int = 20  # 上下文历史保留的最大消息条数


@dataclass
class TtsConfig:
    """TTS 配置：CosyVoice2 模型、参考音频、输出采样率与队列容量。"""
    model_dir: str = "iic/CosyVoice2-0.5B"  # CosyVoice2 模型目录（本地路径或 ModelScope 模型名）
    fp16: bool = True  # GPU 上使用 fp16 推理
    prompt_wav: str = "assets/prompt.wav"  # 参考音频路径（3~10 秒，用于零样本音色克隆）
    prompt_text: str = "你好，我是你的虚拟助手，很高兴见到你。"  # 参考音频对应的文本（必须与实际内容一致）
    min_prompt_seconds: float = 3.0  # 参考音频最短时长（秒）
    max_prompt_seconds: float = 10.0  # 参考音频最长时长（秒）
    output_sample_rate: int = 16000  # 输出 PCM 采样率（16000，供播放与 MuseTalk 使用）
    stream: bool = True  # 是否使用官方流式输出
    inter_sentence_silence_seconds: float = 0.25  # 句间插入的静音时长（秒）
    text_queue_size: int = 64  # 文本队列容量
    audio_queue_size: int = 128  # MuseTalk 音频队列容量


@dataclass
class MuseTalkConfig:
    """MuseTalk 配置：模型/素材路径、推理设备、批次、帧率与队列容量。"""
    unet_model_path: str = "models/musetalkV15/unet.pth"  # MuseTalk UNet 权重路径
    unet_config: str = "models/musetalkV15/musetalk.json"  # MuseTalk UNet 配置路径
    vae_type: str = "sd-vae"  # VAE 类型（官方默认 sd-vae）
    whisper_dir: str = "models/whisper"  # Whisper 特征提取模型目录
    base_image: str = "assets/base.jpg"  # 基准照片路径（单张正面人像）
    device: str = "cuda:0"  # 推理设备（本地 GPU，如 cuda:0）
    fps: int = 25  # 输出帧率
    batch_size: int = 8  # UNet 推理批次大小（8GB 显存建议 2~4，过大 CUDA OOM）
    bbox_shift: int = 0  # 人脸框偏移（v1 使用）
    audio_padding_length_left: int = 2  # 音频特征左填充帧数
    audio_padding_length_right: int = 2  # 音频特征右填充帧数
    chunk_seconds: float = 1.0  # 每次处理的最小音频块时长（秒），分块逼近实时
    version: str = "v15"  # MuseTalk 版本（v15 / v1）
    left_cheek_width: int = 90  # v15 左脸颊解析区域宽度
    right_cheek_width: int = 90  # v15 右脸颊解析区域宽度
    frame_queue_size: int = 256  # 帧缓存队列容量
    audio_queue_size: int = 128  # MuseTalk 音频队列容量


@dataclass
class Config:
    """Take-MH 总配置：聚合 app/asr/llm/tts/musetalk 五个子配置。"""
    app: AppConfig = field(default_factory=AppConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    musetalk: MuseTalkConfig = field(default_factory=MuseTalkConfig)


def _build(section: str, cls: Any, data: Optional[Dict[str, Any]]) -> Any:
    """根据 yaml 字典构造 dataclass，未知字段忽略。"""
    if not data:
        return cls()
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str = "config.yaml") -> Config:
    """从 yaml 文件加载配置。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(
        app=_build("app", AppConfig, raw.get("app")),
        asr=_build("asr", AsrConfig, raw.get("asr")),
        llm=_build("llm", LlmConfig, raw.get("llm")),
        tts=_build("tts", TtsConfig, raw.get("tts")),
        musetalk=_build("musetalk", MuseTalkConfig, raw.get("musetalk")),
    )

