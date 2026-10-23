# Take-MH 实时照片虚拟人对话系统

基于 Python 3.10 的实时照片虚拟人对话系统。四个模型模块通过有界队列和独立线程串联，尽量降低端到端延迟：

```text
麦克风/测试 WAV
    -> SenseVoice-small ASR（CPU，VAD 端点检测）
    -> OpenAI 兼容 LLM（远程 API，流式 token）
    -> CosyVoice2 0.5B TTS（本地 GPU，流式音频）
    -> 16 kHz PCM
    -> MuseTalk（本地 GPU，口型驱动帧）
    -> OpenCV 窗口 + PyAudio 播放
```

## 功能概览

- 麦克风实时采集或 WAV 文件回放测试。
- SenseVoice VAD 端点检测，支持可配置时间戳单位。
- LLM 多轮上下文、流式输出、自动重试与兜底提示。
- CosyVoice 零样本音色克隆，复用参考音频特征缓存。
- MuseTalk 按音频块生成 BGR 帧，主线程按播放进度渲染。
- API Key 不写入配置文件，支持启动时隐藏输入、环境变量或 `--api-key`。
- 启动前预检、模块级单元测试与语法检查。

## 环境要求

- Windows 10/11，PowerShell。
- Python 3.10。
- NVIDIA GPU、可用的 CUDA 驱动，以及与该驱动匹配的 PyTorch 2.7.x。
- FFmpeg，建议可从 `PATH` 直接调用。
- 麦克风和扬声器；若使用虚拟人语音输出，建议佩戴耳机或启用回声消除。

## 安装

### 1. 创建独立虚拟环境

不要直接在系统 Python 或其他模型项目环境中安装本项目依赖。

```powershell
cd D:\GitHub\Take-MH
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. 安装 CUDA 版 PyTorch

先在 PyTorch 官方安装页选择与显卡和驱动匹配的 CUDA wheel。项目锁定在 PyTorch/TorchAudio 2.7.x，避免两者版本错配。

```powershell
# 示例：请以 PyTorch 官方页面给出的当前 CUDA 版本命令为准
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

### 3. 安装项目依赖

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

`pip check` 应无冲突。若出现 OpenCV、NumPy 或 Hydra 冲突，通常是全局环境中混装了其他语音项目，请重建虚拟环境。

### 4. 安装官方模型仓库

CosyVoice 与 MuseTalk 需要按各自官方仓库的说明安装。默认从项目同级目录查找：

```text
D:\GitHub\CosyVoice
D:\GitHub\MuseTalk
D:\GitHub\Take-MH
```

```powershell
cd D:\GitHub
git clone https://github.com/FunAudioLLM/CosyVoice.git
git clone https://github.com/TMElyralab/MuseTalk.git

cd D:\GitHub\CosyVoice
python -m pip install -r requirements.txt

cd D:\GitHub\MuseTalk
python -m pip install -r requirements.txt
```

项目会自动把上述仓库根目录加入 `sys.path`。建议固定已经验证过的官方仓库提交，避免私有接口升级后行为变化。

## 模型与素材准备

| 内容 | 默认位置/标识 | 说明 |
|---|---|---|
| SenseVoice-small | `iic/SenseVoiceSmall` | FunASR 首次运行可从 ModelScope 下载 |
| VAD | `fsmn-vad` | FunASR 官方 VAD 模型 |
| CosyVoice2 0.5B | `iic/CosyVoice2-0.5B` | 可改为本地模型目录 |
| MuseTalk UNet | `models/musetalkV15/unet.pth` | 按官方仓库准备 |
| MuseTalk 配置 | `models/musetalkV15/musetalk.json` | 与 UNet 版本匹配 |
| SD-VAE | `models/sd-vae` | 路径由 MuseTalk 官方代码使用 |
| Whisper | `models/whisper` | MuseTalk 音频特征提取 |
| 基准照片 | `assets/base.jpg` | 单张正面、清晰人脸 |
| 参考音频 | `assets/prompt.wav` | 与 `tts.prompt_text` 内容一致 |

`models/`、`assets/`、`*.wav` 默认被 `.gitignore` 忽略。部署到新机器时需要单独准备这些文件，不能只复制代码仓库。

## API Key 配置

默认 `config.yaml` 不保存 API Key。推荐在启动时由终端隐藏输入：

```powershell
python main.py --config config.yaml
# 终端提示：LLM API Key（输入不回显）:
```

也可以使用环境变量：

```powershell
$env:TAKE_MH_LLM_API_KEY = "your-key"
python main.py --config config.yaml
```

兼容 `OPENAI_API_KEY`。临时测试还可使用：

```powershell
python main.py --config config.yaml --api-key "your-key"
```

> `--api-key` 可能出现在命令历史、进程列表或日志中，不建议在共享机器使用。优先级为：`--api-key` > `TAKE_MH_LLM_API_KEY` > `OPENAI_API_KEY` > 旧配置中的 `llm.api_key` > 交互输入。

若曾经把真实密钥写入 `config.yaml`、日志或版本库，请立即在服务商控制台吊销并重新生成。

## 配置说明

主配置位于 `config.yaml`。

- `app`：输出设备索引、窗口缩放、调试日志。
- `asr`：SenseVoice 模型、CPU 设备、VAD 阈值、时间戳单位、缓冲长度。
- `llm`：OpenAI 兼容地址、模型名、温度、超时、重试与历史长度。
- `tts`：CosyVoice 模型、参考音频及文本、采样率、句间静音、队列容量。
- `musetalk`：UNet/VAE/Whisper/照片路径、GPU 设备、FPS、批大小、音频块长度。

`asr.vad_time_unit_ms` 表示 VAD 返回时间戳的单位：返回毫秒时填 `1`，返回 10 ms 帧号时填 `10`。若端点切分明显偏移，应优先核对该项。

## 启动前检查

```powershell
python scripts/preflight.py --config config.yaml
```

检查 Python 版本、API Key 来源、ASR 设备、VAD 时间单位、本地素材/模型路径和关键依赖。

## 启动

```powershell
python main.py --config config.yaml
```

兼容旧的位置参数写法：

```powershell
python main.py config.yaml
```

按 `Esc`、`q` 或关闭窗口退出。

### 无麦克风测试

```powershell
python main.py --config config.yaml --test-audio assets/prompt.wav
```

测试 WAV 会经过 ASR -> LLM -> TTS -> MuseTalk 全链路。非 16 kHz 单声道 WAV 会自动重采样；多声道文件不支持。

## 测试

```powershell
python -m pytest -q
python -m compileall -q main.py takemh scripts tests
```

测试使用 mock 模型接口和临时音频文件，不加载模型权重、不联网、不依赖 GPU、麦克风或 API Key。

## 命令行参数

```text
python main.py [config_path] [--config CONFIG] [--api-key KEY] [--test-audio WAV]
```

- `config_path`：位置参数形式的配置文件路径。
- `-c, --config`：推荐的配置文件参数写法。
- `--api-key`：临时提供密钥；优先使用隐藏输入或环境变量。
- `--test-audio`：用 WAV 文件替代麦克风输入。

位置参数和 `--config` 不能同时使用。

## 模块职责

| 文件 | 类 | 职责 |
|---|---|---|
| `takemh/asr_sensevoice.py` | `SenseVoiceASR` | CPU 端流式 VAD 与语音识别 |
| `takemh/llm_openai.py` | `LLMApiClient` | OpenAI 兼容接口、流式输出、重试和上下文 |
| `takemh/tts_cosyvoice.py` | `CosyVoiceTTS` | 零样本 TTS、提示特征缓存、16 kHz PCM |
| `takemh/musetalk_driver.py` | `MuseTalkDriver` | 音频块驱动 MuseTalk 生成 BGR 帧 |
| `takemh/pipeline.py` | `VirtualAvatarApp` | 线程编排、队列、播放与窗口渲染 |
| `takemh/text_splitting.py` | `SentenceSplitter` | 流式 token 按句切分并保留标点 |
| `takemh/audio_io.py` | 工具函数 | PCM 转换、重采样、WAV 编码 |
| `takemh/audio_file.py` | `WavFileInputStream` | WAV 测试输入流 |
| `takemh/config.py` | 配置类 | YAML 到类型化配置对象 |

## 性能调优

当前主要瓶颈通常是 TTS 和 MuseTalk，而不是 LLM。

1. 先用 `python scripts/preflight.py --config config.yaml` 确认没有依赖或路径错误。
2. 打开 `app.debug`，记录 TTS 首音频块、实时率（RTF）和 MuseTalk 帧生成速度。
3. 在 8 GB 显存下，`musetalk.batch_size` 建议从 `2` 开始；出现 CUDA OOM 时继续降低。
4. `musetalk.chunk_seconds` 与 `fps` 需要实测权衡，单纯增大分块不一定提高吞吐。
5. 安装与 CUDA 匹配的 `onnxruntime-gpu` 可能改善 CosyVoice 提示特征提取，但收益取决于环境。
6. 若 MuseTalk 持续慢于实时，系统会丢弃过时帧以保持画面更新；这不能消除硬件层面的延迟。

## 常见问题

### `ModuleNotFoundError: cosyvoice` 或 `musetalk`

确认官方仓库位于 `D:\GitHub\CosyVoice`、`D:\GitHub\MuseTalk`，或把对应目录加入 `PYTHONPATH`。同时检查是否在同一个虚拟环境中安装官方依赖。

### CUDA out of memory

降低 `musetalk.batch_size`，关闭其他占用显存的程序，并确认 TTS 与 MuseTalk 没有在初始化阶段叠加峰值。

### 没有声音或麦克风无输入

检查 Windows 输入/输出设备、`app.audio_output_device_index` 以及 PyAudio 设备列表。使用耳机可降低扬声器声音被麦克风再次识别的风险。

### 画面不动或没有帧

检查 MuseTalk 权重、Whisper 目录、基准照片人脸检测结果和 GPU 显存。错误会写入日志。

### VAD 截断语音或长时间不结束

调整 `asr.vad_silence_ms`，并确认 `asr.vad_time_unit_ms` 与当前 FunASR 版本一致。

### 依赖检查出现 Hydra/OpenCV/NumPy 冲突

通常来自多个语音项目共用全局环境。请创建新的 `.venv`，先安装匹配的 PyTorch，再安装本项目依赖。

### 中文日志乱码

程序会把标准输出切换为 UTF-8。若终端仍乱码，可在 PowerShell 中执行：

```powershell
chcp 65001
```

## 已知限制

- MuseTalk 在低功耗 GPU 上可能无法实时，当前实现优先保证播放连续和画面持续更新。
- 未内置声学回声消除。免提使用时，虚拟人的声音可能被麦克风识别。
- CosyVoice 提示特征缓存依赖官方内部接口，升级 CosyVoice 后需要重新做集成验证。
- 当前 ASR 的 `on_interim` 只提供语音段状态，不是逐字中间转写。

## 安全建议

- 不要把 API Key 写入 `config.yaml`、README、日志、截图或提交记录。
- 使用独立虚拟环境和固定版本的官方仓库。
- 模型权重、测试音频和输出视频不要提交到版本库。
- 在共享机器上避免使用 `--api-key`，优先使用环境变量或隐藏输入。

## Contributors

详见 [CONTRIBUTORS.md](CONTRIBUTORS.md)。
