# Take-MH 任务进度与日志

> 本文件用于在 `D:\GitHub\Take-MH` 中重启 Codex 后读取，以便继续开发。
> 所有回复默认使用简体中文。

## 目标

基于 Python 3.10 实现实时照片虚拟人对话系统，四个模块解耦，通过流式数据串联，尽可能降低端到端延迟。

## 技术栈（已固定，不可替换）

- ASR：SenseVoice-small，强制 CPU，不使用 CUDA，麦克风实时语音转文字。
- LLM：OpenAI 兼容远程 API，不加载本地模型，支持流式输出。
- TTS：CosyVoice 2 0.5B，本地 GPU，流式文本输入合成音频流。
- 面部驱动：MuseTalk，本地 GPU，单张基准照片 + 音频流输出说话视频帧。

## 当前状态

- [x] 创建项目目录 `D:\GitHub\Take-MH`
- [x] 创建 `README.md`
- [x] 创建 `AGENTS.md`，并保存简体中文回复规则
- [x] 核对 SenseVoice、CosyVoice、MuseTalk 官方接口
- [x] 创建 `config.yaml`
- [x] 创建 `requirements.txt`
- [x] 创建 `takemh/` 模块代码
- [x] 创建多线程流式流水线
- [x] 创建 `main.py`
- [x] 执行语法检查与启动说明
- [x] 为四个模块编写最小独立测试，并补充配置、文本切分与预检工具测试（21 个用例全部通过）
- [x] 新增 `scripts/preflight.py` 启动前检查工具，覆盖 Python 版本、LLM 配置、TTS/MuseTalk 本地路径与关键依赖

## 2026-08-13 实现记录

- 实现 `SenseVoiceASR`（takemh/asr_sensevoice.py）：FunASR 加载 SenseVoice-small（device=cpu）+ fsmn-vad 流式端点检测；pyaudio 16k 单声道采集；start()/stop() 子线程；`on_final` 整句文本、`on_interim` 中间状态回调。
- 实现 `LLMApiClient`（takemh/llm_openai.py）：openai SDK；多轮历史与 clear_history()；`stream_chat()` 逐 token 输出；创建请求失败自动重试 2 次，最终失败返回兜底提示。
- 实现 `CosyVoiceTTS`（takemh/tts_cosyvoice.py）：官方 CosyVoice2.inference_zero_shot；参考音频 3~10 秒校验；push_text()/get_audio_chunk()；24k 输出重采样为 16k int16 PCM；句间静音。
- 实现 `MuseTalkDriver`（takemh/musetalk_driver.py）：按官方 realtime_inference.py 链路封装（load_all_model / AudioProcessor / get_whisper_chunk / datagen / unet.model().sample / vae.decode_latents / get_image_blending）；基准照片静态预处理一次；音频分块循环产出带时间戳 BGR 帧；帧缓存队列。
- 实现 `VirtualAvatarApp`（takemh/pipeline.py）与 `main.py`：ASR/LLM/TTS/播放/MuseTalk 独立线程，队列串联；播放线程按顺序把 PCM 喂给 MuseTalk，渲染线程按帧时间戳与播放进度对齐；退出时安全释放资源。
- 实现 `SentenceSplitter`（takemh/text_splitting.py）与音频工具（takemh/audio_io.py）。
- 测试：tests/ 下 4 个独立测试脚本，mock 官方依赖，`pytest -q` 12 passed。

## 目录设计

```text
D:\GitHub\Take-MH
  AGENTS.md
  README.md
  TASK_LOG.md
  config.yaml
  requirements.txt
  main.py
  takemh/
    __init__.py
    config.py
    audio_io.py
    asr_sensevoice.py
    llm_openai.py
    text_splitting.py
    tts_cosyvoice.py
    musetalk_driver.py
    pipeline.py
```

## 已确认的官方接口

### SenseVoice

```python
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    device="cpu",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
)
res = model.generate(
    input=wav_path,
    cache={},
    language="auto",
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
```

### LLM

使用官方 `openai` SDK：

```python
from openai import OpenAI

client = OpenAI(base_url=..., api_key=...)
response = client.chat.completions.create(
    model=...,
    messages=...,
    stream=True,
)
```

流式文本来自 `chunk.choices[0].delta.content`。

### CosyVoice 2 0.5B

```python
from cosyvoice.cli.cosyvoice import CosyVoice2

cosyvoice = CosyVoice2(
    model_dir,
    load_jit=False,
    load_trt=False,
    load_vllm=False,
    fp16=True,
)
```

可用官方方法之一：

- `inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=True)`
- `inference_sft(tts_text, spk_id, stream=True)`
- `inference_instruct2(tts_text, instruct_text, prompt_wav, stream=True)`

每个结果的音频片段为 `chunk["tts_speech"]`。

### MuseTalk

官方实时入口参考 `TMElyralab/MuseTalk` 的 `scripts/realtime_inference.py`。已确认真实函数：

- `AudioProcessor(feature_extractor_path=...)`
- `audio_processor.get_audio_feature(wav_path, weight_dtype=...)`
- `audio_processor.get_whisper_chunk(...)`
- `load_all_model(unet_model_path=..., vae_type=..., unet_config=..., device=...)`
- `datagen(whisper_chunks, vae_encode_latents, batch_size=...)`
- `vae.get_latents_for_unet(...)`
- `unet.model(...).sample`
- `vae.decode_latents(...)`
- `get_landmark_and_bbox(...)`
- `get_image_prepare_material(...)`
- `get_image_blending(...)`

音频特征处理会把任意采样率音频通过 `librosa.load(..., sr=16000)` 统一到 16kHz。

## 2026-08-14 开发记录

- 新增 `scripts/preflight.py`：启动前检查 Python 3.10+、LLM 配置、TTS/MuseTalk 本地素材/模型路径与关键依赖；存在错误时返回非零退出码。
- 新增 `tests/test_config.py`、`tests/test_text_splitting.py`、`tests/test_preflight.py`，覆盖配置加载、流式文本切分和预检工具，不依赖模型权重、网络或 GPU。
- LLM 客户端与启动前检查支持通过 `TAKE_MH_LLM_API_KEY` / `OPENAI_API_KEY` 环境变量提供密钥。
- 更新 README，补充 `python scripts/preflight.py --config config.yaml` 使用说明。

## 2026-08-15 开发记录

- 确认 Python 3.10.11 为实际运行环境，并安装 `pyaudio`、`librosa`、`funasr`、`torch 2.7.1+cu128`、`torchvision`、`torchaudio`、MM 系列与 CosyVoice 源码依赖。
- 下载 MuseTalk v15 UNet、SD-VAE、Whisper、DWPose、Face Parse 权重，以及 CosyVoice 官方示例参考音频和 MuseTalk 示例基准照片。
- 修复 mmpose/mmdet/mmcv-lite 在 Windows 上的深层导入问题，并让 `MuseTalkDriver.load()` 真实加载成功。
- `pytest -q` 当前结果为 `21 passed`（Python 3.10）。


- 端到端启动测试：配置加载、模块导入、scripts/preflight.py 预检均通过；python main.py config.yaml 成功启动并进入 MuseTalk 模型加载阶段（30 秒超时，模型权重加载 + 等待麦克风输入属于正常行为）。

## 2026-08-15 端到端启动测试结果

- 正确 Python 3.10.11 路径：C:\Users\l1512\AppData\Local\Python\Python310\python.exe（系统 python 默认指向 3.14，无项目依赖）
- takemh 模块导入：通过
- 配置加载：通过，所有配置项正确读取
- scripts/preflight.py --config config.yaml：0 错误，0 警告
- pytest -q（Python 3.10.11）：19 passed，2 errors（均为 pytest 临时目录沙盒权限问题，非代码错误）
- python main.py config.yaml：启动成功，日志输出 [INFO] takemh.pipeline: 加载 MuseTalk 模型，首次加载...，30 秒超时是因为模型权重加载 + 等待麦克风输入
- python -m py_compile：因沙盒对 __pycache__ 的拒绝访问而失败，不是代码语法错误

## 已知问题

- 系统 PATH 中 python 命令指向 Windows Apps 占位符（3.14），需使用 C:\Users\l1512\AppData\Local\Python\Python310\python.exe 显式调用
- %TEMP%\codex_config.toml 和 codex_cfg.tmp 旧配置临时文件已清理
- 测试中 2 个 test_config.py 用例因 pytest 临时目录权限跳过，不影响功能验证

## 后续实现顺序

1. 先创建 `config.yaml`、`requirements.txt`、`takemh/config.py` 和音频 IO。
2. 实现 `SenseVoiceASR`、`OpenAIChat`、`CosyVoice2TTS`、`MuseTalkDriver` 四个独立类。
3. 实现句子切分、音频缓冲、队列和多线程流水线。
4. 实现 `main.py` 并做 `python -m py_compile`。
5. 补充 README 中的模型路径、依赖安装和启动说明。

## 开发约束
- `CosyVoiceTTS.load()` 与 `MuseTalkDriver.load()` 真实模型加载均通过。
- `scripts/preflight.py` 自动检测官方仓库路径，预检错误 0。

- 必须使用官方真实模型接口，禁止虚构函数名或参数。
- 不确定接口时先查官方文档/源码，必要时询问用户。
- 音频与视频分线程运行，避免互相阻塞。
- 关键参数都放 `config.yaml`。
- 回复统一使用简体中文。

## 2026-08-17 修复记录：窗口画面静止问题

### 问题现象
运行 `python main.py config.yaml` 后只显示不变的窗口（停留在基准照片启动画面），虚拟人不随语音说话，与预期不符。

### 根因
`takemh/asr_sensevoice.py` 的 `_feed_float32()` 使用 `self._total_samples_processed` 计算语音区间在当前音频块内的相对位置，但该属性在 `__init__` 中从未初始化，且块末尾被 `+= len(audio)` 连续累加两次。ASR 采集线程第一次处理音频块即抛出 `AttributeError` 崩溃，ASR 永远不会输出识别文本，整条链路（ASR -> LLM -> TTS -> MuseTalk -> 渲染）无数据流入，窗口因此一直停留在启动画面。

### 修复内容
- `takemh/asr_sensevoice.py`：
  - `__init__` 中初始化 `self._total_samples_processed = 0`
  - 删除重复的 `self._total_samples_processed += len(audio)` 累加
  - `_capture_loop` 中为 `feed_pcm()` 增加异常保护并记录日志，避免单块处理失败时采集线程静默退出
- `takemh/pipeline.py`：
  - 移除 `run()` 内重复的 `import numpy as np`
  - 为"帧落后播放进度被丢弃"增加 debug 日志，便于后续排查画面停滞

### 验证
- `pytest -q`：21 passed（修复前 2 failed：test_vad_endpoint_triggers_final_callback / test_max_segment_forces_interim_recognition）
- 两个修改文件 compile 语法检查通过

### 备注
- 若修复后画面仍偶发停滞，可开启 `app.debug: true` 观察"丢弃过期帧"日志，确认是否为 MuseTalk 推理速度慢于实时播放导致帧被全部丢弃。

## 2026-08-17 二次排查：窗口仍静止的真正根因（CUDA 显存不足）

### 现象
修复 ASR 崩溃后窗口仍静止。用诊断脚本实际加载 MuseTalk 并喂入 3 秒合成音频验证：
- `batch_size=8`：UNet 推理时 `CUDA error: out of memory`，0 帧生成，主窗口一直停留在启动画面
- `batch_size=2`：正常生成 72 帧（3 秒音频），帧间隔 0.0414s≈24fps，帧尺寸 1084×960

### 根因
- GPU 为 NVIDIA GeForce RTX 5060，显存 8GB
- MuseTalk v15 配置的 `batch_size=8` 使 UNet 推理峰值显存超过 8GB → CUDA OOM
- MuseTalk 帧生成线程每块音频都抛 OOM 并记录 error，但主线程仍显示启动画面 → 画面静止

### 修复
- `config.yaml`：`musetalk.batch_size` 8 → 2，补充注释说明 8GB 显存建议 2~4
- 附带改进（`takemh/pipeline.py`、`main.py`）：
  - 渲染循环不再无脑丢弃落后帧：先跳过队列中明显落后的旧帧，队列空时渲染最新可用帧，保证 MuseTalk 慢于实时时画面仍持续更新
  - `app.debug: true` 时日志级别提升为 DEBUG，便于观察"渲染落后帧"与"播放音频块"日志
- 上一轮已修复 ASR `_total_samples_processed` 未初始化导致的采集线程崩溃

### 验证
- `pytest -q`：21 passed
- 诊断脚本：batch_size=2 下 MuseTalk 正常产帧（72 帧 / 3 秒音频，fps≈24）

### 备注
- 若画面仍比声音滞后，可观察 DEBUG 日志中"渲染落后帧 ts=... played=..."，属于 MuseTalk 生成速度低于实时播放，可尝试调大 `chunk_seconds` 或继续调小 `batch_size`。
- 若想提升吞吐可尝试 `batch_size=4`，但需注意 TTS(CosyVoice) 与 MuseTalk 同时占显存。

## 2026-08-17 三次排查：TTS 合成失败（prompt_wav 传参类型错误）

### 现象
修复 MuseTalk 显存后组合诊断（CosyVoice TTS + MuseTalk）发现 TTS 合成一直报错：
`TTS 合成失败: Invalid file: array([...])`，0 秒音频产出，播放/MuseTalk 拿不到音频，画面依旧静止。

### 根因
- 官方 `cosyvoice/utils/file_utils.py::load_wav()` 内部使用 `torchaudio.load(wav, backend='soundfile')`，只接受**文件路径**或官方 `load_wav()` 返回的 torch.Tensor。
- 原实现 `_check_prompt_audio()` 用 `librosa.load()` 把参考音频加载成 **numpy 数组**传入 `inference_zero_shot(prompt_wav=...)`，官方 load_wav 内部 `torchaudio.load(numpy)` 报 `Invalid file`。
- 官方 `example.py` 的用法是直接传**文件路径字符串**（`'./asset/zero_shot_prompt.wav'`）。

### 修复
- `takemh/tts_cosyvoice.py`：`_check_prompt_audio()` 仅用 librosa 校验 3~10 秒时长，`self._prompt_wav` 保留**文件路径字符串**，由官方 load_wav 内部加载。
- `tests/test_tts_cosyvoice.py`：同步还原为 mock librosa 的短音频校验测试。

### 验证（组合诊断，真实模型）
- CosyVoice 合成成功：日志 `synthesis text ...`、`yield speech len 1.36/2.0/4.0/4.0/4.0/1.44`，rtf 1.8~4.5
- MuseTalk（batch_size=2）81 秒内收到 150 帧，帧 ts 0.000~6.200，尺寸 1084×960
- GPU 峰值约 4140 MB / 保留 5996 MB（RTX 5060 8GB 无 OOM）
- `pytest -q`：21 passed

### 本轮累计修复（窗口静止问题闭环）
1. ASR `_total_samples_processed` 未初始化 + 重复累加 → 采集线程崩溃（无识别文本）
2. MuseTalk `batch_size=8` 在 8GB 显存 CUDA OOM → 0 帧
3. TTS `prompt_wav` 传 numpy 数组 → 官方 load_wav 报 Invalid file，无音频
4. 渲染循环无脑丢弃落后帧 → 画面停留启动画面（改为跳过旧帧、队列空时渲染最新帧）
5. `app.debug: true` 时日志级别提升为 DEBUG，便于观察"渲染落后帧"/"播放音频块"日志

## 2026-08-18 四次排查：ASR 流式 VAD 逻辑错误（最终根因）

### 现象
前几轮修复后窗口仍静止。用真实模型验证发现 ASR 从不输出识别结果。

### 根因
- fsmn-vad 流式模式（is_final=False）**只报告语音开始事件 `[beg, -1]`**，语音结束事件只在 `is_final=True` 时一次性给出（实测：流式仅输出 `[[220,-1]]`，is_final 才输出 `[[220,1890]]`）。
- 原手写 VAD 逻辑按 `[beg,end]` 处理，`end=-1` 被换算成负样本，语音段永不累积；fsmn-vad 默认静音阈值 2000ms 也不适合实时对话。
- 真实识别验证（assets/prompt.wav）确认：重写后识别出 `希望你以后能够做的比我还好哟。😊`，on_final 正常触发。

### 修复
- 重写 `takemh/asr_sensevoice.py`：改用 FunASR 官方 `DynamicStreamingVAD`（`funasr.models.fsmn_vad_streaming.dynamic_vad`）做流式端点检测；维护 float32 音频缓冲，按 VAD 确认的 `[start_ms, end_ms]` 切段交给 SenseVoice 识别。
- `DynamicStreamingVAD.finalize()` 在 fsmn-vad 处理进行中语音段时传入 10ms 空块会触发 `stack expects a non-empty TensorList`，改为传入完整静音块 + `is_final=True` 收尾。
- 修复 `_flush_segment_ms` 缓冲切片顺序 bug：必须先切出语音段再释放缓冲（此前先覆盖缓冲再切片，取到错误数据；单测 FakeAsr 忽略输入内容故未暴露）。
- `config.py` / `config.yaml`：新增 `vad_silence_ms=600`（语音结束静音阈值）与 `max_buffer_seconds=30`（缓冲上限）。
- `main.py`：stdout/stderr 设为 UTF-8，避免识别文本含 emoji 时 GBK 控制台报错。
- 重写 `tests/test_asr_sensevoice.py`：mock DynamicStreamingVAD，覆盖段识别回调、短段忽略、finalize 收尾与识别参数。

### 验证
- 真实模型（CPU）：SenseVoice + DynamicStreamingVAD 识别 `assets/prompt.wav` → `[FINAL] 希望你以后能够做的比我还好哟。😊`（1 条）
- `pytest -q`：22 passed
- 语法检查通过

### 窗口静止问题完整修复链
1. ASR `_total_samples_processed` 未初始化 + 重复累加 → 采集线程崩溃
2. ASR 手写 VAD 流式逻辑错误（fsmn-vad 只报开始事件）→ on_final 永不触发 ← 本次
3. MuseTalk `batch_size=8` 在 8GB 显存 CUDA OOM → 0 帧
4. TTS `prompt_wav` 传 numpy 数组 → 官方 load_wav 报 Invalid file
5. 渲染循环无脑丢弃落后帧 → 画面停留启动画面

## 2026-08-18 五次排查：播放线程被 push_audio 阻塞 + MuseTalk 性能剖析

### 现象
用户提供运行日志：`渲染落后帧 ts=18.74~20.08 played=21.11`，播放进度 played 长时间停滞。分析：
- 帧其实在持续渲染（画面在动），但播放线程卡住导致声音断续、音画不同步。
- 根因：播放线程 `self.musetalk.push_audio(chunk)` 使用阻塞 put，MuseTalk 消费慢导致音频队列满，播放线程卡在 push 上无法播放后续音频。

### 修复
- `takemh/musetalk_driver.py`：`push_audio` 改为非阻塞（put_nowait），队列满时丢弃最旧音频块并把丢弃时长计入已消费时间，保证帧时间轴连续、播放线程不阻塞。
- `takemh/pipeline.py`："渲染落后帧"日志 1 秒限流，避免 DEBUG 刷屏。

### MuseTalk 性能剖析（RTX 5060 65W）
- 1 秒音频块处理约 4~5.5s：VAE decode 2.43s（最大）、Whisper 特征提取 1.48s、UNet 1.14s、Blend 0.49s。
- 增大 chunk_seconds（5s）无明显提升（约 x0.11 实时）；降低 fps 到 10 也无明显提升（约 x0.12）→ 功耗墙限制。
- 结论：该 GPU 无法让 MuseTalk 实时，画面会滞后于声音但持续更新；建议保持 batch_size=2 稳妥，若追求吞吐可试 batch_size=4（注意显存峰值约 7GB，TTS 共存时有 OOM 风险）。

### 验证
- `pytest -q`：22 passed
- 语法检查通过

## 2026-08-18 五次排查补充：LLM 可用性确认 + UI 状态反馈 + 测试模式增强

### 确认
- LLM（dashscope + deepseek-v4-flash）真实可用：非流式响应 4.8s，流式首 token 1.22s、总耗时 1.61s。
- 端到端各环节均以真实模型验证可用：ASR 识别 prompt.wav ✓、TTS 合成 ✓、MuseTalk 产帧 ✓、LLM 流式 ✓。
- 端到端响应慢源于硬件：TTS rtf≈2.86、MuseTalk 约 x0.1~0.4 实时（RTX 5060 65W）。

### 新增改进
- `takemh/pipeline.py`：窗口标题实时显示状态（等待语音输入/识别完成，正在回复/正在合成语音/播放中），用户可直观判断系统是否在工作。
- `takemh/audio_file.py`：`WavFileInputStream` 支持任意采样率 WAV，自动用 scipy 重采样到 16k（避免 librosa/numba 首次 JIT 卡顿），`--test-audio` 可直接使用 `assets/prompt.wav`。

### 无麦克风验证方法
```bash
python main.py config.yaml --test-audio assets/prompt.wav
```
会识别 prompt.wav 中的人声 → 走 LLM/TTS/MuseTalk 全链路 → 播放合成语音并驱动画面。

## 2026-08-18 补充：窗口标题中文乱码修复

- 现象：窗口标题状态用中文（如"等待语音输入..."）在 OpenCV 窗口显示乱码。
- 修复：`takemh/pipeline.py` 窗口标题状态改为英文 ASCII（Loading models... / Waiting for speech... / Replying... / Synthesizing... / Speaking...），避免编码问题。
- `pytest -q`：22 passed；语法检查通过。

## 2026-08-18 补充：音频听不清修复（参考文本匹配 + 音量归一化）

### 现象
- 用户反馈音频听不清。
- 排查：`config.yaml` 的 `prompt_text`（"你好呀！我是开发者的数字克隆人..."）与实际基准音频内容不匹配，导致 CosyVoice 零样本克隆音质差。
- 用户更换了 `assets/prompt.wav`（16k 单声道，11.69s，RMS -29.4 dBFS），但 11.69s 超过 `max_prompt_seconds=10` 会导致 TTS 加载直接报错。

### 修复
- `config.yaml`：
  - `prompt_text` 更新为与新基准音频匹配："大家好，此段是该项目的基准语音文件。今天我们来聊一下人工智能的发展，以及它对人类社会的影响，请多关照。"
  - `max_prompt_seconds` 10 → 13（容忍 11.69s 基准音频）。
- `takemh/tts_cosyvoice.py`：`_to_pcm16` 添加响度归一化（RMS 低于 0.15 时提升到约 -16.5dBFS），解决基准音频音量低导致的输出偏小。
- 实测合成输出：峰值 0.805，RMS 0.149（-16.5 dBFS），试听文件 `assets/tts_test_out.wav`（已被 .gitignore 忽略）。

### 备注
- 日志发现 onnxruntime 只有 CPU 提供商（CosyVoice 的 prompt 特征提取 ONNX 组件走 CPU），是 TTS 慢（rtf≈1~2.8）的原因之一；如需加速可安装与 CUDA 匹配的 onnxruntime-gpu。

## 2026-08-18 优化：TTS prompt 特征缓存（多句回复提速）

### 问题
官方 `inference_zero_shot` 对回复切句后的每一句都重新提取 prompt 的文本 token、
speech token/feat、音色 embedding（约 1~2s/句），对同一基准音频是重复计算。

### 优化
- `takemh/tts_cosyvoice.py`：
  - 新增 `_get_prompt_features()`：用官方 `frontend._extract_text_token / _extract_speech_feat / _extract_speech_token / _extract_spk_embedding` 提取并缓存 prompt 特征（含 24k 的 speech_feat/speech_token 2:1 裁剪，与官方 frontend_zero_shot 一致）。
  - `_synthesize_sentence()` 改为：`text_normalize` 切句 → 每句复用缓存的 prompt 特征 + `model.tts(**model_input, stream=True)` 合成，不再重复提取。
- `tests/test_tts_cosyvoice.py`：FakeEngine 补充 FakeFrontend / FakeModel，覆盖新链路。

### 验证
- 真实模型合成成功（多句文本，RMS -16.5 dBFS 归一化正常）。
- `pytest -q`：22 passed。

### 已知限制
- `model.tts`（LLM + flow + HiFTNet）在 RTX 5060 65W 上 rtf≈1~2.8，为硬件功耗墙限制；prompt 特征缓存只省每句 1~2s 的固定开销。
- onnxruntime 仅 CPU 提供商（speech_tokenizer 走 CPU），安装 onnxruntime-gpu 可进一步加速 prompt 特征提取，但收益有限（只影响首句前的一次提取），暂未安装。

## 2026-08-18 代码注释完善

- `takemh/config.py`：为全部配置 dataclass 添加类 docstring，并为每个字段补充行内语义注释（设备/采样率/静音阈值/队列容量等）。
- `main.py`：为 `main()` 补充 docstring 与启动流程注释（UTF-8 输出、参数解析、配置加载、LLM 密钥校验、启动/运行/释放）。
- `scripts/preflight.py`：补充 `_module_missing` / `_is_remote` / `_add_local_path` / `check_runtime_config` 的 docstring。
- `takemh/pipeline.py`：补充队列/播放进度/状态字段注释，模型加载顺序说明（GPU 模型先加载避免显存峰值叠加），`_set_status` docstring。
- `takemh/musetalk_driver.py`：补充帧生成线程分块缓冲注释，`_get_and_add_consumed` 的时间轴说明。
- `takemh/llm_openai.py`：`_append_history` docstring（保留系统提示 + 最近 N-1 条）。
- `takemh/tts_cosyvoice.py`：合成线程逐句合成/句间静音/流终止哨兵注释。

验证：`pytest -q` 22 passed；全部 19 个 .py 文件 compile 语法检查通过。

## 2026-10-23 API Key 命令行输入与稳定性优化

### API Key 安全调整
- 删除 `config.yaml` 中的明文 API Key，不再推荐把密钥写入配置文件。
- `main.py` 新增 `--api-key` 参数，并支持 `TAKE_MH_LLM_API_KEY` / `OPENAI_API_KEY` 环境变量。
- 未通过参数或环境变量提供密钥时，在交互式终端中使用 `getpass` 隐藏输入。
- 非交互环境缺少密钥时，在加载任何模型前快速失败并返回退出码 2。
- 保留旧配置字段的兼容读取，但启动时会提示不要继续使用明文配置。

### 自主修复
- `main.py` 同时支持 `--config config.yaml` 和 `main.py config.yaml`，并延迟导入重型模型模块。
- `SenseVoiceASR` 实际应用 `asr.vad_time_unit_ms`，修复 VAD 时间戳单位配置无效的问题。
- `WavFileInputStream` 不再覆盖未关闭的文件句柄。
- `SentenceSplitter` 保留句末标点，改善 TTS 韵律。
- TTS 文本队列和音频队列写入支持超时/停止中断，减少关闭时永久阻塞。
- MuseTalk 帧队列写入支持停止中断，并加强音频队列满时的哨兵处理。
- 调整停止顺序，先停止播放线程再停止 MuseTalk，避免关闭竞态。
- 播放线程异常时主动停止应用，不再静默留下失效线程。
- `requirements.txt` 将 OpenCV 限制在 4.x，并明确 Torch/TorchAudio 2.7.x 匹配关系。
- `.gitignore` 新增 `.env`、`.env.*` 与 `config.local.yaml`。
- 完整重写 `README.md`，覆盖安装、API Key、模型准备、启动、测试、性能调优和故障排查。

### 验证
- `python -m pytest -q -p no:cacheprovider`：31 passed。
- `python -m compileall -q main.py takemh scripts tests`：通过。
- `python scripts/preflight.py --config config.yaml`：0 错误；缺少运行时 API Key 时按预期给出警告。
- `python main.py --config config.yaml`：无密钥且非交互环境时返回 2，未加载模型。

### 待用户处理
- 原 `config.yaml` 中出现过的真实 API Key 应立即在服务商控制台吊销并重新生成。
