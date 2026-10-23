# Repository Guidelines

> Always respond in chinese-simplified.
> Before continuing, read TASK_LOG.md and follow its current progress.

## Project Structure

Keep the runtime code small and explicit:

- `takemh/` contains application modules: ASR, LLM, TTS, MuseTalk integration, audio IO, and the streaming pipeline.
- `config.yaml` is the main runtime configuration file.
- `main.py` is the entry point and should remain a thin launcher.
- `scripts/` is reserved for model setup, audio extraction, and operational tooling.
- `tests/` is the expected location for unit tests.

## Build, Test, and Development Commands

- `python -m py_compile main.py takemh/*.py` checks syntax without running model code.
- `python main.py --config config.yaml` starts the real-time pipeline with the configured models.
- `pytest -q` runs the test suite. If tests are added, keep them free of model weights and network calls.
- `pip install -r requirements.txt` installs project dependencies. Model weights and repositories are provisioned separately according to each official project's instructions.

## Coding Style

- Use 4 spaces for indentation; do not use tabs.
- Target Python 3.10 syntax and standard library behavior.
- Use `snake_case` for functions, variables, and modules; use `UpperCamelCase` for classes; use `UPPER_SNAKE_CASE` for constants.
- Keep modules decoupled: each model wrapper should expose clear methods such as `load()`, `stream_*()` or `transcribe_*()`.
- Add comments only where behavior is non-obvious. Prefer names that make intent clear.
- Preserve line length around 100 characters and split long function calls across lines.

## Testing Guidelines

- Use `pytest` and `unittest.mock` for unit tests.
- Test parsing, queue behavior, text segmentation, config loading, and streaming contracts without launching models.
- Name tests after the behavior they verify, for example `test_sentence_buffer_flushes_on_punctuation.py`.
- Do not require GPU, microphones, API credentials, or large model files in CI tests.

## Commit and Pull Request Guidelines

- Write commits in the imperative mood, for example `Add CosyVoice streaming wrapper`.
- Keep each commit focused on one module, fix, or behavior.
- Pull requests must include a description of the change, affected modules, and manual verification if runtime behavior changed.
- Link related issues when present. For model or UI changes, include a short description of local test results rather than screenshots unless the change is visual.
- Do not commit downloaded model weights, temporary audio files, or output videos. Add them to `.gitignore`.

## Agent-Specific Instructions

Do not invent model APIs. Every model call must come from the official SenseVoice, CosyVoice, MuseTalk, or OpenAI-compatible SDK interface. If the SDK is not installed or the interface is uncertain, stop and verify before changing model code.
always respond in chinese-simplified
