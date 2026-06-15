# Agent Guide: WildCamTools

## High-Signal Commands

- Install deps: `uv sync --locked --all-extras --dev`
- Run tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Typecheck: `uv run mypy src`
- CLI: `uv run wildcamtools` or `python -m wildcamtools`
- System deps (for tests): `ffmpeg`, `libblas-dev`

## Architecture & Flow

- **Purpose**: Motion-detected wildlife clip generation with lookback.
- **Pipeline**: Camera → MediaMTX (RTSP) → FFMPEG Segmenter → Storage → Motion Detection → Clip Concat.
- **Project Structure**:
    - `src/wildcamtools/lib`: Core business logic (motion, rtsp, segments, etc.).
    - `src/wildcamtools/cli`: CLI commands only (typer subcommands). No business logic.
    - `src/wildcamtools/web`: Web UI components (streamlit).
    - `deployment/`: Camera-side systemd setup scripts.
    - `tests/`: Pytest suite; `tests/bin` contains test binaries (e.g., mediamtx mock).

## Conventions & Quirks

- **Python Version**: Strict `==3.13.*`.
- **OpenCV**: Uses `opencv-contrib-python-headless`; `uv` override blocks `opencv-python`.
- **Dependency Boundaries** (enforced by convention):
    - AI deps (openai, ollama): only in `src/wildcamtools/lib/ai`.
    - CLI deps (typer, click): only in `src/wildcamtools/cli`.
    - Persistence deps (sqlalchemy, sqlmodel): only in `src/wildcamtools/lib/persistence`.
- **Verification Order**: `ruff format` → `ruff check` → `mypy` → `pytest`.
- **Logging**: Use `logging` module with %-formatting (not f-strings).
- **Tests**: When tests fail, assume the test is correct and fix the code unless the test is clearly flawed.
- **Code Review**: After significant changes, run the code-review subagent, fix issues, then re-review until it passes.
- **Imports**: No lazy imports. Refactor to avoid circular dependencies instead.

## CLI Structure

The CLI uses `typer` with subcommands registered in `src/wildcamtools/cli/__init__.py`:
- `motion`, `segment`, `watch`, `rtsp`, `frames`, `label`, `ai`, `results`, `db`, `rescale`, `perftest`

Add new commands by creating a module in `src/wildcamtools/cli/` and registering it in `__init__.py`.
