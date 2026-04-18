# Agent Guide: WildCamTools

## High-Signal Commands
- Run tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Typecheck: `uv run mypy src`
- CLI Entrypoint: `uv run wildcamtools` (or `python -m wildcamtools`)

## Architecture & Flow
- **Purpose**: Motion-detected wildlife clip generation with lookback.
- **Pipeline**: Camera $\to$ MediaMTX (RTSP) $\to$ FFMPEG Segmenter $\to$ Storage $\to$ Motion Detection $\to$ Clip Concat.
- **Project Structure**:
    - `src/wildcamtools/lib`: Core logic (motion, rtsp, segments, etc.).
    - `src/wildcamtools/cli`: CLI tool implementation.
    - `deployment/`: Camera-side setup scripts.
    - `tests/`: Pytest suite with binaries in `tests/bin` (e.g., mediamtx).

## Conventions & Quirks
- **Python Version**: Strict `==3.13.*`.
- **OpenCV**: Uses `opencv-contrib-python-headless` to avoid GUI dependencies; `uv` override prevents `opencv-python` installation.
- **Dependencies**: Managed via `uv` (checked via `uv.lock`).
- **Verification Order**: `ruff format` $\to$ `ruff check` $\to$ `mypy` $\to$ `pytest`.

## Best practices
- **Format after edit**: Use `uv run ruff format .` after each edit to ensure that files are correctly formatted.
- **Before commit**: Use `uv run prek -a` before committing to prevent trivial problems.
