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
    - `src/wildcamtools/lib`: Core business logic (motion, rtsp, segments, etc.).
    - `src/wildcamtools/cli`: CLI tool implementation. Avoid adding business logic here, try to keep it to only code related to the CLI.
    - `deployment/`: Camera-side setup scripts.
    - `tests/`: Pytest suite with binaries in `tests/bin` (e.g., mediamtx).

## Conventions & Quirks

- **Python Version**: Strict `==3.13.*` and use appropriate language features when applicable
- **OpenCV**: Uses `opencv-contrib-python-headless` to avoid GUI dependencies; `uv` override prevents `opencv-python` installation.
- **Dependencies**: Managed via `uv` (checked via `uv.lock`).
    - AI dependencies (openai, ollama) can be used only in `src/wildcamtools/lib/ai`.
    - CLI dependencies (typer, click) can be used only in `src/wildcamtools/cli`.
    - persistence dependencies (sqlalchemy, sqlmodel) can be used only in `src/wildcamtools/lib/persistence`.
- **Verification Order**: `ruff format` $\to$ `ruff check` $\to$ `mypy` $\to$ `pytest`.
- **Logging**: Use python logging module and %-string formatting.
- **Tests**: Write tests to ensure new functionallity is mainitained in future. When tests fail, assume the test is the correct behaviour modify the code rather than test unnless you can see that the test is flawed.
- **Code review**: After each significant change, Use a code-review subagent to review changes, then use another sub-agent to fix any problems identified, then repeat until review passes.


## Best practices
- **Lazy imports**: Imports that are not at the top of a python file are to be avoided. Prefer to create additional files and refactor code to avoid circular dependency issues, rather than creating lazy imports.
- **After editing a python file**: Use `uv run ruff format {filename}` after each edit to ensure that python files are correctly formatted.
