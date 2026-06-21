import logging
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel

from wildcamtools.lib.ai.pipeline import CombinedPipelineOutcome, RichResultPipelineOutcome
from wildcamtools.lib.ai.pipeline_config import AiPipelineConfig
from wildcamtools.lib.persistence.database import create_engine_and_tables, get_session
from wildcamtools.lib.persistence.filename_datetime import infer_recorded_at
from wildcamtools.lib.persistence.repository import save_pipeline_run

app = typer.Typer()
logger = logging.getLogger(__name__)


class PipelineRunOutput(BaseModel):
    """Output format from ai run command."""

    config: AiPipelineConfig
    outcome: RichResultPipelineOutcome | CombinedPipelineOutcome


def _is_video_file(path: Path) -> bool:
    """Check if a path is a video file based on extension."""
    return path.is_file() and path.suffix.lower() in {".mp4"}


def _is_result_file(path: Path) -> bool:
    """Check if a path is a result JSON file based on extension."""
    return path.is_file() and path.suffix.lower() == ".json"


def _find_matching_pairs(video_dir: Path, result_dir: Path) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
    """Find matching video and result file pairs based on relative paths.

    Returns:
        Tuple of (matches, video_warnings, result_warnings)
        - matches: list of (video_path, result_path) tuples
        - video_warnings: list of warning messages for videos without results
        - result_warnings: list of warning messages for results without videos

    """
    video_files = set()
    result_files = set()

    for video_path in video_dir.rglob("*"):
        if _is_video_file(video_path):
            video_files.add(video_path.relative_to(video_dir))

    for result_path in result_dir.rglob("*"):
        if _is_result_file(result_path):
            result_files.add(result_path.relative_to(result_dir))

    matches = []
    video_warnings = []
    result_warnings = []

    for rel_video_path in video_files:
        rel_result_path = rel_video_path.with_suffix(".json")
        if rel_result_path in result_files:
            matches.append((video_dir / rel_video_path, result_dir / rel_result_path))
        else:
            video_warnings.append(f"Warning: No matching result file for video: {rel_video_path}")

    for rel_result_path in result_files:
        expected_video_path = rel_result_path.with_suffix(".mp4")
        if expected_video_path not in video_files:
            result_warnings.append(
                f"Warning: No matching video file for result: {rel_result_path} (expected: {expected_video_path})",
            )

    return matches, video_warnings, result_warnings


def _import_single_result(
    input_file: Path,
    video: Path,
    database: Path,
    engine: Any | None = None,
    filename_date_format: str | None = None,
) -> bool:
    """Import a single result file into the database.

    Returns True on success, False on failure.
    """
    logger.info("Loading result from %s", input_file)
    try:
        content = input_file.read_text()
        data = PipelineRunOutput.model_validate_json(content)
    except Exception as e:
        typer.secho(f"Error: Failed to parse JSON {input_file}: {e}", err=True)
        return False

    if engine is None:
        connection_string = f"sqlite:///{database.absolute()}"
        logger.info("Connecting to database: %s", database.absolute())
        try:
            engine = create_engine_and_tables(connection_string)
        except Exception as e:
            typer.secho(f"Error: Failed to connect to database: {e}", err=True)
            return False

    recorded_at = infer_recorded_at(video.name, filename_date_format)
    if recorded_at is not None:
        logger.info("Inferred recorded_at=%s for filename=%s", recorded_at, video.name)

    logger.info("Importing result into database")
    with get_session(engine) as session:
        try:
            run = save_pipeline_run(
                session,
                video.absolute(),
                data.config,
                data.outcome,
                recorded_at=recorded_at,
            )
            session.commit()
            typer.secho(f"Successfully imported pipeline run: {video} (id={run.id})", fg="green")
        except Exception as e:
            session.rollback()
            typer.secho(f"Error: Failed to save {video} to database: {e}", err=True)
            return False
        else:
            return True


def _import_single_file_mode(
    input_path: Path,
    video_path: Path,
    database: Path,
    filename_date_format: str | None,
) -> None:
    """Handle single file import mode."""
    if not input_path.is_file():
        typer.secho(f"Error: Input path is not a file: {input_path}", err=True)
        raise typer.Exit(code=1)
    if not video_path.is_file():
        typer.secho(f"Error: Video path is not a file: {video_path}", err=True)
        raise typer.Exit(code=1)

    if input_path.suffix.lower() != ".json":
        typer.secho(f"Error: Input file must be a .json file: {input_path}", err=True)
        raise typer.Exit(code=1)

    if video_path.suffix.lower() != ".mp4":
        typer.secho(f"Error: Video file must be a .mp4 or .MP4 file: {video_path}", err=True)
        raise typer.Exit(code=1)

    success = _import_single_result(input_path, video_path, database, filename_date_format=filename_date_format)
    if not success:
        raise typer.Exit(code=1)


def _import_directory_mode(video_dir: Path, result_dir: Path, database: Path, filename_date_format: str | None) -> None:
    """Handle directory bulk import mode."""
    logger.info("Processing directories: %s and %s", video_dir, result_dir)
    matches, video_warnings, result_warnings = _find_matching_pairs(video_dir, result_dir)

    for warning in video_warnings:
        typer.secho(warning, fg="yellow")
    for warning in result_warnings:
        typer.secho(warning, fg="yellow")

    if not matches:
        typer.secho("Error: No matching video-result pairs found", err=True)
        raise typer.Exit(code=1)

    logger.info("Found %d matching pairs to import", len(matches))

    connection_string = f"sqlite:///{database.absolute()}"
    logger.info("Connecting to database: %s", database.absolute())
    try:
        engine = create_engine_and_tables(connection_string)
    except Exception as e:
        typer.secho(f"Error: Failed to connect to database: {e}", err=True)
        raise typer.Exit(code=1) from e

    success_count = 0
    error_count = 0

    for video_file, result_file in matches:
        if _import_single_result(
            result_file,
            video_file,
            database,
            engine,
            filename_date_format=filename_date_format,
        ):
            success_count += 1
        else:
            error_count += 1

    typer.secho(
        f"\nImport completed: {success_count} successful, {error_count} failed",
        fg="green" if error_count == 0 else "yellow",
    )

    if error_count > 0:
        raise typer.Exit(code=1)


@app.command(name="import")
def import_result(
    input_path: Annotated[
        Path,
        typer.Argument(
            metavar="INPUT",
            help="Path to JSON result file or directory of result files from ai run",
        ),
    ],
    video_path: Annotated[
        Path,
        typer.Argument(
            metavar="VIDEO",
            help="Path to video file or directory of video files that were processed",
        ),
    ],
    database: Annotated[
        Path,
        typer.Option("-d", "--database", help="SQLite database path (default: wildcamtools.db)"),
    ] = Path("wildcamtools.db"),
    filename_date_format: Annotated[
        str | None,
        typer.Option(
            "--filename-date-format",
            metavar="STRFTIME",
            help="Optional strftime pattern used to infer a 'recorded_at' timestamp from the video filename (e.g. '%Y%m%d%H%M%S').",
        ),
    ] = None,
) -> None:
    """Import pipeline result JSON file(s) into the database.

    If INPUT and VIDEO are files, imports a single result.
    If INPUT and VIDEO are directories, recursively matches files by relative path
    (e.g., videos/a.mp4 matches results/a.json) and imports all pairs.
    Mismatches are skipped with warnings.
    """
    if not input_path.exists():
        typer.secho(f"Error: Input path not found: {input_path}", err=True)
        raise typer.Exit(code=1)

    if not video_path.exists():
        typer.secho(f"Error: Video path not found: {video_path}", err=True)
        raise typer.Exit(code=1)

    input_is_dir = input_path.is_dir()
    video_is_dir = video_path.is_dir()

    if input_is_dir != video_is_dir:
        typer.secho(
            f"Error: Both paths must be files or both must be directories. "
            f"Got INPUT={input_path} ({'directory' if input_is_dir else 'file'}), "
            f"VIDEO={video_path} ({'directory' if video_is_dir else 'file'})",
            err=True,
        )
        raise typer.Exit(code=1)

    if input_is_dir:
        _import_directory_mode(video_path, input_path, database, filename_date_format)
    else:
        _import_single_file_mode(input_path, video_path, database, filename_date_format)
