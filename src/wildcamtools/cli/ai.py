import logging
from pathlib import Path
from typing import Annotated

import typer

from wildcamtools.lib.ai.batch_processing import (
    BatchPipelineOutput,
    discover_video_files,
    run_batch_pipeline,
)
from wildcamtools.lib.ai.pipeline_config import AiPipelineConfig
from wildcamtools.lib.ai.pipeline_evaluation import evaluate_ai_pipeline

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def run(
    config: Annotated[Path, typer.Argument(metavar="CONFIG", help="Path to JSON configuration file")],
    video: Annotated[Path, typer.Argument(metavar="VIDEO")],
    output: Annotated[Path | None, typer.Option(help="Optional output file for JSON result")] = None,
) -> None:
    if not config.exists():
        typer.secho(f"Error: Config file not found: {config}", err=True)
        raise typer.Exit(code=1)
    if not config.is_file():
        typer.secho(f"Error: Config path is not a file: {config}", err=True)
        raise typer.Exit(code=1)
    if not video.exists():
        typer.secho(f"Error: Video file not found: {video}", err=True)
        raise typer.Exit(code=1)
    if not video.is_file():
        typer.secho(f"Error: Video path is not a file: {video}", err=True)
        raise typer.Exit(code=1)

    logger.info("Loading config from %s", config)
    pipeline_config = AiPipelineConfig.from_json(config)
    logger.info("Processing video %s", video)
    pipeline = pipeline_config.create_pipeline()
    outcome = pipeline.run(video)
    logger.info("Pipeline execution completed")

    output_data = BatchPipelineOutput(config=pipeline_config, outcome=outcome)
    json_output = output_data.model_dump_json(indent=2)

    if output:
        output.write_text(json_output)
    else:
        typer.secho(json_output)


def _validate_run_evaluate_inputs(
    config: Path,
    labels: Path,
    video_dir: Path | None,
    max_workers: int | None,
    comparison_config: Path,
) -> None:
    """Validate inputs for run_evaluate command."""
    if not config.exists():
        typer.secho(f"Error: Config file not found: {config}", err=True)
        raise typer.Exit(code=1)
    if not config.is_file():
        typer.secho(f"Error: Config path is not a file: {config}", err=True)
        raise typer.Exit(code=1)
    if not labels.exists():
        typer.secho(f"Error: Labels file not found: {labels}", err=True)
        raise typer.Exit(code=1)
    if not labels.is_file():
        typer.secho(f"Error: Labels path is not a file: {labels}", err=True)
        raise typer.Exit(code=1)
    if video_dir is not None and not video_dir.exists():
        typer.secho(f"Error: Video directory not found: {video_dir}", err=True)
        raise typer.Exit(code=1)
    if video_dir is not None and not video_dir.is_dir():
        typer.secho(f"Error: Video path is not a directory: {video_dir}", err=True)
        raise typer.Exit(code=1)
    if max_workers is not None and max_workers < 1:
        typer.secho(f"Error: --max-workers must be at least 1, got {max_workers}", err=True)
        raise typer.Exit(code=1)
    if not comparison_config.exists():
        typer.secho(f"Error: Label comparison config file not found: {comparison_config}", err=True)
        raise typer.Exit(code=1)
    if not comparison_config.is_file():
        typer.secho(f"Error: Label comparison config path is not a file: {comparison_config}", err=True)
        raise typer.Exit(code=1)


@app.command()
def run_evaluate(
    config: Annotated[Path, typer.Argument(metavar="CONFIG", help="Path to JSON configuration file")],
    comparison_config_path: Annotated[
        Path,
        typer.Argument(metavar="COMPARISON_CONFIG", help="JSON config for label comparison"),
    ],
    labels: Annotated[Path, typer.Argument(metavar="LABELS", help="Path to JSONL labels file")],
    video_dir: Annotated[
        Path | None,
        typer.Option("-v", "--video-dir", help="Video directory (defaults to labels parent)"),
    ] = None,
    max_workers: Annotated[int | None, typer.Option("-w", "--max-workers", help="Maximum worker processes")] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            metavar="OUTPUT",
            help="Optional output JSON file for results",
        ),
    ] = None,
) -> None:
    """Evaluate AiPipeline against labeled videos using a JSON configuration file.

    Runs the configured pipeline on each video in the labels file and compares
    results against ground truth labels. Outputs summary statistics and detailed
    results as a single JSON object.

    By default, uses exact string matching for comparison. Provide --label-comparison-config
    to enable semantic matching using an LLM.
    """
    _validate_run_evaluate_inputs(config, labels, video_dir, max_workers, comparison_config_path)

    logger.info("Loading config from %s", config)
    logger.info("Loading labels from %s", labels)
    logger.info("Loading comparison config from %s", comparison_config_path)
    logger.info("Evaluating pipeline on labeled videos")

    summary = evaluate_ai_pipeline(
        config_path=config,
        labels_path=labels,
        video_dir=video_dir,
        max_workers=max_workers,
        comparison_config_path=comparison_config_path,
    )

    json_output = summary.model_dump_json(indent=2)

    if output:
        output.write_text(json_output)
    else:
        typer.secho(json_output)

    summary.print_summary()


def _validate_run_batch_inputs(config: Path, video_dir: Path, output_dir: Path, max_workers: int | None) -> None:
    """Validate inputs for run_batch command."""
    if not config.exists():
        typer.secho(f"Error: Config file not found: {config}", err=True)
        raise typer.Exit(code=1)
    if not config.is_file():
        typer.secho(f"Error: Config path is not a file: {config}", err=True)
        raise typer.Exit(code=1)
    if not video_dir.exists():
        typer.secho(f"Error: Video directory not found: {video_dir}", err=True)
        raise typer.Exit(code=1)
    if not video_dir.is_dir():
        typer.secho(f"Error: Video path is not a directory: {video_dir}", err=True)
        raise typer.Exit(code=1)
    if max_workers is not None and max_workers < 1:
        typer.secho(f"Error: --max-workers must be at least 1, got {max_workers}", err=True)
        raise typer.Exit(code=1)

    # Create output directory if it doesn't exist
    if not output_dir.exists():
        logger.info("Creating output directory: %s", output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)


def _report_batch_results(results: list) -> None:
    """Report batch processing results to the user."""
    success_count = sum(1 for r in results if r.error is None)
    error_count = sum(1 for r in results if r.error is not None)

    typer.secho("\nBatch processing completed:")
    typer.secho(f"  Successful: {success_count}")
    typer.secho(f"  Errors: {error_count}")

    if error_count > 0:
        typer.secho("\nFailed videos:", err=True)
        for result in results:
            if result.error:
                typer.secho(f"  - {result.video_path}: {result.error}", err=True)
        raise typer.Exit(code=1)


@app.command()
def run_batch(
    config: Annotated[Path, typer.Argument(metavar="CONFIG", help="Path to JSON configuration file")],
    video_dir: Annotated[Path, typer.Argument(metavar="VIDEO_DIR", help="Directory containing video files")],
    output_dir: Annotated[Path, typer.Argument(metavar="OUTPUT_DIR", help="Directory for JSON output files")],
    max_workers: Annotated[
        int | None,
        typer.Option("-w", "--max-workers", help="Maximum worker processes (default: CPU count)"),
    ] = None,
    recursive: Annotated[bool, typer.Option("-r", "--recursive", help="Search recursively for video files")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite existing output files")] = False,
) -> None:
    """Run AI pipeline on multiple videos in parallel with resume capability.

    Discovers video files in VIDEO_DIR and processes them using the configured AI pipeline.
    Output JSON files are written to OUTPUT_DIR with mirrored directory structure.

    By default, skips videos that already have output JSON files (resume capability).
    Use --overwrite to reprocess videos even if output exists.

    Example:
        wildcamtools ai run-batch config.json wildlife/ results/ --max-workers 4

    """
    _validate_run_batch_inputs(config, video_dir, output_dir, max_workers)

    logger.info("Loading config from %s", config)
    pipeline_config = AiPipelineConfig.from_json(config)

    logger.info("Discovering video files in %s", video_dir)
    video_files = discover_video_files(video_dir, recursive=recursive)

    if not video_files:
        typer.secho("No video files found", err=True)
        raise typer.Exit(code=1)

    logger.info("Found %d video files", len(video_files))

    results = run_batch_pipeline(
        video_paths=video_files,
        video_dir=video_dir,
        output_dir=output_dir,
        pipeline_config=pipeline_config,
        max_workers=max_workers,
        skip_existing=not overwrite,
    )

    _report_batch_results(results)
