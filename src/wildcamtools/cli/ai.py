import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import typer

from wildcamtools.lib.ai import AIEvaluation, Backend, SpeciesResult
from wildcamtools.lib.ai.evaluate import evaluate_frames
from wildcamtools.lib.ai.evaluate import evaluate_pipeline as evaluate_pipeline_lib
from wildcamtools.lib.ai.llm import create_analyser
from wildcamtools.lib.ai.pipeline_config import AiPipelineConfig
from wildcamtools.lib.ai.pipeline_evaluation import evaluate_ai_pipeline
from wildcamtools.lib.frames import CropPanConfig, FrameCreation, TilingConfig, create_frames
from wildcamtools.lib.pipeline import run_pipeline
from wildcamtools.lib.timing import Timer
from wildcamtools.lib.web.label import load_labels

app = typer.Typer()
logger = logging.getLogger(__name__)


def _read_prompt_file(prompt_file: Path | None) -> str:
    """Read prompt from file if provided, validating existence and content."""
    prompt_default = (
        """This is a video image from a UK garden near a river. Is there an animal in this image, if so what?"""
    )
    if prompt_file is None:
        return prompt_default
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    prompt = prompt_file.read_text()
    if not prompt.strip():
        logger.warning("Prompt file %s is empty, using default prompt", prompt_file)
        return prompt_default
    return prompt


def _validate_tiling_parameters(
    tiling_cols: int | None,
    tiling_rows: int | None,
    tiling_overlap: float | None,
) -> tuple[int | None, int | None, float | None]:
    """Validate tiling parameters and exit with error if invalid."""
    if tiling_cols is not None and tiling_cols < 1:
        typer.secho("Error: --tiling-cols must be at least 1", err=True)
        raise typer.Exit(code=1)
    if tiling_rows is not None and tiling_rows < 1:
        typer.secho("Error: --tiling-rows must be at least 1", err=True)
        raise typer.Exit(code=1)
    if tiling_overlap is not None and (tiling_overlap < 0.0 or tiling_overlap >= 1.0):
        typer.secho("Error: --tiling-overlap must be between 0.0 and 1.0", err=True)
        raise typer.Exit(code=1)
    if tiling_cols is not None and tiling_rows is None:
        tiling_rows = 1
    if tiling_rows is not None and tiling_cols is None:
        tiling_cols = 1
    return tiling_cols, tiling_rows, tiling_overlap


@app.command()
def analyze(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    model: Annotated[str, typer.Option(help="Model name to use")],
    backend: Annotated[Backend, typer.Option(help="Backend to use")] = Backend.LLAMACPP,
    url: Annotated[str, typer.Option(help="Base URL for the backend")] = "http://localhost:8080/v1",
    api_key: Annotated[str | None, typer.Option(help="API key for ollama backend")] = None,
    output: Annotated[Path | None, typer.Option(metavar="OUTPUT")] = None,
    prompt_file: Annotated[Path | None, typer.Option(help="Path to a text file containing the prompt")] = None,
) -> None:
    if not input_.exists():
        typer.secho(f"Error: Input path not found: {input_}", err=True)
        raise typer.Exit(code=1)
    if not input_.is_dir():
        typer.secho(f"Error: Input is not a directory: {input_}", err=True)
        raise typer.Exit(code=1)
    images = list(input_.iterdir())
    images = [i for i in images if not i.is_dir()]

    prompt = _read_prompt_file(prompt_file)

    timer = Timer()
    analyser = create_analyser(backend=backend, model=model, url=url, api_key=api_key)

    with timer:
        result_obj = analyser.message_with_schema(
            message=prompt,
            images=images,
            response_class=SpeciesResult,
        )
        result = result_obj.species_name

    typer.secho(f"Processed {len(images):d} in {timer.elapsed:.2f} sec")
    typer.secho(f"Result: {result}")

    if output:
        with open(output, "w") as outfile:
            json.dump({"result": result}, outfile)


@app.command()
def pipeline(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    model: Annotated[str, typer.Option(help="Model name to use")],
    fps: Annotated[
        float,
        typer.Option(help="Target frames per second (supports values < 1.0, e.g., 0.5 for 1 frame every 2 seconds)"),
    ] = 1.0,
    full_frames_dir: Annotated[Path, typer.Option(help="Directory for full-resolution frames")] = Path(
        "tmp_frames_full/"
    ),
    low_res_frames_dir: Annotated[Path, typer.Option(help="Directory for low-resolution frames")] = Path(
        "tmp_frames_low/"
    ),
    cropped_frames_dir: Annotated[Path, typer.Option(help="Directory for cropped frames")] = Path(
        "tmp_frames_cropped/"
    ),
    low_res_size: Annotated[tuple[int, int], typer.Option(help="Low resolution frame size (width,height)")] = (
        640,
        360,
    ),
    crop_expansion: Annotated[float, typer.Option(help="Bounding box expansion factor")] = 0.25,
    backend: Annotated[Backend, typer.Option(help="Backend to use")] = Backend.OLLAMA,
    url: Annotated[str, typer.Option(help="Base URL for the backend")] = "http://localhost:8080/v1",
    api_key: Annotated[str | None, typer.Option(help="API key for ollama backend")] = None,
) -> None:
    if not input_.exists():
        typer.secho(f"Error: Input file not found: {input_}", err=True)
        raise typer.Exit(code=1)
    if not input_.is_file():
        typer.secho(f"Error: Input is not a file: {input_}", err=True)
        raise typer.Exit(code=1)
    if fps is not None and fps <= 0:
        typer.secho(f"Error: FPS must be positive, got {fps}", err=True)
        raise typer.Exit(code=1)

    analyser = create_analyser(backend=backend, model=model, url=url, api_key=api_key)

    run_pipeline(
        video_path=input_,
        full_frames_dir=full_frames_dir,
        low_res_frames_dir=low_res_frames_dir,
        cropped_frames_dir=cropped_frames_dir,
        fps=fps,
        low_res_size=low_res_size,
        analyser=analyser,
        crop_expansion=crop_expansion,
    )

    typer.secho(f"Cropped frames saved to {cropped_frames_dir}")


@app.command()
def evaluate(
    labels: Annotated[Path, typer.Argument(metavar="LABELS")],
    model: Annotated[str, typer.Option(help="Model name to use")],
    backend: Annotated[Backend, typer.Option(help="Backend to use")] = Backend.LLAMACPP,
    url: Annotated[str, typer.Option(help="Base URL for the backend")] = "http://localhost:8080/v1",
    api_key: Annotated[str | None, typer.Option(help="API key for ollama backend")] = None,
    history: int = 30,
    threshold: int = 16,
    kernel_size: float = 0.02,
    x: int | None = None,
    y: int | None = None,
    fps: float | None = None,
    crop_expansion: float | None = None,
    crop_inertia: float | None = None,
    similarity_minimum: float | None = None,
    enable_croppan: Annotated[bool, typer.Option(help="Enable crop and pan")] = False,
    tiling_cols: Annotated[int | None, typer.Option(help="Number of horizontal tiles per frame")] = None,
    tiling_rows: Annotated[int | None, typer.Option(help="Number of vertical tiles per frame")] = None,
    tiling_overlap: Annotated[float | None, typer.Option(help="Overlap proportion between tiles (0.0-1.0)")] = None,
    prompt_file: Annotated[Path | None, typer.Option(help="Path to a text file containing the prompt")] = None,
) -> None:
    if not labels.exists():
        typer.secho(f"Error: Labels file not found: {labels}", err=True)
        raise typer.Exit(code=1)

    tiling_cols, tiling_rows, tiling_overlap = _validate_tiling_parameters(tiling_cols, tiling_rows, tiling_overlap)
    prompt = _read_prompt_file(prompt_file)

    labelled_data = load_labels(labels)
    video_directory = labels.resolve().parent

    crop_pan = None
    if enable_croppan:
        crop_pan = CropPanConfig(
            history=history,
            threshold=float(threshold),
            kernel_size=kernel_size,
            expansion=crop_expansion if crop_expansion is not None else 0.75,
            inertia=crop_inertia if crop_inertia is not None else 10.0,
            motion_type="mog",
        )

    tiling = None
    if tiling_cols is not None and tiling_rows is not None:
        tiling = TilingConfig(
            cols=tiling_cols,
            rows=tiling_rows,
            overlap=tiling_overlap if tiling_overlap is not None else 0.0,
        )

    correct_count = 0
    total_count = 0

    for filename in labelled_data:
        frame_directory = TemporaryDirectory()
        try:
            frame_creation = FrameCreation(
                filename=Path(filename),
                video_directory=video_directory,
                tmpdir=Path(frame_directory.name),
                x=x,
                y=y,
                fps=fps,
                similarity_minimum=similarity_minimum,
                crop_pan=crop_pan,
                tiling=tiling,
            )
            create_frames(frame_creation)

            evaluation = AIEvaluation(
                frame_directory=Path(frame_directory.name),
                label=labelled_data[filename],
                backend=backend,
                url=url,
                model=model,
                api_key=api_key if api_key else "API_KEY",
                prompt=prompt,
            )
            result = evaluate_frames(evaluation)
            logger.info("File %s correct? %s", filename, result.correct)

            if result.correct:
                correct_count += 1
            total_count += 1
        finally:
            frame_directory.cleanup()

    if total_count > 0:
        ratio = correct_count / total_count
        typer.secho(f"Overall: {correct_count} / {total_count} = {ratio:.4f}")


@app.command()
def evaluate_pipeline(
    labels: Annotated[Path, typer.Argument(metavar="LABELS")],
    model: Annotated[str, typer.Option(help="Model name to use")],
    backend: Annotated[Backend, typer.Option(help="Backend to use")] = Backend.OLLAMA,
    url: Annotated[str, typer.Option(help="Base URL for the backend")] = "http://localhost:8080/v1",
    api_key: Annotated[str | None, typer.Option(help="API key for ollama backend")] = None,
    fps: Annotated[float | None, typer.Option()] = None,
    low_res_size: Annotated[tuple[int, int], typer.Option(help="Low resolution frame size (width,height)")] = (
        640,
        360,
    ),
    crop_expansion: Annotated[float, typer.Option(help="Bounding box expansion factor")] = 0.25,
    crops_output_dir: Annotated[
        Path, typer.Option(help="Directory for cropped frames (subdirectory per video)")
    ] = Path("crops_output/"),
) -> None:
    if not labels.exists():
        typer.secho(f"Error: Labels file not found: {labels}", err=True)
        raise typer.Exit(code=1)
    if not labels.is_file():
        typer.secho(f"Error: Labels is not a file: {labels}", err=True)
        raise typer.Exit(code=1)

    if low_res_size[0] <= 0 or low_res_size[1] <= 0:
        typer.secho("Error: low_res_size must be positive", err=True)
        raise typer.Exit(code=1)

    summary = evaluate_pipeline_lib(
        labels_path=labels,
        model=model,
        backend=backend,
        url=url,
        api_key=api_key,
        fps=fps,
        low_res_size=low_res_size,
        crop_expansion=crop_expansion,
        crops_output_dir=crops_output_dir,
    )

    summary.print_summary()


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
    result = pipeline.run(video)
    logger.info("Pipeline execution completed")

    json_output = result.model_dump_json(indent=2)

    if output:
        output.write_text(json_output)
    else:
        typer.secho(json_output)


def _validate_run_evaluate_inputs(
    config: Path, labels: Path, video_dir: Path | None, max_workers: int | None, comparison_config: Path
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
        Path, typer.Argument(metavar="COMPARISON_CONFIG", help="JSON config for label comparison")
    ],
    labels: Annotated[Path, typer.Argument(metavar="LABELS", help="Path to JSONL labels file")],
    video_dir: Annotated[
        Path | None, typer.Option("-v", "--video-dir", help="Video directory (defaults to labels parent)")
    ] = None,
    max_workers: Annotated[int | None, typer.Option("-w", "--max-workers", help="Maximum worker processes")] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            metavar="OUTPUT",
            help="Optional output JSONL file for results (defaults to ./pipeline_evaluation_result.jsonl)",
        ),
    ] = None,
) -> None:
    """Evaluate AiPipeline against labeled videos using a JSON configuration file.

    Runs the configured pipeline on each video in the labels file and compares
    results against ground truth labels. Outputs summary statistics and optional
    detailed results to JSONL file.

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
        result_jsonl_path=output,
        comparison_config_path=comparison_config_path,
    )

    summary.print_summary()
