import json
import logging
import multiprocessing
from collections.abc import Sequence
from dataclasses import asdict
from multiprocessing.pool import AsyncResult
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, cast

import typer

from wildcamtools.lib.ai import AbstractAnalyser, AIEvaluation, Backend
from wildcamtools.lib.ai.evaluate import (
    evaluate_frames,
)
from wildcamtools.lib.ai.llamacpp import LlamaCppAnalyser
from wildcamtools.lib.ai.ollama import OllamaAnalyser
from wildcamtools.lib.frames import (
    CropPanConfig,
    FrameCreation,
    FrameCreationResult,
    TilingConfig,
    create_frames,
)
from wildcamtools.lib.pipeline import generate_frames_for_video, process_pipeline_result, run_pipeline
from wildcamtools.lib.timing import Timer
from wildcamtools.lib.web.label import load_labels

app = typer.Typer()
logger = logging.getLogger(__name__)


def _read_prompt_file(prompt_file: Path | None) -> str | None:
    """Read prompt from file if provided, validating existence and content."""
    if prompt_file is None:
        return None
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    prompt = prompt_file.read_text()
    if not prompt.strip():
        logger.warning("Prompt file %s is empty, using default prompt", prompt_file)
        return None
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


def _build_evaluate_parameters(
    kernel_size: float,
    threshold: int,
    history: int,
    crop_expansion: float | None,
    crop_inertia: float | None,
    enable_croppan: bool,
    x: int | None,
    y: int | None,
    fps: float | None,
    similarity_minimum: float | None,
    tiling_cols: int | None,
    tiling_rows: int | None,
    tiling_overlap: float | None,
) -> list[dict[str, Any]]:
    """Build list of parameter dictionaries for frame creation."""
    parameter_dicts: list[dict[str, Any]] = []

    # Build crop_pan config only if explicitly enabled
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

    # Build tiling config if tiling parameters are set
    tiling = None
    if tiling_cols is not None and tiling_rows is not None:
        tiling = TilingConfig(
            cols=tiling_cols,
            rows=tiling_rows,
            overlap=tiling_overlap if tiling_overlap is not None else 0.0,
        )

    parameter_dicts.append({
        "x": x,
        "y": y,
        "fps": fps,
        "similarity_minimum": similarity_minimum,
        "crop_pan": crop_pan,
        "tiling": tiling,
    })
    return parameter_dicts


def _process_frame_result(
    future: tuple[AsyncResult, str, TemporaryDirectory[str], dict[str, Any]],
    labelled_data: dict[str, Any],
    backend: Backend,
    url: str,
    model: str,
    api_key: str | None,
    prompt: str | None,
    counters: dict[tuple[Any, ...], list[int]],
) -> None:
    """Process a single frame result, evaluate it, and write to result file."""
    future_result, filename, frame_directory, parameter_dict = future
    frame_creation_result = cast(FrameCreationResult, future_result.get())

    label = labelled_data[filename]
    evaluated_result = evaluate_frames(
        AIEvaluation(
            frame_directory=Path(frame_directory.name),
            label=label,
            backend=backend,
            url=url,
            model=model,
            api_key=api_key if api_key else "API_KEY",
            prompt=prompt,
        )
    )
    logger.info("File %s correct? %s", filename, evaluated_result.correct)

    counter_key = tuple(
        sorted((k, v) for k, v in parameter_dict.items() if v is None or isinstance(v, int | float | str | bool))
    )
    if counter_key not in counters:
        counters[counter_key] = [0, 0]
    counters[counter_key][0] += 1 if evaluated_result.correct else 0
    counters[counter_key][1] += 1

    with open("result.jsonl", "a") as result_file:
        output_parameter_dict = {
            "x": parameter_dict["x"],
            "y": parameter_dict["y"],
            "fps": parameter_dict["fps"],
            "similarity_minimum": parameter_dict["similarity_minimum"],
            "crop_pan": asdict(parameter_dict["crop_pan"]) if parameter_dict["crop_pan"] is not None else None,
            "tiling": asdict(parameter_dict["tiling"]) if parameter_dict["tiling"] is not None else None,
            "filename": filename,
            "model": model,
        }
        result_dict = {
            "result": evaluated_result.correct,
            "raw_result": evaluated_result.raw_result,
            "label": label,
            "frame_count": frame_creation_result.frame_count,
        }
        output_dict = (
            output_parameter_dict,
            result_dict,
        )
        result_file.write(json.dumps(output_dict))
        result_file.write("\n")

    frame_directory.cleanup()


def _process_pipeline_result(
    frame_result: tuple[str, Sequence[Path], Sequence[Path]],
    labelled_data: dict[str, Any],
    crops_output_dir: Path,
    analyser: AbstractAnalyser,
    crop_expansion: float,
    counters: dict[tuple[Any, ...], list[int]],
) -> None:
    """Process pipeline result: AI crop detection, evaluation, and output.

    Runs in main process. Creates video subdirectory in crops_output_dir, runs AI detection,
    applies crops to full-res frames, evaluates result, and writes JSONL entry.
    """
    filename, _, _ = frame_result
    label = labelled_data[filename]

    correct, raw_result, crop_count = process_pipeline_result(
        frame_result=frame_result,
        labelled_data=labelled_data,
        crops_output_dir=crops_output_dir,
        analyser=analyser,
        crop_expansion=crop_expansion,
        filename=filename,
    )

    counter_key = ("crop_expansion", crop_expansion)
    if counter_key not in counters:
        counters[counter_key] = [0, 0]
    counters[counter_key][0] += 1 if correct else 0
    counters[counter_key][1] += 1

    with open("result.jsonl", "a") as result_file:
        output_parameter_dict = {
            "filename": filename,
            "model": analyser.model,
            "backend": analyser.backend.value,
            "crop_expansion": crop_expansion,
        }
        result_dict = {
            "result": correct,
            "raw_result": raw_result,
            "label": label,
            "crop_count": crop_count,
        }
        output_dict = (output_parameter_dict, result_dict)
        result_file.write(json.dumps(output_dict))
        result_file.write("\n")

    logger.info("Video %s: correct=%s", filename, correct)


def _print_evaluation_summary(counters: dict[tuple[Any, ...], list[int]]) -> None:
    """Print evaluation summary statistics."""
    for counter_key in counters:
        successes = counters[counter_key][0]
        attempts = counters[counter_key][1]
        ratio = successes / attempts
        typer.secho(f"{counter_key}: {successes} / {attempts} = {ratio:.4f}")


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
    analyser: AbstractAnalyser
    match backend:
        case Backend.LLAMACPP:
            analyser = LlamaCppAnalyser(model=model, base_url=url, message=prompt)
        case Backend.OLLAMA:
            analyser = OllamaAnalyser(model=model, host=url, api_key=api_key, message=prompt)
        case _:
            raise ValueError(f"Unsupported backend: {backend}")

    with timer:
        result = analyser.analyze_video(images)

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
    prompt_file: Annotated[Path | None, typer.Option(help="Path to a text file containing the prompt")] = None,
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

    prompt = _read_prompt_file(prompt_file)

    analyser: AbstractAnalyser
    match backend:
        case Backend.LLAMACPP:
            analyser = LlamaCppAnalyser(model=model, base_url=url, message=prompt)
        case Backend.OLLAMA:
            analyser = OllamaAnalyser(model=model, host=url, api_key=api_key, message=prompt)
        case _:
            raise ValueError(f"Unsupported backend: {backend}")

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
    tiling_cols, tiling_rows, tiling_overlap = _validate_tiling_parameters(tiling_cols, tiling_rows, tiling_overlap)
    prompt = _read_prompt_file(prompt_file)

    labelled_data = load_labels(labels)
    video_directory = labels.resolve().parent

    parameter_dicts = _build_evaluate_parameters(
        kernel_size=kernel_size,
        threshold=threshold,
        history=history,
        crop_expansion=crop_expansion,
        crop_inertia=crop_inertia,
        enable_croppan=enable_croppan,
        x=x,
        y=y,
        fps=fps,
        similarity_minimum=similarity_minimum,
        tiling_cols=tiling_cols,
        tiling_rows=tiling_rows,
        tiling_overlap=tiling_overlap,
    )

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool() as pool:
        futures: list[tuple[AsyncResult, str, TemporaryDirectory[str], dict[str, Any]]] = []
        for parameter_dict in parameter_dicts:
            for filename in labelled_data:
                frame_directory = TemporaryDirectory()
                frame_creation = FrameCreation(
                    filename=Path(filename),
                    video_directory=video_directory,
                    tmpdir=Path(frame_directory.name),
                    **parameter_dict,
                )
                future = pool.apply_async(
                    create_frames,
                    args=(frame_creation,),
                )
                futures.append((future, filename, frame_directory, parameter_dict))

        counters: dict[tuple[Any, ...], list[int]] = {}
        for future_tuple in futures:
            _process_frame_result(
                future=future_tuple,
                labelled_data=labelled_data,
                backend=backend,
                url=url,
                model=model,
                api_key=api_key,
                prompt=prompt,
                counters=counters,
            )

    _print_evaluation_summary(counters)


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
    prompt_file: Annotated[Path | None, typer.Option(help="Path to a text file containing the prompt")] = None,
) -> None:
    if not labels.exists():
        typer.secho(f"Error: Labels file not found: {labels}", err=True)
        raise typer.Exit(code=1)
    if not labels.is_file():
        typer.secho(f"Error: Labels is not a file: {labels}", err=True)
        raise typer.Exit(code=1)

    prompt = _read_prompt_file(prompt_file)
    labelled_data = load_labels(labels)
    video_directory = labels.resolve().parent

    if low_res_size[0] <= 0 or low_res_size[1] <= 0:
        typer.secho("Error: low_res_size must be positive", err=True)
        raise typer.Exit(code=1)

    crops_output_dir.mkdir(parents=True, exist_ok=True)

    result_jsonl_path = Path("result.jsonl")
    result_jsonl_path.unlink(missing_ok=True)

    analyser: AbstractAnalyser
    match backend:
        case Backend.LLAMACPP:
            analyser = LlamaCppAnalyser(model=model, base_url=url, message=prompt)
        case Backend.OLLAMA:
            analyser = OllamaAnalyser(model=model, host=url, api_key=api_key, message=prompt)
        case _:
            raise ValueError(f"Unsupported backend: {backend}")

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool() as pool:
        frame_tmpdirs: list[TemporaryDirectory] = []
        futures: list[tuple[AsyncResult, str, TemporaryDirectory]] = []
        for filename in labelled_data:
            video_path = video_directory / filename
            if not video_path.exists():
                logger.warning("Video not found: %s", filename)
                continue
            frame_tmpdir = TemporaryDirectory()
            frame_tmpdirs.append(frame_tmpdir)
            future = pool.apply_async(
                generate_frames_for_video,
                args=(
                    filename,
                    video_directory,
                    fps,
                    low_res_size,
                    Path(frame_tmpdir.name),
                ),
            )
            futures.append((future, filename, frame_tmpdir))

        counters: dict[tuple[Any, ...], list[int]] = {}
        for future, _, frame_tmpdir in futures:
            try:
                frame_result = future.get()
                _process_pipeline_result(
                    frame_result=frame_result,
                    labelled_data=labelled_data,
                    crops_output_dir=crops_output_dir,
                    analyser=analyser,
                    crop_expansion=crop_expansion,
                    counters=counters,
                )
            except Exception:
                logger.exception("Worker failed")
                raise
            finally:
                frame_tmpdir.cleanup()

    _print_evaluation_summary(counters)
