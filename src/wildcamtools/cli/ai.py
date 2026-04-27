import json
import logging
import multiprocessing
from multiprocessing.pool import AsyncResult
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

import typer

from wildcamtools.lib.ai import AbstractAnalyser, Backend
from wildcamtools.lib.ai.evaluate import create_frames, evaluate_frames
from wildcamtools.lib.ai.llamacpp import LlamaCppAnalyser
from wildcamtools.lib.ai.ollama import OllamaAnalyser
from wildcamtools.lib.errors.cli import InputNotDirectoryError
from wildcamtools.lib.label import load_labels
from wildcamtools.lib.timing import Timer

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def analyze(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    model: Annotated[str, typer.Option(help="Model name to use")],
    backend: Annotated[Backend, typer.Option(help="Backend to use")] = Backend.LLAMACPP,
    url: Annotated[str, typer.Option(help="Base URL for the backend")] = "http://localhost:8080/v1",
    api_key: Annotated[str | None, typer.Option(help="API key for ollama backend")] = None,
    output: Annotated[Path | None, typer.Option(metavar="OUTPUT")] = None,
) -> None:
    if not input_.exists():
        raise FileNotFoundError()
    if not input_.is_dir():
        raise InputNotDirectoryError()
    images = list(input_.iterdir())

    timer = Timer()
    analyser: AbstractAnalyser
    match backend:
        case Backend.LLAMACPP:
            analyser = LlamaCppAnalyser(model=model, base_url=url)
        case Backend.OLLAMA:
            analyser = OllamaAnalyser(model=model, host=url, api_key=api_key)
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
def evaluate(
    labels: Annotated[Path, typer.Argument(metavar="LABELS")],
    model: Annotated[str, typer.Option(help="Model name to use")],
    backend: Annotated[Backend, typer.Option(help="Backend to use")] = Backend.LLAMACPP,
    url: Annotated[str, typer.Option(help="Base URL for the backend")] = "http://localhost:8080/v1",
    api_key: Annotated[str | None, typer.Option(help="API key for ollama backend")] = None,
    history: int = 30,
    kernel_size: float = 0.02,
    x: int | None = None,
    y: int | None = None,
    fps: float | None = None,
    crop_expansion: float = 0.75,
    crop_inertia: float = 10.0,
    similarity_minimum: float | None = None,
) -> None:
    # TODO add option to read prompt from text file

    # read json lines
    labelled_data = load_labels(labels)
    video_directory = labels.resolve().parent

    # build a big ol' list of what we want to run
    parameter_dicts: list[dict[str, Any]] = []
    # TODO parameterize kernel_size as a reciprocal (e.g 1/100, 1/200, etc)
    for kernel_size in [0.01]:  # [0.02, 0.015, 0.01, 0.0075, 0.005]:
        for threshold in [16]:  # [8, 16, 24, 32, 48, 64]:
            parameter_dicts.append({
                "kernel_size": kernel_size,
                "threshold": threshold,
                "history": history,
                "crop_expansion": crop_expansion,
                "crop_inertia": crop_inertia,
                "x": x,
                "y": y,
                "fps": fps,
                "similarity_minimum": similarity_minimum,
                # "do_croppan": True,
            })

    # run image generation in parallel
    # use "spawn" rather than default "fork" for opencv compatibility
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool() as pool:
        futures: list[tuple[AsyncResult, str, TemporaryDirectory[str], dict[str, Any]]] = []
        for parameter_dict in parameter_dicts:
            for filename in labelled_data:
                frame_directory = TemporaryDirectory()
                future = pool.apply_async(
                    create_frames,
                    args=(video_directory / filename, video_directory, Path(frame_directory.name)),
                    kwds=parameter_dict,
                )
                futures.append((future, filename, frame_directory, parameter_dict))

        counters: dict[tuple[Any, ...], list[int]] = {}
        for future, filename, frame_directory, parameter_dict in futures:
            # as image generation finishes, detect and compare to label
            # use get() with a None return to ensure exceptions are raised
            future.get()

            label = labelled_data[filename]
            result = evaluate_frames(
                Path(frame_directory.name),
                label,
                backend,
                url,
                model,
                api_key=api_key if api_key else "API_KEY",
            )
            logger.info("File %s correct? %s", filename, result)

            counter_key = tuple(sorted(parameter_dict.items()))
            if counter_key not in counters:
                counters[counter_key] = [0, 0]
            counters[counter_key][0] += 1 if result else 0
            counters[counter_key][1] += 1

            # TODO parameterize filename
            with open("result.jsonl", "a") as result_file:
                out_dict = dict(parameter_dict)
                out_dict["filename"] = filename
                out_dict["model"] = model
                # this is a tuple of a dictionary of fixed variables and a dictionary of unknown result variables
                result_out = (
                    out_dict,
                    {
                        "result": result,
                        "label": label,
                        # TODO add more variables here e.g. frame count, motion proportion (min,q1,median,q2,max)
                    },
                )
                result_file.write(json.dumps(result_out))
                result_file.write("\n")

            # now cleanup the frames
            frame_directory.cleanup()

    for counter_key in counters:
        successes = counters[counter_key][0]
        attempts = counters[counter_key][1]
        ratio = successes / attempts
        typer.secho(f"{counter_key}: {successes} / {attempts} = {ratio:.4f}")
