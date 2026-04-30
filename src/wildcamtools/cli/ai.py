import json
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from wildcamtools.lib.ai import AbstractAnalyser
from wildcamtools.lib.ai.llamacpp import LlamaCppAnalyser
from wildcamtools.lib.ai.ollama import OllamaAnalyser
from wildcamtools.lib.errors.cli import InputNotDirectoryError
from wildcamtools.lib.timing import Timer

app = typer.Typer()


class Backend(str, Enum):
    LLAMACPP = "llamacpp"
    OLLAMA = "ollama"


@app.command()
def analyze(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    model: Annotated[str, typer.Option(help="Model name to use")],
    backend: Annotated[Backend, typer.Option(help="Backend to use")] = Backend.LLAMACPP,
    url: Annotated[str, typer.Option(help="Base URL for the backend")] = "http://localhost:8080/v1",
    api_key: Annotated[str | None, typer.Option(help="API key for ollama backend")] = None,
    output: Annotated[Path | None, typer.Option(metavar="OUTPUT")] = None,
) -> None:
    if input_.exists() and not input_.is_dir():
        raise InputNotDirectoryError()
    images = list(input_.iterdir())

    timer = Timer()
    analyser: AbstractAnalyser
    match backend:
        case Backend.LLAMACPP:
            analyser = LlamaCppAnalyser(model=model, base_url=url)
        case Backend.OLLAMA:
            analyser = OllamaAnalyser(model=model, host=url, api_key=api_key)

    with timer:
        result = analyser.analyze_video(images)

    typer.secho(f"Processed {len(images):d} in {timer.elapsed:.2f} sec")
    typer.secho(f"Result: {result}")

    if output:
        with open(output, "w") as outfile:
            json.dump({"result": result}, outfile)
