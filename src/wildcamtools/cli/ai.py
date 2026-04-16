import json
from pathlib import Path
from typing import Annotated

import typer

from wildcamtools.lib.ai.llamacpp import LlamaCppAnalyser
from wildcamtools.lib.errors.cli import InputNotDirectoryError
from wildcamtools.lib.timing import Timer

app = typer.Typer()


@app.command()
def llamacpp(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    url: str,
    model: str,
    output: Annotated[Path | None, typer.Option(metavar="OUTPUT")] = None,
) -> None:
    if input_.exists() and not input_.is_dir():
        raise InputNotDirectoryError()
    # TODO filter on files, image endings
    images = list(input_.iterdir())
    # TODO error if no image found

    timer = Timer()
    analyser = LlamaCppAnalyser(model=model, base_url=url)

    with timer:
        result = analyser.analyze_video(images)

    typer.secho(f"Processed {len(images):d} in {timer.elapsed:.2f} sec")
    typer.secho(f"Result: {result}")

    if output:
        with open(output, "w") as outfile:
            json.dump({"result": result}, outfile)
