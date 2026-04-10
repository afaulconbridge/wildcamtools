from pathlib import Path
from typing import Annotated

import typer

from wildcamtools.lib.errors import OutputNotDirectoryError
from wildcamtools.lib.segment import create_segment_process

app = typer.Typer()


@app.command()
def segment(
    input_: Annotated[str, typer.Argument(metavar="INPUT")],
    output: Annotated[Path, typer.Argument(metavar="OUTPUT", help="directory to write into")],
    duration: Annotated[int, typer.Option(help="segment duration in seconds")] = 15,
) -> None:
    output = output.resolve()
    if output.exists() and not output.is_dir():
        raise OutputNotDirectoryError()
    output.mkdir(parents=True, exist_ok=True)

    p = create_segment_process(
        input_=input_,
        output=output,
        duration=duration,
    )

    p.wait()


if __name__ == "__main__":
    app()
