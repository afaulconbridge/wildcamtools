from pathlib import Path
from typing import Annotated

import typer

from wildcamtools.lib.errors import OutputNotDirectoryError
from wildcamtools.lib.segment import VideoSegmenter

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

    with VideoSegmenter(input_=input_, segment_dir=output, segment_duration=float(duration)) as segmenter:
        for _ in segmenter:
            pass


if __name__ == "__main__":
    app()
