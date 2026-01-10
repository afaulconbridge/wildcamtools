from pathlib import Path
from typing import Annotated

import click
import typer

import ffmpeg

app = typer.Typer()


@app.command()
def segment(
    input_: Annotated[str, typer.Argument(metavar="INPUT")],
    output: Annotated[Path, typer.Argument(metavar="OUTPUT", help="directory to write into")],
    duration: Annotated[int, typer.Option(help="segment duration in seconds")] = 15,
) -> None:
    output = output.resolve()
    if output.exists() and not output.is_dir():
        raise click.BadArgumentUsage("Output must be a directory that can be created")

    output.mkdir(parents=True, exist_ok=True)
    f = ffmpeg.input(input_).output(
        codec="copy",
        f="segment",
        muxer_options=ffmpeg.formats.muxers.segment(
            segment_time=duration,  # every N seconds
            segment_format="mp4",
            segment_format_options="movflags=+faststart",
            reset_timestamps=1,
            strftime=1,
        ),
        filename=f"{output}/seg_%Y_%m_%d__%H_%M_%S.mp4",
    )
    f.global_args(hide_banner=True, loglevel="error").overwrite_output().run()


if __name__ == "__main__":
    app()
