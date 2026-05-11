from pathlib import Path
from typing import Annotated

import typer

from wildcamtools.lib.errors.cli import OutputNotDirectoryError
from wildcamtools.lib.frames import AIFrameCreation, create_frames
from wildcamtools.lib.timing import Timer

app = typer.Typer()


@app.command()
def ssim(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    output: Annotated[Path | None, typer.Argument(metavar="OUTPUT")] = None,
    threshold: float | None = None,
    x: int | None = None,
    y: int | None = None,
    fps: float | None = None,
) -> None:
    if output is None:
        output = input_.with_suffix("")

    if output.exists() and not output.is_dir():
        raise OutputNotDirectoryError()

    frame_creation = AIFrameCreation(
        filename=Path(input_.name),
        video_directory=input_.parent,
        tmpdir=output,
        x=x,
        y=y,
        fps=fps,
        similarity_minimum=threshold,
    )

    timer = Timer()
    with timer:
        result = create_frames(frame_creation)

    fps_actual = result.frame_count / timer.elapsed if timer.elapsed > 0 else 0.0
    typer.secho(f"Processed {result.frame_count:d} frames in {timer.elapsed:.2f} sec; {fps_actual:.2f}FPS")


@app.command()
def flow(
    input_: Annotated[Path, typer.Argument(metavar="INPUT")],
    output: Annotated[Path | None, typer.Argument(metavar="OUTPUT")] = None,
    history: int = 30,
    threshold: float = 0.1,
    kernel_size: float = 0.02,
    x: int | None = None,
    y: int | None = None,
    fps: float | None = None,
    crop_expansion: float = 0.75,
    crop_inertia: float = 10.0,
    similarity_minimum: float = 0.5,
) -> None:
    if output is None:
        output = input_.with_suffix("")

    if output.exists() and not output.is_dir():
        raise OutputNotDirectoryError()

    frame_creation = AIFrameCreation(
        filename=Path(input_.name),
        video_directory=input_.parent,
        tmpdir=output,
        history=history,
        threshold=threshold,
        kernel_size=kernel_size,
        x=x,
        y=y,
        fps=fps,
        crop_expansion=crop_expansion,
        crop_inertia=crop_inertia,
        similarity_minimum=similarity_minimum,
        do_croppan=True,
        motion_type="flow",
    )
    timer = Timer()
    with timer:
        result = create_frames(frame_creation)

    fps_actual = result.frame_count / timer.elapsed if timer.elapsed > 0 else 0.0
    typer.secho(f"Processed {result.frame_count:d} frames in {timer.elapsed:.2f} sec; {fps_actual:.2f}FPS")


if __name__ == "__main__":
    app()
