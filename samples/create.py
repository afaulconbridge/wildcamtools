import multiprocessing
import shutil
import tempfile
from math import pi, sqrt
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel


class VideoJobArgs(BaseModel):
    tmpdir: Path
    frame_count: int
    width: int
    height: int
    area_proportion: float
    grey: float
    padding_count: int
    output_path: Path
    fps: int
    repeats: int


def create_circle_frames(
    tmpdir: Path,
    frame_count: int,
    width: int,
    height: int,
    area_proportion: float,
    grey: float,
    padding_count: int = 0,
) -> None:
    circle_area = width * height * area_proportion
    radius = sqrt(circle_area / (2.0 * pi))

    colour = (int(255 * grey), int(255 * grey), int(255 * grey))

    height_half = height / 2
    move_distance = (2 * radius) + width + (2 * radius)
    move_per_frame = move_distance / frame_count

    for frame_no in range(padding_count):
        frame = Image.new(mode="RGB", size=(width, height))
        frame_out = tmpdir / f"frame_{frame_no:06d}.png"
        frame.save(frame_out)

    for frame_no in range(frame_count):
        frame = Image.new(mode="RGB", size=(width, height))
        draw = ImageDraw.Draw(frame)
        move_offset = move_per_frame * frame_no
        draw.ellipse(
            [
                [int((-2 * radius) + move_offset), int(height_half - radius)],
                [int(move_offset), int(height_half + radius)],
            ],
            fill=colour,
            width=0,
        )

        frame_out = tmpdir / f"frame_{padding_count + frame_no:06d}.png"
        frame.save(frame_out)


def create_video_from_frames(path_wildcard: Path, output: Path | str, fps: int = 30, repeats: int = 1) -> None:
    output_path = Path(output)
    with av.open(str(output_path), mode="w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "medium"}

        frame_files = sorted(path_wildcard.parent.glob(path_wildcard.name))
        expected_size: tuple[int, int] | None = None
        for _ in range(repeats):
            for frame_file in frame_files:
                with Image.open(frame_file) as img:
                    img_rgb = img.convert("RGB")
                    frame_array = np.array(img_rgb)
                    current_size = (img_rgb.width, img_rgb.height)
                    if expected_size is None:
                        expected_size = current_size
                        stream.width = img.width
                        stream.height = img.height
                    elif current_size != expected_size:
                        msg = f"Frame size mismatch: expected {expected_size}, got {current_size} in {frame_file}"
                        raise ValueError(msg)
                av_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                av_frame.pts = stream.frames
                for packet in stream.encode(av_frame):
                    container.mux(packet)

        if expected_size is None:
            msg = f"No frames found matching pattern {path_wildcard}"
            raise ValueError(msg)

        for packet in stream.encode(None):
            container.mux(packet)


def _create_video_worker(
    tmpdir: Path,
    frame_count: int,
    width: int,
    height: int,
    area_proportion: float,
    grey: float,
    padding_count: int,
    output_path: Path,
    fps: int,
    repeats: int,
) -> Path:
    try:
        create_circle_frames(
            tmpdir=tmpdir,
            frame_count=frame_count,
            width=width,
            height=height,
            area_proportion=area_proportion,
            grey=grey,
            padding_count=padding_count,
        )

        create_video_from_frames(
            path_wildcard=tmpdir / "frame_*.png",
            output=output_path,
            fps=fps,
            repeats=repeats,
        )

        return output_path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    fps = 30
    duration_still = 5.0
    duration_movement = 10.0

    jobs: list[dict[str, Any]] = []
    for area_proportion in (0.1, 0.05, 0.01, 0.005, 0.001, 0.0005):
        for grey in (0.1, 0.25, 0.5, 1.0):
            jobs.append({
                "area_proportion": area_proportion,
                "grey": grey,
            })

    futures: list[AsyncResult] = []

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool() as pool:
        for job in jobs:
            output_path = (
                Path("samples/synth").resolve() / f"synth_{job['area_proportion']:0.4f}_{job['grey']:0.3f}.mp4"
            )

            job_args = VideoJobArgs(
                tmpdir=Path(tempfile.mkdtemp()),
                frame_count=int(fps * duration_movement),
                width=1920,
                height=1080,
                area_proportion=job["area_proportion"],
                grey=job["grey"],
                padding_count=int(fps * duration_still),
                output_path=output_path,
                fps=fps,
                repeats=4,
            )

            future = pool.apply_async(_create_video_worker, kwds=job_args.model_dump())
            futures.append(future)

        for future in futures:
            future.get()
