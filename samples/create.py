import tempfile
from math import pi, sqrt
from pathlib import Path

import ffmpeg
from PIL import Image, ImageDraw


def create_circle_frames(
    tmpdir: Path,
    frame_count: int,
    width: int,
    height: int,
    area_proportion: float,
    grey: float,
    padding_count: int = 0,
) -> None:
    # have to convert circle_area_proportion -> radius
    circle_area = width * height * area_proportion
    # area = 2*pi*r*r
    # sqrt(area/(2*pi)) = r
    radius = sqrt(circle_area / (2.0 * pi))

    colour = (int(255 * grey), int(255 * grey), int(255 * grey))

    height_half = height / 2
    # in n frames move 2*radius+width+2*radius
    move_distance = (2 * radius) + width + (2 * radius)
    move_per_frame = move_distance / frame_count

    for frame_no in range(padding_count):
        frame = Image.new(mode="RGB", size=(width, height))
        # could use a t-string here, and for the wildcard input to ffmpeg!
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

        # could use a t-string here, and for the wildcard input to ffmpeg!
        frame_out = tmpdir / f"frame_{padding_count + frame_no:06d}.png"
        frame.save(frame_out)


def create_video_from_frames(path_wildcard: Path, output: Path | str, fps: int = 30) -> None:
    ffmpeg.input(
        path_wildcard,
        demuxer_options=ffmpeg.formats.demuxers.image2(
            pattern_type="glob",
            framerate=str(fps),
        ),
    ).output(
        codec="libx264",
        encoder_options=ffmpeg.codecs.encoders.libx264(),
        filename=output,
    ).global_args(
        hide_banner=True,
        loglevel="error",
    ).overwrite_output().run(
        quiet=True,
    )


if __name__ == "__main__":
    output = "samples/samples/synth.mp4"
    fps = 30
    duration_still = 5.0
    duration_movement = 10.0

    for area_proportion in (0.01, 0.005, 0.001, 0.0005):
        for grey in (0.1, 0.25, 0.5, 1.0):
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmpdir = Path(tmpdirname).resolve()
                # create circle moving from left to right
                create_circle_frames(
                    tmpdir=tmpdir,
                    frame_count=int(fps * duration_movement),
                    width=1920,
                    height=1080,
                    area_proportion=area_proportion,
                    grey=grey,
                    padding_count=int(fps * duration_still),
                )

                create_video_from_frames(
                    path_wildcard=tmpdir / "frame_*.png",
                    output=f"synth_{area_proportion:0.4f}_{grey:0.3f}.mp4",
                    fps=fps,
                )
