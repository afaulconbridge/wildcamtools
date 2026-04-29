import tempfile
from math import pi, sqrt
from pathlib import Path

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
    import av
    import numpy as np

    output_path = Path(output)
    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "medium"}

        frame_files = sorted(path_wildcard.parent.glob(path_wildcard.name.replace("*", "*")))
        expected_size: tuple[int, int] | None = None
        for frame_file in frame_files:
            with Image.open(frame_file) as img:
                img = img.convert("RGB")
                frame_array = np.array(img)
                current_size = (img.width, img.height)
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

        for packet in stream.encode(None):
            container.mux(packet)


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
