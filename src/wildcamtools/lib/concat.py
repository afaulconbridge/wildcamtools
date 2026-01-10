import logging
import tempfile
from collections.abc import Iterable
from pathlib import Path

import ffmpeg

logger = logging.getLogger(__name__)


def concat_ffmpeg(inputs: Iterable[Path], output: Path) -> None:

    # https://stackoverflow.com/questions/7333232/how-to-concatenate-two-mp4-files-using-ffmpeg
    # see https://ffmpeg.org/ffmpeg-formats.html#concat-1

    with tempfile.NamedTemporaryFile("w+t", suffix=".txt") as list_file:
        list_file.write("ffconcat version 1.0\n")
        for filename in inputs:
            # TODO escape special characters, but WTF are you doing with special character in filenames?!?
            list_file.write(f"file '{filename.resolve()}'\n")
        list_file.flush()
        ff = (
            ffmpeg.input(
                list_file.name,
                f="concat",
                extra_options={"safe": "0"},  # allow absolute paths
            )
            .output(
                filename=output.resolve(),
                c="copy",
            )
            .global_args(hide_banner=True, loglevel="error")
            .overwrite_output()
        )
        logger.debug(ff.compile_line())
        ff.run()
