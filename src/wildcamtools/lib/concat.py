import logging
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path

import av

from wildcamtools.lib.errors.core import translate_av_error

logger = logging.getLogger(__name__)


def concat_videos(inputs: Iterable[Path], output: Path) -> None:
    """Concatenate video files using PyAV concat demuxer.

    Uses FFmpeg concat demuxer via PyAV with stream copy (no re-encoding).

    Args:
        inputs: Iterable of input video file paths
        output: Output video file path

    Raises:
        ContainerError: If concat operation fails
    """
    inputs_list = list(inputs)
    logger.debug("Concatenating %d videos to %s", len(inputs_list), output)

    list_file_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w+t", suffix=".txt", delete=False) as list_file:
            list_file.write("ffconcat version 1.0\n")
            for filename in inputs_list:
                # TODO escape special characters in filenames
                list_file.write(f"file '{filename.resolve()}'\n")
            list_file_path = list_file.name

        with (
            av.open(
                list_file_path,
                format="concat",
                options={"safe": "0"},
            ) as container,
            av.open(output, "w") as output_container,
        ):
            for packet in container.demux():
                output_container.mux(packet)

        logger.debug("Successfully concatenated %d videos", len(inputs_list))

    except Exception as e:
        raise translate_av_error(e, str(output), "concat") from e
    finally:
        if list_file_path is not None:
            with suppress(Exception):
                Path(list_file_path).unlink()
