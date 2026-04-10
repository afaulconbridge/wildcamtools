from pathlib import Path
from subprocess import Popen

import ffmpeg
import ffmpeg.formats.muxers

from wildcamtools.lib.errors import ProcessTypeMismatchError


def create_segment_process(*, input_: str | Path, output: str | Path, duration: float) -> Popen:
    f = ffmpeg.input(
        input_,
        # demuxer_options=ffmpeg.formats.demuxers.rtsp(rtsp_transport="tcp"),
    ).output(
        codec="copy",
        f="segment",
        muxer_options=ffmpeg.formats.muxers.segment(
            segment_time=str(duration),  # seconds
            segment_format="mp4",
            segment_format_options="movflags=+faststart",
            segment_atclocktime=True,
            reset_timestamps=True,
            strftime=True,
        ),
        filename=f"{output}/seg_%Y_%m_%d__%H_%M_%S.mp4",
    )
    res = f.global_args(hide_banner=True, loglevel="error").overwrite_output().run_async()
    if not isinstance(res, Popen):
        raise ProcessTypeMismatchError()
    return res
