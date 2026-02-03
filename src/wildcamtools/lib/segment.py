from pathlib import Path
from subprocess import Popen

import ffmpeg


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
            segment_atclocktime=1,
            reset_timestamps=1,
            strftime=1,
        ),
        filename=f"{output}/seg_%Y_%m_%d__%H_%M_%S.mp4",
    )
    return f.global_args(hide_banner=True, loglevel="error").overwrite_output().run_async()
