import socket
from typing import override

import ffmpeg
from wildcamtools.lib.background_process import BackgroundProcess


def socket_check(host: str = "127.0.0.1", port: int = 8554, timeout: float = 1.0) -> bool:
    """
    This is a lightweight readiness check for a server being up.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class BackgroundMediaMTX(BackgroundProcess):
    def __init__(self):
        super().__init__(
            [
                "/home/adam/vscode/wildcamtools/tests/bin/mediamtx",
                "/home/adam/vscode/wildcamtools/tests/bin/mediamtx.yml",
            ],
        )


class BackgroundFFMPEGBroadcast(BackgroundProcess):
    def __init__(self):
        super().__init__(
            [
                "ffmpeg",
                "-re",
                # loop the input into the output stream
                "-stream_loop",
                "-1",
                # recalculate timestamps
                "-fflags",
                "+genpts",
                # input
                "-i",
                "/home/adam/vscode/wildcamtools/tests/data/04-51-08.mp4",
                "-c",
                "copy",
                # output
                "-f",
                "rtsp",
                "rtsp://localhost:8554/stream",
                "-use_wallclock_as_timestamps",
                "1",
            ],
            ready_check=socket_check,
        )

    @override
    def _create_process(self) -> None:
        ffmpeg_cmd = ffmpeg.input(
            "/home/adam/vscode/wildcamtools/tests/data/04-51-08.mp4",
            stream_loop=-1,
            re=True,
        ).output(
            filename="rtsp://localhost:8554/stream",
            f="rtsp",
            codec="copy",
        )
        print(ffmpeg_cmd.compile_line())
        self.process = ffmpeg_cmd.run_async()
