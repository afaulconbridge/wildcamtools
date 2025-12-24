import logging
import socket
from pathlib import Path
from typing import override

import ffmpeg
from wildcamtools.lib.background_process import BackgroundProcess

logger = logging.getLogger(__name__)


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
    path: Path

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        super().__init__([])

    @override
    def _create_process(self) -> None:
        ffmpeg_cmd = ffmpeg.input(
            self.path,
            stream_loop=-1,
            re=True,
        ).output(
            filename="rtsp://localhost:8554/stream",
            f="rtsp",
            # see https://trac.ffmpeg.org/wiki/Encode/H.264
            encoder_options=ffmpeg.codecs.encoders.libx264(tune="fastdecode"),
        )
        logger.debug(ffmpeg_cmd.compile_line())
        self.process = ffmpeg_cmd.run_async()
