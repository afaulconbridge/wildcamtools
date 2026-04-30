import logging
import socket
import threading
from pathlib import Path
from typing import Self, override

import av
import ffmpeg
import ffmpeg.codecs.encoders

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
    def __init__(self) -> None:
        super().__init__(
            [
                "./tests/bin/mediamtx",
                "./tests/bin/mediamtx.yml",
            ],
            ready_check=lambda: socket_check(port=8554),
            ready_timeout=10.0,
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
            # codec="libx264",
            # see https://trac.ffmpeg.org/wiki/Encode/H.264
            encoder_options=ffmpeg.codecs.encoders.libx264(
                tune="fastdecode",
                crf=23.0,
                preset="ultrafast",
            ),
        )
        logger.debug(ffmpeg_cmd.compile_line())
        self.process = ffmpeg_cmd.run_async()


class RTSPBroadcaster:
    def __init__(
        self,
        source: str | Path,
        rtsp_url: str,
        loop: bool = True,
    ):
        self.source = Path(source).resolve()
        self.rtsp_url = rtsp_url
        self.loop = loop
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._start_exception: Exception | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._started_event.clear()
        self._start_exception = None
        self._thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        self._thread.start()
        self._started_event.wait(timeout=30.0)
        if self._start_exception is not None:
            raise self._start_exception
        if not self._started_event.is_set():
            msg = "RTSP broadcaster failed to start within timeout"
            raise TimeoutError(msg)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _broadcast_loop(self) -> None:
        error_count = 0
        max_errors = 5
        base_backoff = 1.0
        max_backoff = 30.0
        first_attempt = True
        while not self._stop_event.is_set():
            try:
                self._broadcast_once()
                if first_attempt:
                    self._started_event.set()
                    first_attempt = False
                error_count = 0
                if not self.loop:
                    break
            except Exception as e:
                if first_attempt:
                    self._start_exception = e
                    self._started_event.set()
                if self._stop_event.is_set():
                    break
                error_count += 1
                logger.exception("Broadcast error (attempt %d/%d)", error_count, max_errors)
                if error_count >= max_errors:
                    logger.exception("Max broadcast errors reached, stopping")
                    break
                if not self.loop:
                    raise
                backoff = min(base_backoff * (2 ** (error_count - 1)), max_backoff)
                self._stop_event.wait(backoff)

    def _broadcast_once(self) -> None:
        with av.open(str(self.source), mode="r") as input_container:
            input_stream = input_container.streams.video[0]

            with av.open(
                self.rtsp_url,
                mode="w",
                format="rtsp",
                options={"rtsp_transport": "tcp"},
            ) as output_container:
                output_stream = output_container.add_stream("libx264", rate=input_stream.average_rate)
                output_stream.width = input_stream.width
                output_stream.height = input_stream.height
                output_stream.pix_fmt = "yuv420p"

                for packet in input_container.demux(input_stream):
                    if self._stop_event.is_set():
                        break
                    for frame in packet.decode():
                        for encoded_packet in output_stream.encode(frame):
                            output_container.mux(encoded_packet)

                for encoded_packet in output_stream.encode(None):
                    output_container.mux(encoded_packet)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object | None) -> None:
        self.stop()

    # TODO: support frame iterator source in addition to file path
