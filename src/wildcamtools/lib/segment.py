from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

import av
import av.container
import av.stream

from wildcamtools.lib import Frame
from wildcamtools.lib.errors.core import VideoNotInContextError
from wildcamtools.lib.segment_metadata import SegmentMetadata
from wildcamtools.lib.utils import is_stream_url

logger = logging.getLogger(__name__)


class VideoSegmenter:
    """
    FrameSource that segments video input while emitting frames.

    Uses two separate PyAV containers:
    - Container 1: Segment muxer that writes segment files to disk
    - Container 2: Frame decoder that emits Frame objects

    This dual-container architecture allows decoding once while muxing
    to both outputs simultaneously.

    Parameters:
        input_: Path to input video file or RTSP URL
        segment_dir: Directory to write segment files
        segment_duration: Duration of each segment in seconds
        format_options: Optional dict of format options for segment muxer

    TODO: Optimize to use single container with custom output callback
          to avoid maintaining two separate container instances.
    """

    input_: str | Path
    segment_dir: Path
    segment_duration: float
    format_options: dict[str, str] | None
    _is_stream: bool

    _input_container: av.container.InputContainer | None
    _segment_container: av.container.OutputContainer | None
    _video_stream: av.VideoStream | None
    _frame_no: int
    _segment_count: int
    _last_segment_time: float
    _segment_file: Path | None
    _decoded_frames: list[av.VideoFrame]
    _fps: float | None
    _segment_start_frame: int
    _segment_start_time: datetime | None

    def __init__(
        self,
        input_: str | Path,
        segment_dir: str | Path,
        segment_duration: float,
        format_options: dict[str, str] | None = None,
    ) -> None:
        self.input_ = input_
        self.segment_dir = Path(segment_dir)
        self.segment_duration = segment_duration
        self.format_options = format_options
        self._is_stream = is_stream_url(input_)

        self._input_container = None
        self._segment_container = None
        self._video_stream = None
        self._frame_no = 0
        self._segment_count = 0
        self._last_segment_time = 0.0
        self._segment_file = None
        self._decoded_frames: list[av.VideoFrame] = []
        self._fps = None
        self._segment_start_frame = 0
        self._segment_start_time = None

    def __iter__(self) -> Self:
        return self

    def __enter__(self) -> Self:
        self.segment_dir.mkdir(parents=True, exist_ok=True)
        self._input_container = av.open(str(self.input_), mode="r")
        if not self._input_container.streams.video:
            msg = f"No video streams found in {self.input_}"
            raise ValueError(msg)
        self._video_stream = self._input_container.streams.video[0]
        self._fps = float(self._video_stream.average_rate or self._video_stream.base_rate or 0.0)
        if self._fps <= 0:
            logger.warning("Could not detect FPS from video stream, defaulting to 30.0")
            self._fps = 30.0
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any | None) -> Literal[False]:
        self._close_segment_container()
        if self._input_container:
            try:
                self._input_container.close()
            except Exception:
                logger.exception("Error closing input container")
            self._input_container = None
        self._video_stream = None
        return False

    def _close_segment_container(self) -> None:
        if self._segment_container:
            stream = self._segment_container.streams.video[0]
            for packet in stream.encode(None):
                self._segment_container.mux(packet)
            segment_path = Path(self._segment_container.name)
            self._segment_container.close()
            self._segment_container = None

            # Write metadata sidecar file
            if self._fps is not None:
                metadata = SegmentMetadata(
                    start_frame=self._segment_start_frame,
                    end_frame=self._frame_no,
                    start_time=self._segment_start_time,
                    end_time=datetime.now(UTC),
                    fps=self._fps,
                )
                metadata_path = SegmentMetadata.get_metadata_path(segment_path)
                metadata.save(metadata_path)
                logger.debug("Wrote segment metadata to %s", metadata_path)

    def _create_segment_container(self) -> None:
        if not self._video_stream:
            raise VideoNotInContextError()

        if self._is_stream:
            segment_name = f"seg_{datetime.now(UTC):%Y_%m_%d__%H_%M_%S}_{self._segment_count:04d}.mp4"
        else:
            segment_name = f"seg_frame{self._segment_start_frame:06d}.mp4"

        segment_path = self.segment_dir / segment_name

        self._segment_container = av.open(str(segment_path), mode="w", options=self.format_options)
        output_stream = self._segment_container.add_stream(
            codec_name="libx264",
            rate=self._video_stream.average_rate or self._video_stream.base_rate,
        )
        output_stream.width = self._video_stream.width
        output_stream.height = self._video_stream.height
        output_stream.pix_fmt = "yuv420p"
        output_stream.time_base = self._video_stream.time_base
        self._segment_count += 1

    def _write_frame_to_segment(self, frame: av.VideoFrame) -> None:
        if not self._segment_container or not self._video_stream:
            return

        stream = self._segment_container.streams.video[0]
        for packet in stream.encode(frame):
            self._segment_container.mux(packet)

    def _maybe_rotate_segment(self, frame_time: float | None) -> None:
        if self._segment_container is None:
            self._create_segment_container()
            self._last_segment_time = frame_time if frame_time is not None else 0.0
            self._segment_start_frame = self._frame_no
            self._segment_start_time = datetime.now(UTC) if frame_time is not None else None
            return

        if frame_time is not None and frame_time - self._last_segment_time >= self.segment_duration:
            self._close_segment_container()
            self._create_segment_container()
            self._last_segment_time = frame_time
            self._segment_start_frame = self._frame_no
            self._segment_start_time = datetime.now(UTC)

    def __next__(self) -> Frame:
        if not self._input_container or not self._video_stream:
            raise VideoNotInContextError()

        while True:
            if self._decoded_frames:
                return self._process_next_buffered_frame()

            self._decode_next_packet()

            if not self._decoded_frames:
                raise StopIteration

    def _decode_next_packet(self) -> None:
        """Decode the next packet and buffer all video frames."""
        if not self._input_container or not self._video_stream:
            raise VideoNotInContextError()
        for packet in self._input_container.demux(self._video_stream):
            decoded = packet.decode()
            for frame in decoded:
                if isinstance(frame, av.VideoFrame):
                    self._decoded_frames.append(frame)
            if self._decoded_frames:
                break

    def _process_next_buffered_frame(self) -> Frame:
        """Process and return the next buffered frame."""
        if not self._video_stream:
            raise VideoNotInContextError()
        frame = self._decoded_frames.pop(0)
        frame_time = (
            frame.time
            if frame.time is not None
            else (frame.pts * self._video_stream.time_base if frame.pts is not None else None)
        )
        timestamp: float | None = float(frame_time) if frame_time is not None else None
        self._maybe_rotate_segment(frame_time)
        self._write_frame_to_segment(frame)

        rgb_frame = frame.to_rgb().to_ndarray()
        result = Frame(raw=rgb_frame, frame_no=self._frame_no, timestamp=timestamp)
        self._frame_no += 1
        return result

    @property
    def segment_count(self) -> int:
        return self._segment_count

    @property
    def frame_count(self) -> int:
        return self._frame_no


class _PyAVSegmentProcess:
    """
    Popen-like wrapper for PyAV-based stream segmenter.

    Runs segmentation in a background thread to mimic subprocess behavior.
    Uses stream copy (no re-encoding) for efficiency.

    Attributes:
        stdin: Always None (not used for segmenting)
        stdout: Always None (not used for segmenting)
        stderr: Always None (not used for segmenting)
        returncode: Exit code (0 for success, 1 for error, 2 for no segments, None for running)
        pid: Fake PID (thread identifier)
        restart_on_exit: Whether process should restart on successful completion
    """

    stdin = None
    stdout = None
    stderr = None
    returncode: int | None
    pid: int
    restart_on_exit: bool

    def __init__(
        self,
        input_: str | Path,
        output: str | Path,
        duration: float,
        restart_on_exit: bool | None = None,
    ) -> None:
        self._input = str(input_)
        self._output = Path(output)
        self._duration = duration
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._error: Exception | None = None
        self.returncode = None
        self.pid = id(self) & 0xFFFF
        self._fps: float | None = None
        self._global_frame_no: int = 0
        self._segment_start_frame: int = 0
        self._segment_start_time: float | None = None
        # Auto-detect restart behavior if not explicitly specified
        # Stream URLs (rtsp://, http://, etc.) default to restart_on_exit=True
        # File paths default to restart_on_exit=False
        self.restart_on_exit = restart_on_exit if restart_on_exit is not None else is_stream_url(input_)

    def _run(self) -> None:
        """Main segmentation loop running in background thread."""
        input_container: av.container.InputContainer | None = None

        try:
            self._output.mkdir(parents=True, exist_ok=True)

            input_container = av.open(self._input, mode="r")

            video_stream, audio_stream = self._get_streams(input_container)

            segment_count = self._process_packets(input_container, video_stream, audio_stream)

            if segment_count == 0:
                msg = "Segmentation completed without producing any segments"
                logger.error(msg)
                self.returncode = 2
            else:
                logger.info("Segmentation completed: %d segments created", segment_count)
                self.returncode = 0

        except Exception:
            logger.exception("Segmentation error")
            self.returncode = 1

        finally:
            if input_container:
                try:
                    input_container.close()
                except Exception:
                    logger.exception("Error closing input container")

    def _process_packets(  # noqa: C901
        self,
        input_container: av.container.InputContainer,
        video_stream: av.VideoStream,
        audio_stream: av.AudioStream | None,
    ) -> int:
        """Process packets and create segments. Returns segment count."""
        segment_start_time: float | None = None
        segment_index = 0
        segment_count = 0
        output_streams: dict[int, av.stream.Stream] = {}
        output_container: av.container.OutputContainer | None = None
        segment_start_frame: int = 0
        prev_packet_frame: int = -1

        for packet in input_container.demux():
            if self._stop_event.is_set():
                break

            if not self._should_process_packet(packet):
                continue

            packet_time = self._get_packet_time(packet)
            if packet_time is None:
                continue

            # Calculate frame number from packet time using rounded multiplication to avoid drift
            current_frame = round(packet_time * (self._fps or 30.0))
            if prev_packet_frame < 0:
                # First packet, initialize frame counter
                self._global_frame_no = current_frame
            elif current_frame >= prev_packet_frame:
                self._global_frame_no += current_frame - prev_packet_frame
            else:
                logger.warning("Out-of-order packet detected: %d < %d", current_frame, prev_packet_frame)
                self._global_frame_no += max(0, current_frame - prev_packet_frame)
            prev_packet_frame = current_frame

            if segment_start_time is None:
                segment_start_time = packet_time
                segment_start_frame = self._global_frame_no

            if self._should_rotate_segment(packet_time, segment_start_time):
                if output_container:
                    self._close_output_container(
                        output_container,
                        start_frame=segment_start_frame,
                        end_frame=self._global_frame_no,
                        start_time=segment_start_time,
                        end_time=packet_time,
                    )
                    segment_count += 1
                    output_container = None
                    output_streams = {}
                segment_start_time = packet_time
                segment_start_frame = self._global_frame_no
                segment_index += 1

            if output_container is None:
                output_container, output_streams = self._create_output_container(
                    video_stream, audio_stream, segment_index, segment_start_frame
                )

            self._mux_packet(output_container, packet, output_streams)

        if output_container:
            self._close_output_container(
                output_container,
                start_frame=segment_start_frame,
                end_frame=self._global_frame_no,
                start_time=segment_start_time,
                end_time=prev_packet_frame / (self._fps or 30.0),
            )
            segment_count += 1

        return segment_count

    def _should_process_packet(self, packet: av.Packet) -> bool:
        """Check if packet should be processed."""
        return packet.stream.type in ("video", "audio") and packet.dts is not None

    def _should_rotate_segment(self, packet_time: float, segment_start_time: float) -> bool:
        """Check if a new segment should be started."""
        return packet_time - segment_start_time >= self._duration

    def _get_streams(
        self, input_container: av.container.InputContainer
    ) -> tuple[av.VideoStream, av.AudioStream | None]:
        """Extract video and audio streams from input container."""
        if not input_container.streams.video:
            msg = f"No video streams found in {self._input}"
            raise ValueError(msg)

        video_stream = input_container.streams.video[0]
        audio_stream = input_container.streams.audio[0] if input_container.streams.audio else None

        # Detect FPS from video stream
        self._fps = float(video_stream.average_rate or video_stream.base_rate or 0.0)
        if self._fps <= 0:
            logger.warning("Could not detect FPS from video stream, defaulting to 30.0")
            self._fps = 30.0

        return video_stream, audio_stream

    def _get_packet_time(self, packet: av.Packet) -> float | None:
        """Get packet timestamp in seconds."""
        if packet.dts is None or packet.stream.time_base is None:
            return None
        return float(packet.dts * packet.stream.time_base)

    def _mux_packet(
        self,
        container: av.container.OutputContainer,
        packet: av.Packet,
        output_streams: dict[int, av.stream.Stream],
    ) -> None:
        """Assign packet to output stream and mux it."""
        if packet.stream.index not in output_streams:
            return
        packet.stream = output_streams[packet.stream.index]
        container.mux(packet)

    def _create_output_container(
        self,
        video_stream: av.VideoStream,
        audio_stream: av.AudioStream | None,
        segment_index: int,
        segment_start_frame: int = 0,
    ) -> tuple[av.container.OutputContainer, dict[int, av.stream.Stream]]:
        """Create a new output segment file and return stream mapping."""
        if is_stream_url(self._input):
            segment_name = f"seg_{datetime.now(UTC):%Y_%m_%d__%H_%M_%S}_{segment_index:04d}.mp4"
        else:
            segment_name = f"seg_frame{segment_start_frame:06d}.mp4"

        segment_path = self._output / segment_name

        output_container = av.open(
            str(segment_path),
            mode="w",
            options={"movflags": "+faststart"},
        )

        output_streams: dict[int, av.stream.Stream] = {}
        output_streams[video_stream.index] = output_container.add_stream_from_template(video_stream)
        if audio_stream:
            output_streams[audio_stream.index] = output_container.add_stream_from_template(audio_stream)

        logger.debug("Created segment %s", segment_path)
        return output_container, output_streams

    def _close_output_container(
        self,
        container: av.container.OutputContainer,
        start_frame: int = 0,
        end_frame: int = 0,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> None:
        """Flush and close output container and write metadata file."""
        try:
            container.close()
            logger.debug("Closed segment container")

            # Write metadata sidecar file
            segment_path = Path(container.name)
            metadata = SegmentMetadata(
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=datetime.fromtimestamp(start_time, tz=UTC) if start_time is not None else None,
                end_time=datetime.fromtimestamp(end_time, tz=UTC) if end_time is not None else None,
                fps=self._fps or 30.0,
            )
            metadata_path = SegmentMetadata.get_metadata_path(segment_path)
            metadata.save(metadata_path)
            logger.debug("Wrote segment metadata to %s", metadata_path)
        except Exception:
            logger.exception("Error closing output container or writing metadata")

    def start(self) -> None:
        """Start the background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def poll(self) -> int | None:
        """Check if process has terminated. Returns returncode if done, None if running."""
        if self._thread and not self._thread.is_alive():
            return self.returncode
        return None

    def wait(self, timeout: float | None = None) -> int:
        """Wait for process to terminate. Returns returncode."""
        if self._thread:
            self._thread.join(timeout=timeout)
        return self.returncode or 0

    def terminate(self) -> None:
        """Signal the thread to stop."""
        self._stop_event.set()

    def kill(self) -> None:
        """Alias for terminate."""
        self.terminate()


def create_segment_process(
    *,
    input_: str | Path,
    output: str | Path,
    duration: float,
    restart_on_exit: bool | None = None,
) -> _PyAVSegmentProcess:
    """
    Create a segmenter process using PyAV (no subprocess).

    Uses stream copy (no re-encoding) for efficiency.
    Runs in a background thread to mimic subprocess behavior.

    Parameters:
        input_: Path to input video file or RTSP URL
        output: Directory to write segment files
        duration: Duration of each segment in seconds
        restart_on_exit: If True, process should restart on successful completion.
                        If False, process terminates on completion.
                        If None (default), auto-detects from input type:
                        - Stream URLs (rtsp://, http://, etc.) → True
                        - File paths → False

    Returns:
        _PyAVSegmentProcess object with poll(), wait(), terminate() methods.
        The object has a `restart_on_exit` attribute indicating the configured behavior.
    """
    process = _PyAVSegmentProcess(
        input_=input_,
        output=output,
        duration=duration,
        restart_on_exit=restart_on_exit,
    )
    process.start()
    return process
