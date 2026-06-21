import logging
from collections.abc import Iterable
from pathlib import Path

import av
from pydantic import BaseModel, Field, model_validator

from wildcamtools.lib.vidio import VideoWriter

logger = logging.getLogger(__name__)


class SegmentInfo(BaseModel):
    """Information about a segment for trim calculation.

    Attributes:
        path: Path to the segment file
        start_frame: First frame number in the segment
        end_frame: Last frame number in the segment
        fps: Frames per second
        duration: Actual segment duration in seconds (optional)
        actual_frames: Actual frame count (optional)

    """

    path: Path
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    fps: float = Field(gt=0)
    duration: float | None = Field(default=None, ge=0)
    actual_frames: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_frame_range(self) -> "SegmentInfo":
        if self.end_frame < self.start_frame:
            msg = f"end_frame ({self.end_frame}) must be >= start_frame ({self.start_frame})"
            raise ValueError(msg)
        return self


def _calculate_concat_frame_range(
    trim_start_frame: int | None,
    trim_end_frame: int | None,
    segment_metadata: list[SegmentInfo],
) -> tuple[int | None, int | None]:
    """Map source frame numbers to concatenated video frame positions."""
    concat_start_frame: int | None = None
    concat_end_frame: int | None = None
    cumulative_frames = 0

    for seg in segment_metadata:
        seg_start_frame_concat = cumulative_frames
        if seg.actual_frames is not None:
            seg_frame_count = seg.actual_frames
        elif seg.duration is not None and seg.duration > 0:
            seg_frame_count = int(seg.duration * seg.fps + 0.5)
        else:
            seg_frame_count = seg.end_frame - seg.start_frame + 1

        if (
            trim_start_frame is not None
            and concat_start_frame is None
            and seg.start_frame <= trim_start_frame <= seg.end_frame
        ):
            concat_start_frame = seg_start_frame_concat + (trim_start_frame - seg.start_frame)

        if (
            trim_end_frame is not None
            and concat_end_frame is None
            and seg.start_frame <= trim_end_frame <= seg.end_frame
        ):
            concat_end_frame = seg_start_frame_concat + (trim_end_frame - seg.start_frame)

        cumulative_frames += seg_frame_count

    return concat_start_frame, concat_end_frame


def _resolve_trim_frames(
    trim_start_frame: int | None,
    trim_end_frame: int | None,
    segment_metadata: list[SegmentInfo] | None,
) -> tuple[int | None, int | None]:
    """Resolve trim frame numbers to concatenated video frame positions."""
    if segment_metadata and (trim_start_frame is not None or trim_end_frame is not None):
        concat_start_frame, concat_end_frame = _calculate_concat_frame_range(
            trim_start_frame, trim_end_frame, segment_metadata
        )
        if trim_start_frame is not None and concat_start_frame is None:
            msg = f"Trim start frame {trim_start_frame} not found in any segment metadata"
            raise ValueError(msg)
        if trim_end_frame is not None and concat_end_frame is None:
            msg = f"Trim end frame {trim_end_frame} not found in any segment metadata"
            raise ValueError(msg)
        return concat_start_frame, concat_end_frame
    return trim_start_frame, trim_end_frame


def _inspect_first_segment(input_path: Path, source_fps: float | None) -> tuple[float, int, int, bool]:
    """Open the first segment and extract FPS, resolution, and audio presence."""
    with av.open(str(input_path)) as container:
        if not container.streams.video:
            msg = f"Input file has no video stream: {input_path}"
            raise ValueError(msg)
        video_stream = container.streams.video[0]
        rate = video_stream.average_rate or video_stream.base_rate
        if rate:
            fps = round(float(rate), 2)
        elif source_fps is not None:
            fps = source_fps
        else:
            logger.warning(
                "Could not detect FPS from segment %s and no source_fps provided, defaulting to 30.0", input_path
            )
            fps = 30.0
        ref_width = video_stream.width
        ref_height = video_stream.height
        has_audio = bool(container.streams.audio)
    return fps, ref_width, ref_height, has_audio


def _concat_reencode(  # noqa: C901
    inputs: list[Path],
    output_path: Path,
    trim_start_frame: int | None = None,
    trim_end_frame: int | None = None,
    source_fps: float | None = None,
    segment_metadata: list[SegmentInfo] | None = None,
) -> None:
    """Decode all segments frame-by-frame and re-encode using VideoWriter."""
    if not inputs:
        msg = "No input files provided"
        raise ValueError(msg)

    if trim_start_frame is not None and trim_end_frame is not None and trim_start_frame > trim_end_frame:
        msg = f"trim_start_frame ({trim_start_frame}) must be <= trim_end_frame ({trim_end_frame})"
        raise ValueError(msg)

    concat_start_frame, concat_end_frame = _resolve_trim_frames(trim_start_frame, trim_end_frame, segment_metadata)

    fps, ref_width, ref_height, has_audio = _inspect_first_segment(inputs[0], source_fps)
    if has_audio:
        logger.warning("Audio tracks in input segments will be dropped (not yet supported)")

    with VideoWriter(output_path, fps=fps) as writer:
        frame_no = 0
        written_frames = 0
        video_done = False
        for input_path in inputs:
            if video_done:
                break
            logger.debug("Processing segment %s", input_path)
            with av.open(str(input_path)) as input_container:
                if not input_container.streams.video:
                    msg = f"Segment has no video stream: {input_path}"
                    raise ValueError(msg)
                in_video = input_container.streams.video[0]
                if in_video.width != ref_width or in_video.height != ref_height:
                    msg = (
                        f"Segment {input_path} resolution {in_video.width}x{in_video.height} "
                        f"differs from first segment {ref_width}x{ref_height}"
                    )
                    raise ValueError(msg)
                if input_container.streams.audio and not has_audio:
                    logger.warning("Segment %s has audio but first segment does not; audio will be dropped", input_path)

                for packet in input_container.demux(input_container.streams.video[0]):
                    try:
                        decoded_frames = packet.decode()
                    except av.error.FFmpegError:
                        logger.warning("Failed to decode packet in segment %s, skipping", input_path)
                        continue
                    for frame in decoded_frames:
                        if isinstance(frame, av.VideoFrame):
                            if concat_start_frame is not None and frame_no < concat_start_frame:
                                frame_no += 1
                                continue
                            if concat_end_frame is not None and frame_no > concat_end_frame:
                                video_done = True
                                break

                            rgb = frame.to_ndarray(format="rgb24")
                            writer.write(rgb)
                            frame_no += 1
                            written_frames += 1

                    if video_done:
                        break

    logger.debug("Wrote %d frames to %s", written_frames, output_path)


def concat_videos(
    inputs: Iterable[Path],
    output: Path,
    trim_start_frame: int | None = None,
    trim_end_frame: int | None = None,
    source_fps: float | None = None,
    segment_metadata: list[SegmentInfo] | None = None,
) -> None:
    """Concatenate video files using PyAV with optional frame-accurate trimming.

    Decodes all input segments frame-by-frame to numpy arrays and re-encodes
    them into a single output file using VideoWriter. When trim_start_frame
    and/or trim_end_frame are specified, only frames within that range are
    included in the output.

    Args:
        inputs: Iterable of input video file paths
        output: Output video file path
        trim_start_frame: First frame to include (for trimming)
        trim_end_frame: Last frame to include (for trimming)
        source_fps: Fallback FPS used when the segment file has no detectable
            frame rate. Required when trim parameters are provided.
        segment_metadata: Optional list of segment metadata for accurate trim calculation

    Raises:
        ValueError: If trim parameters provided without source_fps, if segments
            have mismatched resolution, if an input has no video stream, or if
            trim frames are not found in segment metadata

    """
    inputs_list = list(inputs)
    logger.debug("Concatenating %d videos to %s", len(inputs_list), output)

    if not inputs_list:
        logger.warning("No input videos to concatenate")
        return

    if (trim_start_frame is not None or trim_end_frame is not None) and source_fps is None:
        msg = "source_fps is required when using trim_start_frame or trim_end_frame"
        raise ValueError(msg)

    _concat_reencode(
        inputs_list,
        output,
        trim_start_frame=trim_start_frame,
        trim_end_frame=trim_end_frame,
        source_fps=source_fps,
        segment_metadata=segment_metadata,
    )

    logger.debug("Successfully concatenated %d videos", len(inputs_list))
