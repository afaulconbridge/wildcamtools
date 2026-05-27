import logging
from pathlib import Path

from wildcamtools.lib.ai.crop import AICropFinder
from wildcamtools.lib.ai.llm.abstract import AbstractLlm
from wildcamtools.lib.frames import FrameImageRecreator, FrameImageWriter, Rescaler
from wildcamtools.lib.stats import get_video_stats
from wildcamtools.lib.vidio import VideoReader

logger = logging.getLogger(__name__)


def run_pipeline(
    video_path: Path,
    full_frames_dir: Path,
    low_res_frames_dir: Path,
    cropped_frames_dir: Path,
    fps: float,
    low_res_size: tuple[int, int],
    analyser: AbstractLlm,
    crop_expansion: float,
) -> None:
    """Run the complete pipeline on a single video.

    This is the main entry point for the pipeline business logic.
    """
    stats = get_video_stats(video_path)
    rescaler_fps = Rescaler(stats, fps=fps)
    rescaler_res = Rescaler(stats, x=low_res_size[0], y=low_res_size[1])
    writer_raw = FrameImageWriter(full_frames_dir)
    writer_low_res = FrameImageWriter(low_res_frames_dir)
    video_reader = VideoReader(video_path)
    with video_reader:
        for frame in video_reader:
            frame = rescaler_fps.handle(frame)
            if not frame.filter_keep:
                continue
            writer_raw.handle(frame)
            frame = rescaler_res.handle(frame)
            writer_low_res.handle(frame)

    cropper = AICropFinder(analyser=analyser, expansion=crop_expansion)
    cropper.run_detection(writer_low_res.outputs)

    writer_crops = FrameImageWriter(cropped_frames_dir)
    for frame in FrameImageRecreator(writer_raw.outputs, writer_low_res.outputs):
        frame = cropper.handle(frame)
        if frame.filter_keep:
            writer_crops.handle(frame)

    logger.info("Cropped frames saved to %s", cropped_frames_dir)
