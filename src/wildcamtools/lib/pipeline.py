import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from wildcamtools.lib.ai import AbstractAnalyser
from wildcamtools.lib.ai.crop import AICropFinder
from wildcamtools.lib.ai.evaluate import AIEvaluation, evaluate_frames
from wildcamtools.lib.frames import FrameImageRecreator, FrameImageWriter, Rescaler
from wildcamtools.lib.stats import get_video_stats
from wildcamtools.lib.vidio import VideoReader

logger = logging.getLogger(__name__)


def generate_frames_for_video(
    filename: str,
    video_directory: Path,
    fps: float | None,
    low_res_size: tuple[int, int],
    frame_tmpdir: Path,
) -> tuple[str, list[Path], list[Path]]:
    """Worker function that generates frames for a single video.

    Runs in worker process. Creates TemporaryDirectory with subdirs for full and low-res frames.
    Reads video, applies FPS rescaling, writes both resolutions.
    Returns tuple of (filename, full_frame_paths, low_res_paths).
    """
    full_frames_dir = frame_tmpdir / "full"
    low_res_frames_dir = frame_tmpdir / "low_res"
    full_frames_dir.mkdir(parents=True, exist_ok=True)
    low_res_frames_dir.mkdir(parents=True, exist_ok=True)

    stats = get_video_stats(video_directory / filename)
    rescaler_fps = Rescaler(stats, fps=fps)
    rescaler_res = Rescaler(stats, x=low_res_size[0], y=low_res_size[1])
    writer_raw = FrameImageWriter(full_frames_dir)
    writer_low_res = FrameImageWriter(low_res_frames_dir)
    video_reader = VideoReader(video_directory / filename)
    with video_reader:
        for frame in video_reader:
            frame = rescaler_fps.handle(frame)
            if not frame.filter_keep:
                continue
            writer_raw.handle(frame)
            frame = rescaler_res.handle(frame)
            writer_low_res.handle(frame)

    return (filename, writer_raw.outputs, writer_low_res.outputs)


def process_pipeline_result(
    frame_result: tuple[str, Sequence[Path], Sequence[Path]],
    labelled_data: dict[str, Any],
    crops_output_dir: Path,
    analyser: AbstractAnalyser,
    crop_expansion: float,
    filename: str,
) -> tuple[bool, str, int]:
    """Process pipeline result: AI crop detection, evaluation, and output.

    Runs in main process. Creates video subdirectory in crops_output_dir, runs AI detection,
    applies crops to full-res frames, evaluates result.
    Returns tuple of (correct, raw_result, crop_count).
    """
    _, full_frame_paths, low_res_paths = frame_result
    label = labelled_data[filename]

    video_crop_dir = crops_output_dir / filename
    if video_crop_dir.exists():
        shutil.rmtree(video_crop_dir)
    video_crop_dir.mkdir(parents=True, exist_ok=True)

    cropper = AICropFinder(analyser=analyser, expansion=crop_expansion)
    cropper.run_detection(low_res_paths)

    writer_crops = FrameImageWriter(video_crop_dir)
    for frame in FrameImageRecreator(full_frame_paths, low_res_paths):
        frame = cropper.handle(frame)
        if frame.filter_keep:
            writer_crops.handle(frame)

    cropped_frame_paths = list(video_crop_dir.iterdir())
    if not cropped_frame_paths:
        raw_result = "no"
        correct = label.lower() == raw_result.lower()
    else:
        evaluation = AIEvaluation(
            frame_directory=video_crop_dir,
            label=label,
            backend=analyser.backend,
            url=analyser.url,
            model=analyser.model,
            api_key=getattr(analyser, "api_key", "API_KEY"),
            prompt=analyser.message,
        )
        evaluated_result = evaluate_frames(evaluation)
        raw_result = evaluated_result.raw_result
        correct = evaluated_result.correct

    return (correct, raw_result, len(cropped_frame_paths))


def run_pipeline(
    video_path: Path,
    full_frames_dir: Path,
    low_res_frames_dir: Path,
    cropped_frames_dir: Path,
    fps: float,
    low_res_size: tuple[int, int],
    analyser: AbstractAnalyser,
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
