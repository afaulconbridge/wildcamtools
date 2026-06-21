#!/usr/bin/env python3
"""Evaluate motion detection parameters against synthetic videos.

Runs motion detection on synthetic videos with known ground truth and compares
detected motion windows against expected motion periods.

Usage:
    uv run python samples/evaluate.py [OPTIONS]
"""

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()


class VideoMetadata(BaseModel):
    video_path: str
    area_proportion: float
    grey: float
    fps: int
    width: int
    height: int
    padding_frames: int
    motion_frames: int
    total_frames: int
    motion_start_frame: int
    motion_end_frame: int
    duration_still_seconds: float
    duration_motion_seconds: float
    repeats: int


class MotionWindowResult(BaseModel):
    start_frame: int
    end_frame: int
    start_time: str | None = None
    end_time: str | None = None


class EvaluationResult(BaseModel):
    video_name: str
    video_path: str
    threshold: int
    kernel_size: float
    scale: float
    fps: float
    history: int
    green_to_amber_motion_min: float
    detection_rate: float
    false_positive_rate: float
    motion_proportion_mean: float
    motion_proportion_std: float
    state_transition_latency_frames: int | None
    motion_windows_detected: int
    motion_windows: list[MotionWindowResult]
    output_json_files: list[str]
    evaluation_time: str
    processing_duration_seconds: float


@dataclass
class ParameterGrid:
    thresholds: list[int]
    kernel_sizes: list[float]
    scale: float = 0.25
    fps: float = 5.0
    history: int = 30
    green_to_amber_motion_min: float = 0.01


def load_metadata(metadata_path: Path) -> VideoMetadata | None:
    if not metadata_path.exists():
        logger.warning("Metadata file not found: %s", metadata_path)
        return None
    try:
        with open(metadata_path, encoding="utf-8") as f:
            data = json.load(f)
        return VideoMetadata(**data)
    except (json.JSONDecodeError, KeyError):
        logger.exception("Failed to parse metadata %s", metadata_path)
        return None


def run_motion_detection(
    video_path: Path,
    params: dict[str, Any],
    temp_dir: Path,
) -> list[Path]:
    segments_dir = temp_dir / "segments"
    output_dir = temp_dir / "output"
    segments_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "wildcamtools",
        "watch",
        str(video_path),
        str(segments_dir),
        str(output_dir),
        "--threshold",
        str(params["threshold"]),
        "--kernel-size",
        str(params["kernel_size"]),
        "--scale",
        str(params["scale"]),
        "--fps",
        str(params["fps"]),
        "--history",
        str(params["history"]),
        "--green-to-amber-motion-min",
        str(params["green_to_amber_motion_min"]),
        "--offset-start",
        "0",
        "--offset-end",
        "0",
        "--keep-count",
        "100",
        "--segment-duration",
        "5",
    ]

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # noqa: S603

    if result.returncode != 0:
        logger.error("Motion detection failed with exit code %d", result.returncode)
        logger.error("Command: %s", " ".join(cmd))
        logger.error("Stderr: %s", result.stderr)
        raise RuntimeError(f"Motion detection failed with exit code {result.returncode}: {result.stderr}")

    output_jsons = sorted(output_dir.glob("*.json"))
    return output_jsons


def parse_output_jsons(json_paths: list[Path]) -> tuple[list[MotionWindowResult], list[float]]:
    motion_windows: list[MotionWindowResult] = []
    motion_proportions: list[float] = []

    for json_path in json_paths:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            window = MotionWindowResult(
                start_frame=data.get("start_frame", 0),
                end_frame=data.get("end_frame", 0),
                start_time=data.get("start_time"),
                end_time=data.get("end_time"),
            )
            motion_windows.append(window)

            if "motion_window" in data:
                mw = data["motion_window"]
                if "transition_window_metrics" in mw:
                    for _state, metrics in mw["transition_window_metrics"].items():
                        if "mean" in metrics:
                            motion_proportions.append(metrics["mean"])
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse output JSON %s: %s", json_path, e)

    return motion_windows, motion_proportions


def evaluate_detection(
    metadata: VideoMetadata,
    motion_windows: list[MotionWindowResult],
    motion_proportions: list[float],
) -> dict[str, Any]:
    motion_start = metadata.motion_start_frame
    motion_end = metadata.motion_end_frame
    still_end = metadata.motion_start_frame - 1

    total_motion_frames = motion_end - motion_start + 1
    detected_motion_frames = 0
    false_positive_frames = 0

    for window in motion_windows:
        overlap_start = max(window.start_frame, motion_start)
        overlap_end = min(window.end_frame, motion_end)
        if overlap_end >= overlap_start:
            detected_motion_frames += overlap_end - overlap_start + 1

        fp_start = max(window.start_frame, 0)
        fp_end = min(window.end_frame, still_end)
        if fp_end >= fp_start:
            false_positive_frames += fp_end - fp_start + 1

    detection_rate = detected_motion_frames / total_motion_frames if total_motion_frames > 0 else 0.0
    total_still_frames = still_end + 1
    false_positive_rate = false_positive_frames / total_still_frames if total_still_frames > 0 else 0.0

    motion_proportion_mean = sum(motion_proportions) / len(motion_proportions) if motion_proportions else 0.0

    if len(motion_proportions) > 1:
        mean = motion_proportion_mean
        variance = sum((p - mean) ** 2 for p in motion_proportions) / len(motion_proportions)
        motion_proportion_std = variance**0.5
    else:
        motion_proportion_std = 0.0

    min_latency = None
    for window in motion_windows:
        if window.start_frame >= motion_start:
            latency = window.start_frame - motion_start
            if min_latency is None or latency < min_latency:
                min_latency = latency

    return {
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
        "motion_proportion_mean": motion_proportion_mean,
        "motion_proportion_std": motion_proportion_std,
        "state_transition_latency_frames": min_latency,
        "motion_windows_detected": len(motion_windows),
    }


def evaluate_video(
    video_path: Path,
    metadata: VideoMetadata,
    params: dict[str, Any],
) -> EvaluationResult | None:
    logger.info(
        "Evaluating %s with threshold=%d, kernel_size=%.4f",
        video_path.name,
        params["threshold"],
        params["kernel_size"],
    )

    start_time = datetime.now()

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        output_jsons = run_motion_detection(video_path, params, temp_path)

        if not output_jsons:
            logger.warning("No output generated for %s", video_path.name)
            return None

        motion_windows, motion_proportions = parse_output_jsons(output_jsons)

        metrics = evaluate_detection(metadata, motion_windows, motion_proportions)

    end_time = datetime.now()
    processing_duration = (end_time - start_time).total_seconds()

    result = EvaluationResult(
        video_name=video_path.name,
        video_path=str(video_path),
        threshold=params["threshold"],
        kernel_size=params["kernel_size"],
        scale=params["scale"],
        fps=params["fps"],
        history=params["history"],
        green_to_amber_motion_min=params["green_to_amber_motion_min"],
        detection_rate=metrics["detection_rate"],
        false_positive_rate=metrics["false_positive_rate"],
        motion_proportion_mean=metrics["motion_proportion_mean"],
        motion_proportion_std=metrics["motion_proportion_std"],
        state_transition_latency_frames=metrics["state_transition_latency_frames"],
        motion_windows_detected=metrics["motion_windows_detected"],
        motion_windows=motion_windows,
        output_json_files=[str(p) for p in output_jsons],
        evaluation_time=start_time.isoformat(),
        processing_duration_seconds=processing_duration,
    )

    return result


@app.command()
def main(
    videos_dir: Annotated[Path, typer.Argument(help="Directory containing synthetic videos")] = Path("samples/synth"),
    output_dir: Annotated[Path, typer.Argument(help="Output directory for evaluation results")] = Path(
        "samples/eval_results",
    ),
    thresholds: Annotated[
        list[int] | None,
        typer.Option("--threshold", "-t", help="Motion threshold values to test"),
    ] = None,
    kernel_sizes: Annotated[
        list[float] | None,
        typer.Option("--kernel-size", "-k", help="Kernel size values to test"),
    ] = None,
    scale: Annotated[float, typer.Option("--scale", "-s", help="Scale factor")] = 0.25,
    fps: Annotated[float, typer.Option("--fps", "-f", help="FPS")] = 5.0,
    history: Annotated[int, typer.Option("--history", "-h", help="History frames")] = 30,
    green_to_amber_motion_min: Annotated[
        float,
        typer.Option("--green-to-amber", "-g", help="Green to amber threshold"),
    ] = 0.01,
) -> None:
    """Evaluate motion detection parameters against synthetic videos."""
    if thresholds is None:
        thresholds = [8, 16, 32]
    if kernel_sizes is None:
        kernel_sizes = [0.001, 0.005, 0.01]

    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(videos_dir.glob("*.mp4"))
    if not video_paths:
        logger.error("No video files found in %s", videos_dir)
        raise typer.Exit(1)

    logger.info("Found %d videos to evaluate", len(video_paths))

    results: list[EvaluationResult] = []
    param_grid = ParameterGrid(
        thresholds=thresholds,
        kernel_sizes=kernel_sizes,
        scale=scale,
        fps=fps,
        history=history,
        green_to_amber_motion_min=green_to_amber_motion_min,
    )

    for video_path in video_paths:
        metadata_path = video_path.with_suffix(".json")
        metadata = load_metadata(metadata_path)

        if metadata is None:
            logger.warning("Skipping %s: no metadata found", video_path.name)
            continue

        for threshold in param_grid.thresholds:
            for kernel_size in param_grid.kernel_sizes:
                params = {
                    "threshold": threshold,
                    "kernel_size": kernel_size,
                    "scale": param_grid.scale,
                    "fps": param_grid.fps,
                    "history": param_grid.history,
                    "green_to_amber_motion_min": param_grid.green_to_amber_motion_min,
                }

                result = evaluate_video(video_path, metadata, params)
                if result:
                    results.append(result)
                    logger.info(
                        "Completed %s: %.2fs (detection=%.2f%%, fp=%.2f%%)",
                        video_path.name,
                        result.processing_duration_seconds,
                        result.detection_rate * 100,
                        result.false_positive_rate * 100,
                    )

    summary = {
        "evaluation_summary": {
            "total_videos": len(video_paths),
            "videos_with_metadata": len([v for v in video_paths if v.with_suffix(".json").exists()]),
            "parameter_combinations": len(param_grid.thresholds) * len(param_grid.kernel_sizes),
            "total_evaluations": len(results),
            "evaluation_time": datetime.now().isoformat(),
            "parameters": {
                "thresholds": param_grid.thresholds,
                "kernel_sizes": param_grid.kernel_sizes,
                "scale": param_grid.scale,
                "fps": param_grid.fps,
                "history": param_grid.history,
                "green_to_amber_motion_min": param_grid.green_to_amber_motion_min,
            },
        },
        "results": [r.model_dump() for r in results],
    }

    output_file = output_dir / "evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Evaluation complete. Results saved to %s", output_file)
    logger.info("Total evaluations: %d", len(results))

    if results:
        avg_detection = sum(r.detection_rate for r in results) / len(results)
        avg_fp = sum(r.false_positive_rate for r in results) / len(results)
        logger.info("Average detection rate: %.2f%%", avg_detection * 100)
        logger.info("Average false positive rate: %.2f%%", avg_fp * 100)


if __name__ == "__main__":
    app()
