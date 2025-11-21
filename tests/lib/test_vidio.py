from pathlib import Path

import numpy as np

from wildcamtools.lib.stats import (
    Colourspace,
    VideoFileStats,
    get_video_stats,
)
from wildcamtools.lib.vidio import (
    generate_frames_cv2,
    generate_frames_cv2_rtsp,
    generate_latest_frames_cv2_rtsp,
    save_video,
)


def test_generate_frames_cv2(video_path: Path) -> None:
    for frame_no, array in generate_frames_cv2(video_path):
        assert frame_no > 0
        assert frame_no < (5 + 1) * 30  # expect just over 5 frames at 30 fps
        assert isinstance(array, np.ndarray)
        assert array.ndim == 3
        assert array.shape == (2160, 3840, 3)  # 4k colour


def test_generate_frames_cv2_rtsp(rtsp_server: str) -> None:
    seen = 0
    for frame_no, array in enumerate(generate_frames_cv2_rtsp(rtsp_server)):
        seen += 1
        assert isinstance(array, np.ndarray)
        assert array.ndim == 3
        assert array.shape == (2160, 3840, 3)  # 4k colour

        if frame_no > 30:
            # stream will loop by itself indefinately
            break
    assert seen > 0


def test_generate_latest_frames_cv2_rtsp(rtsp_server: str) -> None:
    seen = 0
    for frame_no, array in enumerate(generate_latest_frames_cv2_rtsp(rtsp_server)):
        seen += 1
        assert isinstance(array, np.ndarray)
        assert array.ndim == 3
        assert array.shape == (2160, 3840, 3)  # 4k colour

        if frame_no > 30:
            # stream will loop by itself indefinately
            break
    assert seen > 0


def test_save_video(tmp_path: Path) -> None:
    # Create temporary output path
    output_path = tmp_path / "test_output.mp4"

    # Generate sample frames (3 frames of different colors)
    red_frame = np.zeros((256, 256, 3), dtype=np.uint8)
    red_frame[:, :, 2] = 255
    green_frame = np.zeros((256, 256, 3), dtype=np.uint8)
    green_frame[:, :, 1] = 255
    blue_frame = np.zeros((256, 256, 3), dtype=np.uint8)
    blue_frame[:, :, 0] = 255

    # Create video stats for our test frames
    test_video_stats = VideoFileStats(fps=30.0, frame_count=3, x=256, y=256, colourspace=Colourspace.RGB)

    # Use save_video to write the frames
    with save_video(output_path, test_video_stats) as writer:
        for _ in range(int(test_video_stats.fps)):
            writer.write(red_frame)
        for _ in range(int(test_video_stats.fps)):
            writer.write(green_frame)
        for _ in range(int(test_video_stats.fps)):
            writer.write(blue_frame)

    # Verify the output video file
    stats = get_video_stats(output_path)

    assert stats.frame_count == 89  # rounds down
    assert stats.colourspace == Colourspace.RGB
    assert stats.fps == 30.0

    # Clean up the test file
    output_path.unlink()


def test_save_video_greyscale(tmp_path: Path) -> None:
    # Create temporary output path
    output_path = tmp_path / "test_output.mp4"

    # Generate sample frames (3 frames of different colors)
    white_frame = np.zeros((256, 256), dtype=np.uint8)
    white_frame[:, :] = 255
    grey_frame = np.zeros((256, 256), dtype=np.uint8)
    grey_frame[:, :] = 128
    black_frame = np.zeros((256, 256), dtype=np.uint8)

    # Create video stats for our test frames
    test_video_stats = VideoFileStats(fps=30.0, frame_count=3, x=256, y=256, colourspace=Colourspace.greyscale)

    # Use save_video to write the frames
    with save_video(output_path, test_video_stats) as writer:
        for _ in range(int(test_video_stats.fps)):
            writer.write(white_frame)
        for _ in range(int(test_video_stats.fps)):
            writer.write(grey_frame)
        for _ in range(int(test_video_stats.fps)):
            writer.write(black_frame)

    # Verify the output video file
    stats = get_video_stats(output_path)

    assert stats.frame_count == 89  # rounds down
    assert stats.colourspace == Colourspace.RGB  # currently always reads as colour
    assert stats.fps == 30.0

    # Clean up the test file
    output_path.unlink()
