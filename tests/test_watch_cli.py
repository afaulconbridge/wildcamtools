from datetime import datetime
from unittest.mock import patch

from wildcamtools.cli.watch import WatcherManager, find_segments_for_timespan
from wildcamtools.cli.watch import app as watch_app
from wildcamtools.lib.states import WatcherTransitionMetrics


def test_watch_invalid_paths(runner, tmp_path):
    segments_dir = tmp_path / "segments"
    output_dir = tmp_path / "output"
    segments_dir.mkdir()
    output_dir.mkdir()
    missing = tmp_path / "nonexistent"

    # Invalid segments path
    result = runner.invoke(watch_app, ["rtsp://localhost", str(missing), str(output_dir)])
    assert result.exit_code != 0
    assert "segments" in result.stdout or "segments" in result.stderr

    # Invalid output path
    result = runner.invoke(watch_app, ["rtsp://localhost", str(segments_dir), str(missing)])
    assert result.exit_code != 0
    assert "output" in result.stdout or "output" in result.stderr


def test_watch_motion_mask_not_exists(runner, temp_dirs):
    segments_dir, output_dir = temp_dirs
    mask = temp_dirs[0].parent / "nonexistent_mask.png"

    result = runner.invoke(
        watch_app, ["rtsp://localhost", str(segments_dir), str(output_dir), "--motion-mask", str(mask)]
    )
    assert result.exit_code != 0


@patch("wildcamtools.cli.watch.WatcherManager.run")
def test_watch_initialization(mock_run, runner, temp_dirs):
    segments_dir, output_dir = temp_dirs

    result = runner.invoke(
        watch_app, ["rtsp://test-stream", str(segments_dir), str(output_dir), "--keep-count", "10", "--threshold", "20"]
    )

    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_find_segments_for_timespan(dummy_segments):
    # Timespan 10:00:05 -> 10:00:35 should overlap at least one 15s segment
    start = datetime(2026, 4, 21, 10, 0, 5)
    end = datetime(2026, 4, 21, 10, 0, 35)

    res = find_segments_for_timespan(start, end, dummy_segments)
    expected = [
        "seg_2026_04_21__10_00_00.mp4",
        "seg_2026_04_21__10_00_15.mp4",
        "seg_2026_04_21__10_00_30.mp4",
    ]
    assert res is not None
    assert {res.name for res in res} == set(expected)


def test_find_segments_incomplete_file(dummy_segments):
    # Test the "incomplete file" edge case where end_position == len(segments_files)
    start = datetime(2026, 4, 21, 10, 0, 0)
    end = datetime(2026, 4, 21, 10, 1, 1)  # Beyond the last file

    res = find_segments_for_timespan(start, end, dummy_segments)
    assert res is None


def test_cleanup_old_segments(temp_dirs):
    segments_dir, _ = temp_dirs
    # Create 10 dummy files
    for i in range(10):
        (segments_dir / f"seg_{i}.mp4").touch()

    # We need to call the actual method, but manager is a mock.
    # Let's just instantiate a real WatcherManager with minimum args.

    wm = WatcherManager(
        rtsp_stream="test",
        segments_dir=segments_dir,
        output_dir=temp_dirs[1],
        keep_count=5,
        offset_start=0,
        offset_end=0,
        history=0,
        threshold=0,
        kernel_size=0,
        scale=0,
        fps=0,
        hwaccel="",
        segment_duration=0,
        transition_metrics=WatcherTransitionMetrics(
            preparing_duration=0,
            green_to_amber_motion_min=0,
            amber_to_green_proportion_max=0,
            amber_to_red_duration=0,
            red_to_red_amber_proportion_max=0,
            red_amber_to_red_proportion_min=0,
            red_amber_to_green_duration=0,
        ),
    )

    wm.cleanup_old_segments()

    remaining_files = list(segments_dir.iterdir())
    assert len(remaining_files) == 5
