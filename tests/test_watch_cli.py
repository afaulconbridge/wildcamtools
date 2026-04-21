from datetime import datetime
from unittest.mock import MagicMock, patch

from wildcamtools.cli.watch import WatcherManager, find_segments_for_timespan
from wildcamtools.cli.watch import app as watch_app


def test_watch_invalid_paths(runner, temp_dirs):
    segments_dir, output_dir = temp_dirs

    # Invalid segments path
    result = runner.invoke(watch_app, ["rtsp://localhost", str(temp_dirs[0].parent / "nonexistent"), str(output_dir)])
    assert result.exit_code != 0
    assert (
        "segments must be an existing directory" in result.stdout
        or "segments must be an existing directory" in result.stderr
    )

    # Invalid output path
    result = runner.invoke(watch_app, ["rtsp://localhost", str(segments_dir), str(temp_dirs[0].parent / "nonexistent")])
    assert result.exit_code != 0
    assert (
        "output must be an existing directory" in result.stdout
        or "output must be an existing directory" in result.stderr
    )


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
    # Created 5 segments from 10:00:00 to 10:00:45 (15s intervals)
    # seg_2026_04_21__10_00_00.mp4
    # seg_2026_04_21__10_00_15.mp4
    # seg_2026_04_21__10_00_30.mp4
    # seg_2026_04_21__10_00_45.mp4
    # seg_2026_04_21__10_01_00.mp4 (actually index 4 is 10:01:00 if loop 5)

    # Wait, my fixture did:
    # i=0: 10:00:00
    # i=1: 10:00:15
    # i=2: 10:00:30
    # i=3: 10:00:45
    # i=4: 10:01:00 (since 0 + 4*15 = 60s)

    start = datetime(2026, 4, 21, 10, 0, 5)
    end = datetime(2026, 4, 21, 10, 0, 35)

    # Finding from 10:00:05 to 10:00:35
    # Should include segments that overlap this range.
    # The logic in find_segments_for_timespan uses bisect_left/right on filenames.
    # and return segments_files[max(start_position - 1, 0) : end_position]

    res = find_segments_for_timespan(start, end, dummy_segments)
    assert res is not None
    # Expecting segments that cover the period
    assert len(res) >= 1


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

    manager = MagicMock(spec=WatcherManager)
    manager.segments_dir = segments_dir
    manager.keep_count = 5

    # We need to call the actual method, but manager is a mock.
    # Let's just instantiate a real WatcherManager with minimum args.
    from wildcamtools.lib.states import WatcherTransitionMetrics

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
