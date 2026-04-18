"""Tests for the motion_mog2 CLI commands added/modified in this PR.

Covers:
- New `flow` command with threshold/kernel_size/history validation
- Updated `mog2` command with kernel_size <= 0 validation (new guard)
- Updated `avg` command with kernel_size <= 0 validation (new guard)
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from wildcamtools.cli.motion_mog2 import app
from wildcamtools.lib.stats import Colourspace, VideoStats

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_STATS_ENOUGH = VideoStats(frame_count=150, fps=30.0, x=640, y=480, colourspace=Colourspace.RGB)


def _fake_stats(frame_count: int = 150) -> VideoStats:
    return VideoStats(frame_count=frame_count, fps=30.0, x=640, y=480, colourspace=Colourspace.RGB)


# ---------------------------------------------------------------------------
# `flow` command — new command introduced in this PR
# ---------------------------------------------------------------------------


class TestFlowCommand:
    def test_threshold_zero_exits_with_code_1(self, tmp_path: Path):
        """threshold=0 must exit with error code 1 and print a message."""
        with patch("wildcamtools.cli.motion_mog2.get_video_stats", return_value=_FAKE_STATS_ENOUGH):
            result = runner.invoke(
                app,
                ["flow", "input.mp4", str(tmp_path / "out.mp4"), "--threshold", "0"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "Threshold must be greater than 0" in result.output

    def test_negative_threshold_exits_with_code_1(self, tmp_path: Path):
        """threshold=-1 must also exit with error code 1."""
        with patch("wildcamtools.cli.motion_mog2.get_video_stats", return_value=_FAKE_STATS_ENOUGH):
            result = runner.invoke(
                app,
                ["flow", "input.mp4", str(tmp_path / "out.mp4"), "--threshold", "-1"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "Threshold must be greater than 0" in result.output

    def test_kernel_size_zero_exits_with_code_1(self, tmp_path: Path):
        """kernel_size=0 must exit with error code 1."""
        with patch("wildcamtools.cli.motion_mog2.get_video_stats", return_value=_FAKE_STATS_ENOUGH):
            result = runner.invoke(
                app,
                ["flow", "input.mp4", str(tmp_path / "out.mp4"), "--kernel-size", "0"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "Kernel size must be greater than 0" in result.output

    def test_negative_kernel_size_exits_with_code_1(self, tmp_path: Path):
        """kernel_size=-0.5 must exit with error code 1."""
        with patch("wildcamtools.cli.motion_mog2.get_video_stats", return_value=_FAKE_STATS_ENOUGH):
            result = runner.invoke(
                app,
                ["flow", "input.mp4", str(tmp_path / "out.mp4"), "--kernel-size", "-0.5"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "Kernel size must be greater than 0" in result.output

    def test_history_longer_than_video_exits_with_code_1(self, tmp_path: Path):
        """history > frame_count must exit with error code 1."""
        with patch("wildcamtools.cli.motion_mog2.get_video_stats", return_value=_fake_stats(frame_count=5)):
            result = runner.invoke(
                app,
                ["flow", "input.mp4", str(tmp_path / "out.mp4"), "--history", "100"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "Must have input longer than history" in result.output

    def test_threshold_validation_happens_after_history_check(self, tmp_path: Path):
        """Threshold validation runs after the history guard — zero threshold still caught."""
        with patch("wildcamtools.cli.motion_mog2.get_video_stats", return_value=_FAKE_STATS_ENOUGH):
            result = runner.invoke(
                app,
                ["flow", "input.mp4", str(tmp_path / "out.mp4"), "--threshold", "0.0"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "Threshold must be greater than 0" in result.output

    def test_kernel_validation_happens_after_threshold_check(self, tmp_path: Path):
        """Kernel validation runs only after threshold passes — zero kernel still caught."""
        with patch("wildcamtools.cli.motion_mog2.get_video_stats", return_value=_FAKE_STATS_ENOUGH):
            result = runner.invoke(
                app,
                ["flow", "input.mp4", str(tmp_path / "out.mp4"), "--threshold", "5.0", "--kernel-size", "0.0"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "Kernel size must be greater than 0" in result.output


# ---------------------------------------------------------------------------
# `mog2` command — kernel_size <= 0 guard is new in this PR
# ---------------------------------------------------------------------------


class TestMog2Command:
    def test_kernel_size_zero_exits_with_code_1(self, tmp_path: Path):
        """kernel_size=0 must exit with error code 1 (newly added validation)."""
        with patch("wildcamtools.cli.motion_mog2.get_video_stats", return_value=_FAKE_STATS_ENOUGH):
            result = runner.invoke(
                app,
                ["mog2", "input.mp4", str(tmp_path / "out.mp4"), "--kernel-size", "0"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "Kernel size must be greater than 0" in result.output

    def test_negative_kernel_size_exits_with_code_1(self, tmp_path: Path):
        """kernel_size=-1 must exit with error code 1."""
        with patch("wildcamtools.cli.motion_mog2.get_video_stats", return_value=_FAKE_STATS_ENOUGH):
            result = runner.invoke(
                app,
                ["mog2", "input.mp4", str(tmp_path / "out.mp4"), "--kernel-size", "-1"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "Kernel size must be greater than 0" in result.output

    def test_history_longer_than_video_exits_with_code_1(self, tmp_path: Path):
        """history > frame_count must exit with error code 1."""
        with patch("wildcamtools.cli.motion_mog2.get_video_stats", return_value=_fake_stats(frame_count=5)):
            result = runner.invoke(
                app,
                ["mog2", "input.mp4", str(tmp_path / "out.mp4"), "--history", "100"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "Must have input longer than history" in result.output

    def test_kernel_size_accepts_float(self, tmp_path: Path):
        """kernel_size as a positive float must pass validation."""
        with (
            patch("wildcamtools.cli.motion_mog2.get_video_stats", return_value=_FAKE_STATS_ENOUGH),
            patch("wildcamtools.cli.motion_mog2._shared") as mock_shared,
        ):
            result = runner.invoke(
                app,
                ["mog2", "input.mp4", str(tmp_path / "out.mp4"), "--kernel-size", "0.05"],
                catch_exceptions=False,
            )
        # Should not exit with code 1 due to kernel validation
        assert "Kernel size must be greater than 0" not in result.output
        mock_shared.assert_called_once()
        _, _, _, _, motion_handler = mock_shared.call_args.args
        assert motion_handler.kernel_size == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# `avg` command — kernel_size <= 0 guard is new in this PR
# ---------------------------------------------------------------------------


class TestAvgCommand:
    def test_kernel_size_zero_exits_with_code_1(self, tmp_path: Path):
        """kernel_size=0 must exit with error code 1 (newly added validation)."""
        with patch("wildcamtools.cli.motion_mog2.get_video_stats", return_value=_FAKE_STATS_ENOUGH):
            result = runner.invoke(
                app,
                ["avg", "input.mp4", str(tmp_path / "out.mp4"), "--kernel-size", "0"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "Kernel size must be greater than 0" in result.output

    def test_negative_kernel_size_exits_with_code_1(self, tmp_path: Path):
        """kernel_size=-0.1 must exit with error code 1."""
        with patch("wildcamtools.cli.motion_mog2.get_video_stats", return_value=_FAKE_STATS_ENOUGH):
            result = runner.invoke(
                app,
                ["avg", "input.mp4", str(tmp_path / "out.mp4"), "--kernel-size", "-0.1"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "Kernel size must be greater than 0" in result.output

    def test_history_longer_than_video_exits_with_code_1(self, tmp_path: Path):
        """history > frame_count must exit with error code 1."""
        with patch("wildcamtools.cli.motion_mog2.get_video_stats", return_value=_fake_stats(frame_count=5)):
            result = runner.invoke(
                app,
                ["avg", "input.mp4", str(tmp_path / "out.mp4"), "--history", "100"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1
        assert "Must have input longer than history" in result.output

    def test_kernel_size_accepts_float(self, tmp_path: Path):
        """kernel_size as a positive float must pass validation."""
        with (
            patch("wildcamtools.cli.motion_mog2.get_video_stats", return_value=_FAKE_STATS_ENOUGH),
            patch("wildcamtools.cli.motion_mog2._shared") as mock_shared,
        ):
            result = runner.invoke(
                app,
                ["avg", "input.mp4", str(tmp_path / "out.mp4"), "--kernel-size", "0.02"],
                catch_exceptions=False,
            )
        assert "Kernel size must be greater than 0" not in result.output
        mock_shared.assert_called_once()
        _, _, _, _, motion_handler = mock_shared.call_args.args
        assert motion_handler.kernel_size == pytest.approx(0.02)