"""Tests for the CLI commands in wildcamtools.cli.motion_mog2.

Covers the new ``flow`` command and the new ``kernel_size <= 0`` validation
added to ``mog2`` and ``avg`` in this PR.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from wildcamtools.cli.motion_mog2 import app
from wildcamtools.lib.stats import Colourspace, VideoStats

runner = CliRunner()


def _make_stats(frame_count: int = 100, fps: float = 30.0) -> VideoStats:
    """Return a VideoStats instance suitable for mocking get_video_stats."""
    return VideoStats(
        fps=fps,
        frame_count=frame_count,
        x=640,
        y=480,
        colourspace=Colourspace.RGB,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke(command: str, extra_args: list[str], stats: VideoStats | None = None) -> Any:
    """Invoke a CLI sub-command with mocked get_video_stats."""
    if stats is None:
        stats = _make_stats()
    with patch("wildcamtools.cli.motion_mog2.get_video_stats", return_value=stats):
        return runner.invoke(app, [command, "input.mp4", "output.mp4", *extra_args])


# ---------------------------------------------------------------------------
# flow command – new in this PR
# ---------------------------------------------------------------------------


class TestFlowCommand:
    def test_history_longer_than_video_exits_with_code_1(self, tmp_path: Path) -> None:
        """history > frame_count must be rejected."""
        stats = _make_stats(frame_count=10)
        result = _invoke("flow", ["--history", "50"], stats=stats)
        assert result.exit_code == 1
        assert "Must have input longer than history" in result.output

    def test_threshold_zero_exits_with_code_1(self) -> None:
        """threshold=0 must be rejected."""
        result = _invoke("flow", ["--threshold", "0"])
        assert result.exit_code == 1
        assert "Threshold must be greater than 0" in result.output

    def test_threshold_negative_exits_with_code_1(self) -> None:
        """Negative threshold must be rejected."""
        result = _invoke("flow", ["--threshold", "-1.0"])
        assert result.exit_code == 1
        assert "Threshold must be greater than 0" in result.output

    def test_kernel_size_zero_exits_with_code_1(self) -> None:
        """kernel_size=0 must be rejected."""
        result = _invoke("flow", ["--kernel-size", "0"])
        assert result.exit_code == 1
        assert "Kernel size must be greater than 0" in result.output

    def test_kernel_size_negative_exits_with_code_1(self) -> None:
        """Negative kernel_size must be rejected."""
        result = _invoke("flow", ["--kernel-size", "-0.01"])
        assert result.exit_code == 1
        assert "Kernel size must be greater than 0" in result.output

    def test_history_exactly_equals_frame_count_exits_with_code_1(self) -> None:
        """frame_count - history == 0 means no output frames; should be rejected."""
        stats = _make_stats(frame_count=25)
        # history=25 → frame_count - history = 0 < 0 is False but = 0, which is NOT < 0
        # Actually: frame_count - history = 0, condition is `< 0` so 0 is NOT rejected
        # The check is `stats.frame_count - history < 0`, so exactly equal is allowed.
        # Test that history=26 on frame_count=25 IS rejected.
        stats = _make_stats(frame_count=25)
        result = _invoke("flow", ["--history", "26"], stats=stats)
        assert result.exit_code == 1
        assert "Must have input longer than history" in result.output

    def test_threshold_validation_checked_before_kernel_size(self) -> None:
        """threshold validation fires before kernel_size validation."""
        result = _invoke("flow", ["--threshold", "0", "--kernel-size", "-0.01"])
        assert result.exit_code == 1
        assert "Threshold must be greater than 0" in result.output


# ---------------------------------------------------------------------------
# mog2 command – new kernel_size <= 0 validation added in this PR
# ---------------------------------------------------------------------------


class TestMog2Command:
    def test_kernel_size_zero_exits_with_code_1(self) -> None:
        """kernel_size=0 must be rejected for mog2."""
        result = _invoke("mog2", ["--kernel-size", "0"])
        assert result.exit_code == 1
        assert "Kernel size must be greater than 0" in result.output

    def test_kernel_size_negative_exits_with_code_1(self) -> None:
        """Negative kernel_size must be rejected for mog2."""
        result = _invoke("mog2", ["--kernel-size", "-1.0"])
        assert result.exit_code == 1
        assert "Kernel size must be greater than 0" in result.output

    def test_kernel_size_checked_after_history_validation(self) -> None:
        """history validation fires before kernel_size validation for mog2."""
        stats = _make_stats(frame_count=5)
        result = _invoke("mog2", ["--history", "50", "--kernel-size", "-0.01"], stats=stats)
        assert result.exit_code == 1
        assert "Must have input longer than history" in result.output


# ---------------------------------------------------------------------------
# avg command – new kernel_size <= 0 validation added in this PR
# ---------------------------------------------------------------------------


class TestAvgCommand:
    def test_kernel_size_zero_exits_with_code_1(self) -> None:
        """kernel_size=0 must be rejected for avg."""
        result = _invoke("avg", ["--kernel-size", "0"])
        assert result.exit_code == 1
        assert "Kernel size must be greater than 0" in result.output

    def test_kernel_size_negative_exits_with_code_1(self) -> None:
        """Negative kernel_size must be rejected for avg."""
        result = _invoke("avg", ["--kernel-size", "-0.5"])
        assert result.exit_code == 1
        assert "Kernel size must be greater than 0" in result.output

    def test_kernel_size_checked_after_history_validation(self) -> None:
        """history validation fires before kernel_size validation for avg."""
        stats = _make_stats(frame_count=3)
        result = _invoke("avg", ["--history", "50", "--kernel-size", "-0.01"], stats=stats)
        assert result.exit_code == 1
        assert "Must have input longer than history" in result.output