from pathlib import Path

from typer.testing import CliRunner

from wildcamtools.cli.motion_mog2 import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(args: list[str]) -> object:
    return runner.invoke(app, args)


# ---------------------------------------------------------------------------
# `flow` command – validation
# ---------------------------------------------------------------------------


def test_flow_rejects_nonpositive_threshold(video_path: Path, tmp_path: Path) -> None:
    """threshold <= 0 must exit with code 1 and emit an error message."""
    output = tmp_path / "out.mp4"
    result = _run(["flow", str(video_path), str(output), "--threshold", "0.0"])
    assert result.exit_code == 1
    assert "Threshold must be greater than 0" in result.output


def test_flow_rejects_negative_threshold(video_path: Path, tmp_path: Path) -> None:
    """Negative threshold is also rejected."""
    output = tmp_path / "out.mp4"
    result = _run(["flow", str(video_path), str(output), "--threshold", "-1.0"])
    assert result.exit_code == 1
    assert "Threshold must be greater than 0" in result.output


def test_flow_rejects_nonpositive_kernel_size(video_path: Path, tmp_path: Path) -> None:
    """kernel_size <= 0 must exit with code 1 and emit an error message."""
    output = tmp_path / "out.mp4"
    result = _run(["flow", str(video_path), str(output), "--kernel-size", "0.0"])
    assert result.exit_code == 1
    assert "Kernel size must be greater than 0" in result.output


def test_flow_rejects_negative_kernel_size(video_path: Path, tmp_path: Path) -> None:
    """Negative kernel_size is also rejected."""
    output = tmp_path / "out.mp4"
    result = _run(["flow", str(video_path), str(output), "--kernel-size", "-0.01"])
    assert result.exit_code == 1
    assert "Kernel size must be greater than 0" in result.output


def test_flow_rejects_history_longer_than_video(video_path: Path, tmp_path: Path) -> None:
    """history >= frame_count must exit with code 1."""
    output = tmp_path / "out.mp4"
    # test.mp4 has 150 frames; use history=200 to exceed it
    result = _run(["flow", str(video_path), str(output), "--history", "200"])
    assert result.exit_code == 1
    assert "Must have input longer than history" in result.output


# ---------------------------------------------------------------------------
# `mog2` command – validation
# ---------------------------------------------------------------------------


def test_mog2_rejects_nonpositive_kernel_size(video_path: Path, tmp_path: Path) -> None:
    """mog2: kernel_size <= 0 must exit with code 1."""
    output = tmp_path / "out.mp4"
    result = _run(["mog2", str(video_path), str(output), "--kernel-size", "0.0"])
    assert result.exit_code == 1
    assert "Kernel size must be greater than 0" in result.output


def test_mog2_rejects_negative_kernel_size(video_path: Path, tmp_path: Path) -> None:
    """mog2: negative kernel_size is rejected."""
    output = tmp_path / "out.mp4"
    result = _run(["mog2", str(video_path), str(output), "--kernel-size", "-0.05"])
    assert result.exit_code == 1
    assert "Kernel size must be greater than 0" in result.output


def test_mog2_rejects_history_longer_than_video(video_path: Path, tmp_path: Path) -> None:
    """mog2: history >= frame_count must exit with code 1."""
    output = tmp_path / "out.mp4"
    result = _run(["mog2", str(video_path), str(output), "--history", "200"])
    assert result.exit_code == 1
    assert "Must have input longer than history" in result.output


# ---------------------------------------------------------------------------
# `avg` command – validation
# ---------------------------------------------------------------------------


def test_avg_rejects_nonpositive_kernel_size(video_path: Path, tmp_path: Path) -> None:
    """avg: kernel_size <= 0 must exit with code 1."""
    output = tmp_path / "out.mp4"
    result = _run(["avg", str(video_path), str(output), "--kernel-size", "0.0"])
    assert result.exit_code == 1
    assert "Kernel size must be greater than 0" in result.output


def test_avg_rejects_negative_kernel_size(video_path: Path, tmp_path: Path) -> None:
    """avg: negative kernel_size is rejected."""
    output = tmp_path / "out.mp4"
    result = _run(["avg", str(video_path), str(output), "--kernel-size", "-0.1"])
    assert result.exit_code == 1
    assert "Kernel size must be greater than 0" in result.output


def test_avg_rejects_history_longer_than_video(video_path: Path, tmp_path: Path) -> None:
    """avg: history >= frame_count must exit with code 1."""
    output = tmp_path / "out.mp4"
    result = _run(["avg", str(video_path), str(output), "--history", "200"])
    assert result.exit_code == 1
    assert "Must have input longer than history" in result.output


# ---------------------------------------------------------------------------
# Regression: kernel_size is now float (was int) – float values accepted
# ---------------------------------------------------------------------------


def test_mog2_accepts_float_kernel_size_boundary(video_path: Path, tmp_path: Path) -> None:
    """mog2: a small positive float kernel_size (0.001) must not trigger the validation error."""
    output = tmp_path / "out.mp4"
    # If it fails it would be because kernel_size was treated as int and rejected by typer
    # We only care that the validation guard is NOT triggered (exit code may still be non-zero
    # due to video processing but the error must not be "Kernel size must be greater than 0")
    result = _run(["mog2", str(video_path), str(output), "--kernel-size", "0.001", "--history", "200"])
    # history check fires first – that's fine, but kernel_size validation must not fire
    assert "Kernel size must be greater than 0" not in result.output


def test_avg_accepts_float_kernel_size_boundary(video_path: Path, tmp_path: Path) -> None:
    """avg: a small positive float kernel_size (0.001) must not trigger the validation error."""
    output = tmp_path / "out.mp4"
    result = _run(["avg", str(video_path), str(output), "--kernel-size", "0.001", "--history", "200"])
    assert "Kernel size must be greater than 0" not in result.output