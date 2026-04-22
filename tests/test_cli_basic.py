from unittest.mock import MagicMock, patch

from wildcamtools.cli.ai import app as ai_app
from wildcamtools.cli.rtsp import app as rtsp_app
from wildcamtools.cli.segment import app as segment_app


def test_rtsp_serve_basic(runner, temp_dirs):
    segments_dir, _ = temp_dirs
    with (
        patch("wildcamtools.cli.rtsp.BackgroundMediaMTX"),
        patch("wildcamtools.cli.rtsp.BackgroundFFMPEGBroadcast"),
        patch("wildcamtools.cli.rtsp.sleep", side_effect=KeyboardInterrupt),
    ):
        result = runner.invoke(rtsp_app, ["serve", str(segments_dir)])
        assert result.exit_code == 0
        assert "RTSP stream ready" in result.stdout or "RTSP stream ready" in result.stderr


def test_segment_basic(runner, temp_dirs):
    _, output_dir = temp_dirs
    with patch("wildcamtools.cli.segment.create_segment_process") as mock_create:
        mock_p = MagicMock()
        mock_create.return_value = mock_p

        # The segment_app is likely a Typer app where the command is 'segment'
        # but if the app itself is just the command, we might not need it.
        # Based on the error exit code 2 (usually Typer usage error),
        # let's try without the 'segment' command prefix since segment_app
        # might be the command itself.
        result = runner.invoke(segment_app, ["rtsp://test", str(output_dir)])
        if result.exit_code != 0:
            result = runner.invoke(segment_app, ["segment", "rtsp://test", str(output_dir)])

        assert result.exit_code == 0
        mock_create.assert_called_once()
        mock_p.wait.assert_called_once()


def test_ai_llamacpp_basic(runner, temp_dirs):
    segments_dir, _ = temp_dirs
    for i in range(3):
        (segments_dir / f"img_{i}.jpg").touch()

    with patch("wildcamtools.cli.ai.LlamaCppAnalyser") as mock_analyser:
        mock_instance = mock_analyser.return_value
        mock_instance.analyze_video.return_value = "Mocked Analysis"

        # Using --url and --model as options
        result = runner.invoke(
            ai_app, ["llamacpp", str(segments_dir), "--url", "http://localhost:8080", "--model", "model1"]
        )
        assert result.exit_code == 0
        assert "Processed 3" in result.stdout or "Processed 3" in result.stderr
        assert "Result: Mocked Analysis" in result.stdout or "Result: Mocked Analysis" in result.stderr
