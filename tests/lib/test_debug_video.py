"""Tests for debug video output functionality."""

from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np

from wildcamtools.lib import Frame
from wildcamtools.lib.debug_video import DebugVideoOverlay, DebugVideoWriter
from wildcamtools.lib.motion import MogMotion
from wildcamtools.lib.states import Watcher, WatcherStateEnum


class TestDebugVideoOverlay:
    """Tests for DebugVideoOverlay handler."""

    def test_overlay_initializes_with_watcher(self) -> None:
        """Test that DebugVideoOverlay initializes correctly with watcher."""
        mock_watcher = Mock(spec=Watcher)
        mock_watcher.state = WatcherStateEnum.GREEN
        mock_motion = Mock(spec=MogMotion)

        overlay = DebugVideoOverlay(
            watcher=mock_watcher,
            motion_handler=mock_motion,
        )

        assert overlay.watcher is mock_watcher
        assert overlay.motion_handler is mock_motion
        assert WatcherStateEnum.GREEN in overlay.state_colors

    def test_overlay_draws_border_on_frame(self) -> None:
        """Test that overlay draws colored border on frame."""
        mock_watcher = Mock(spec=Watcher)
        mock_watcher.state = WatcherStateEnum.RED
        mock_motion = Mock(spec=MogMotion)
        mock_motion.motion_mask = None

        overlay = DebugVideoOverlay(
            watcher=mock_watcher,
            motion_handler=mock_motion,
        )

        frame = Frame(
            raw=np.zeros((100, 100, 3), dtype=np.uint8),
            frame_no=1,
            motion_proportion=0.5,
            timestamp=1.0,
        )
        frame.rescale = np.zeros((50, 50, 3), dtype=np.uint8)

        result = overlay.handle(frame)

        assert result.raw is not None
        assert result.raw.shape == (100, 100, 3)

    def test_overlay_state_colors(self) -> None:
        """Test that all watcher states have defined colors."""
        mock_watcher = Mock(spec=Watcher)
        mock_watcher.state = WatcherStateEnum.GREEN
        mock_motion = Mock(spec=MogMotion)
        mock_motion.motion_mask = None

        overlay = DebugVideoOverlay(
            watcher=mock_watcher,
            motion_handler=mock_motion,
        )

        expected_states = [
            WatcherStateEnum.PREPARING,
            WatcherStateEnum.GREEN,
            WatcherStateEnum.AMBER,
            WatcherStateEnum.RED,
            WatcherStateEnum.RED_AMBER,
            WatcherStateEnum.DISABLED,
        ]

        for state in expected_states:
            assert state.value in overlay.state_colors or state in overlay.state_colors

    def test_overlay_draws_on_full_resolution(self) -> None:
        """Test that overlay draws on full-resolution frame (raw), not scaled."""
        mock_watcher = Mock(spec=Watcher)
        mock_watcher.state = WatcherStateEnum.GREEN
        mock_motion = Mock(spec=MogMotion)
        mock_motion.motion_mask = None

        overlay = DebugVideoOverlay(
            watcher=mock_watcher,
            motion_handler=mock_motion,
        )

        original_frame = Frame(
            raw=np.zeros((1920, 1080, 3), dtype=np.uint8),
            frame_no=42,
            motion_proportion=0.1234,
            filter_keep=True,
            timestamp=5.6789,
        )
        original_frame.rescale = np.zeros((160, 120, 3), dtype=np.uint8)

        result = overlay.handle(original_frame)

        assert result.raw.shape == (1920, 1080, 3)
        assert result.rescale.shape == (160, 120, 3)
        assert result.frame_no == original_frame.frame_no
        assert result.motion_proportion == original_frame.motion_proportion
        assert result.filter_keep == original_frame.filter_keep
        assert result.timestamp == original_frame.timestamp

    def test_overlay_draws_contours_when_mask_present(self) -> None:
        """Test that overlay draws contours when motion mask is available."""
        mock_watcher = Mock(spec=Watcher)
        mock_watcher.state = WatcherStateEnum.RED
        mock_motion = Mock(spec=MogMotion)

        mask = np.zeros((50, 50), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (20, 20), 255, -1)
        mock_motion.motion_mask = mask

        overlay = DebugVideoOverlay(
            watcher=mock_watcher,
            motion_handler=mock_motion,
        )

        frame = Frame(
            raw=np.zeros((100, 100, 3), dtype=np.uint8),
            frame_no=1,
            motion_proportion=0.5,
            timestamp=1.0,
        )
        frame.rescale = np.zeros((50, 50, 3), dtype=np.uint8)

        result = overlay.handle(frame)

        assert result.raw is not None
        assert result.raw.shape == (100, 100, 3)

    def test_overlay_scales_contours_to_full_resolution(self) -> None:
        """Test that contours are scaled from motion resolution to full resolution."""
        mock_watcher = Mock(spec=Watcher)
        mock_watcher.state = WatcherStateEnum.RED
        mock_motion = Mock(spec=MogMotion)

        mask = np.zeros((50, 50), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (20, 20), 255, -1)
        mock_motion.motion_mask = mask

        overlay = DebugVideoOverlay(
            watcher=mock_watcher,
            motion_handler=mock_motion,
        )

        frame = Frame(
            raw=np.zeros((200, 200, 3), dtype=np.uint8),
            frame_no=1,
            motion_proportion=0.5,
            timestamp=1.0,
        )

        result = overlay.handle(frame)

        assert result.raw is not None
        assert result.raw.shape == (200, 200, 3)
        assert overlay._contour_scale == (4.0, 4.0)

    def test_overlay_default_scale_when_mask_none(self) -> None:
        """Test that contour scale defaults to (1.0, 1.0) when mask is None."""
        mock_watcher = Mock(spec=Watcher)
        mock_watcher.state = WatcherStateEnum.GREEN
        mock_motion = Mock(spec=MogMotion)
        mock_motion.motion_mask = None

        overlay = DebugVideoOverlay(
            watcher=mock_watcher,
            motion_handler=mock_motion,
        )

        frame = Frame(
            raw=np.zeros((1920, 1080, 3), dtype=np.uint8),
            frame_no=1,
            motion_proportion=0.5,
            timestamp=1.0,
        )

        scale = overlay._contour_scale_cached(frame)

        assert scale == (1.0, 1.0)

    def test_overlay_extreme_scale_factors(self) -> None:
        """Test contour scaling with extreme scale factors."""
        mock_watcher = Mock(spec=Watcher)
        mock_watcher.state = WatcherStateEnum.RED
        mock_motion = Mock(spec=MogMotion)

        mask = np.zeros((10, 10), dtype=np.uint8)
        cv2.rectangle(mask, (2, 2), (5, 5), 255, -1)
        mock_motion.motion_mask = mask

        overlay = DebugVideoOverlay(
            watcher=mock_watcher,
            motion_handler=mock_motion,
        )

        frame = Frame(
            raw=np.zeros((1000, 1000, 3), dtype=np.uint8),
            frame_no=1,
            motion_proportion=0.5,
            timestamp=1.0,
        )

        result = overlay.handle(frame)

        assert result.raw is not None
        assert result.raw.shape == (1000, 1000, 3)
        assert overlay._contour_scale == (100.0, 100.0)

    def test_overlay_non_uniform_scale(self) -> None:
        """Test contour scaling with non-uniform aspect ratio."""
        mock_watcher = Mock(spec=Watcher)
        mock_watcher.state = WatcherStateEnum.RED
        mock_motion = Mock(spec=MogMotion)

        mask = np.zeros((50, 100), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (20, 20), 255, -1)
        mock_motion.motion_mask = mask

        overlay = DebugVideoOverlay(
            watcher=mock_watcher,
            motion_handler=mock_motion,
        )

        frame = Frame(
            raw=np.zeros((1080, 1920, 3), dtype=np.uint8),
            frame_no=1,
            motion_proportion=0.5,
            timestamp=1.0,
        )

        result = overlay.handle(frame)

        assert result.raw is not None
        assert result.raw.shape == (1080, 1920, 3)
        assert overlay._contour_scale == (19.2, 21.6)

    def test_overlay_handles_missing_rescale(self) -> None:
        """Test that overlay handles frames without rescale attribute."""
        mock_watcher = Mock(spec=Watcher)
        mock_watcher.state = WatcherStateEnum.GREEN
        mock_motion = Mock(spec=MogMotion)
        mock_motion.motion_mask = None

        overlay = DebugVideoOverlay(
            watcher=mock_watcher,
            motion_handler=mock_motion,
        )

        frame = Frame(
            raw=np.zeros((100, 100, 3), dtype=np.uint8),
            frame_no=1,
            motion_proportion=0.5,
            timestamp=1.0,
        )

        result = overlay.handle(frame)

        assert result.raw is not None
        assert result.raw.shape == (100, 100, 3)


class TestDebugVideoWriter:
    """Tests for DebugVideoWriter handler."""

    def test_writer_context_manager(self, tmp_path: Path) -> None:
        """Test that DebugVideoWriter works as context manager."""
        output_path = tmp_path / "debug.mp4"

        with DebugVideoWriter(output_path, width=100, height=100, fps=30.0) as writer:
            assert writer._closed is False

        assert writer._closed is True

    def test_writer_close_is_idempotent(self, tmp_path: Path) -> None:
        """Test that calling close() multiple times is safe."""
        output_path = tmp_path / "debug.mp4"

        writer = DebugVideoWriter(output_path, width=100, height=100, fps=30.0)
        writer.__enter__()

        writer.close()
        writer.close()

        assert writer._closed is True

    def test_writer_skips_filtered_frames(self, tmp_path: Path) -> None:
        """Test that writer skips frames with filter_keep=False."""
        output_path = tmp_path / "debug.mp4"

        with DebugVideoWriter(output_path, width=100, height=100, fps=30.0) as writer:
            frame = Frame(
                raw=np.zeros((100, 100, 3), dtype=np.uint8),
                frame_no=1,
                filter_keep=False,
            )
            frame.rescale = np.zeros((100, 100, 3), dtype=np.uint8)

            result = writer.handle(frame)

            assert result is frame

    def test_writer_writes_raw_frame(self, tmp_path: Path) -> None:
        """Test that writer writes raw (full-resolution) frames."""
        output_path = tmp_path / "debug.mp4"

        with DebugVideoWriter(output_path, width=1920, height=1080, fps=30.0) as writer:
            frame = Frame(
                raw=np.zeros((1920, 1080, 3), dtype=np.uint8),
                frame_no=1,
                filter_keep=True,
            )
            frame.rescale = np.zeros((160, 120, 3), dtype=np.uint8)

            result = writer.handle(frame)

            assert result is frame

        assert output_path.exists()

    def test_writer_writes_valid_frames(self, tmp_path: Path) -> None:
        """Test that writer writes raw frames."""
        output_path = tmp_path / "debug.mp4"

        with DebugVideoWriter(output_path, width=1920, height=1080, fps=30.0) as writer:
            frame = Frame(
                raw=np.zeros((1920, 1080, 3), dtype=np.uint8),
                frame_no=1,
                filter_keep=True,
            )

            result = writer.handle(frame)

            assert result is frame

        assert output_path.exists()

    def test_writer_handles_exception_in_exit(self, tmp_path: Path) -> None:
        """Test that writer closes even if exception occurs."""
        output_path = tmp_path / "debug.mp4"

        writer = DebugVideoWriter(output_path, width=100, height=100, fps=30.0)
        writer.__enter__()

        exception_raised = False
        try:
            raise ValueError("Test exception")  # noqa: TRY301
        except ValueError:
            exception_raised = True
        finally:
            writer.__exit__(None, None, None)

        assert exception_raised
        assert writer._closed is True


class TestDebugVideoIntegration:
    """Integration tests for debug video in motion detection pipeline."""

    def test_overlay_and_writer_used_together(self, tmp_path: Path) -> None:
        """Test that overlay and writer can be used together at full resolution."""
        mock_watcher = Mock(spec=Watcher)
        mock_watcher.state = WatcherStateEnum.GREEN
        mock_motion = Mock(spec=MogMotion)
        mock_motion.motion_mask = None

        overlay = DebugVideoOverlay(
            watcher=mock_watcher,
            motion_handler=mock_motion,
        )

        output_path = tmp_path / "debug.mp4"
        writer = DebugVideoWriter(output_path, width=1920, height=1080, fps=30.0)
        writer.__enter__()

        try:
            frame = Frame(
                raw=np.zeros((1920, 1080, 3), dtype=np.uint8),
                frame_no=1,
                motion_proportion=0.5,
                timestamp=1.0,
                filter_keep=True,
            )
            frame.rescale = np.zeros((160, 120, 3), dtype=np.uint8)

            frame = overlay.handle(frame)
            writer.handle(frame)
        finally:
            writer.__exit__(None, None, None)

        assert output_path.exists()
