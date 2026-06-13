import logging
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

logger = logging.getLogger(__name__)

PLACEHOLDER_WIDTH = 320
PLACEHOLDER_HEIGHT = 180


def _build_placeholder_bytes() -> bytes:
    """Return a small JPEG 'no preview' placeholder image."""
    image = np.full((PLACEHOLDER_HEIGHT, PLACEHOLDER_WIDTH, 3), 40, dtype=np.uint8)
    text = "No preview"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (PLACEHOLDER_WIDTH - text_width) // 2
    y = (PLACEHOLDER_HEIGHT + text_height) // 2
    cv2.putText(image, text, (x, y), font, scale, (200, 200, 200), thickness, cv2.LINE_AA)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("Failed to encode placeholder JPEG")
    return buffer.tobytes()


_PLACEHOLDER_BYTES: bytes = _build_placeholder_bytes()


def extract_thumbnail(video_path: Path, max_width: int = 320) -> bytes | None:
    """Extract a JPEG thumbnail from the middle frame of a video.

    The returned image keeps the source aspect ratio and has a width no greater
    than ``max_width``. Returns ``None`` if the video cannot be opened or read.
    """
    if not video_path.exists() or not video_path.is_file():
        logger.debug("Video path does not exist or is not a file: %s", video_path)
        return None

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            logger.warning("Failed to open video: %s", video_path)
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)

        ok, frame = cap.read()
        if not ok or frame is None:
            logger.warning("Failed to read frame from video: %s", video_path)
            return None

        height, width = frame.shape[:2]
        if width > max_width and width > 0:
            scale = max_width / width
            new_width = max_width
            new_height = max(int(height * scale), 1)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            logger.warning("Failed to encode thumbnail for: %s", video_path)
            return None
        return buffer.tobytes()
    finally:
        cap.release()


def thumbnail_or_placeholder(video_path: Path, max_width: int = 320) -> bytes:
    """Return a JPEG thumbnail for ``video_path`` or a placeholder image."""
    thumbnail = extract_thumbnail(video_path, max_width=max_width)
    if thumbnail is not None:
        return thumbnail
    return _PLACEHOLDER_BYTES


@st.cache_data
def cached_thumbnail(path_str: str, max_width: int) -> bytes:
    """Cached wrapper around :func:`thumbnail_or_placeholder`."""
    return thumbnail_or_placeholder(Path(path_str), max_width)
