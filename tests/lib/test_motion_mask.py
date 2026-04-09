
import numpy as np
import pytest

from wildcamtools.lib.motion import MogMotion


def test_motion_mask_filtering():
    # Initialize motion handler with mask in constructor
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50:, :] = 255  # Mask bottom half
    motion = MogMotion(history=1, motion_mask=mask)

    # Create a binary motion frame with two blobs
    # Blob 1: Top half (should be kept)
    # Blob 2: Bottom half (should be removed)
    frame_raw = np.zeros((100, 100), dtype=np.uint8)
    frame_raw[10:20, 10:20] = 255 # Top blob
    frame_raw[60:70, 60:70] = 255 # Bottom blob

    prop = motion.get_motion_proportion(frame_raw)

    # We use pytest.approx because cv2.contourArea can be slightly
    # different from the exact pixel count of a rectangle.
    assert prop == pytest.approx(0.0081, abs=1e-4)

def test_motion_multiple_lowest_points_masked():
    # Initialize motion handler with mask in constructor
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[70, 40:60] = 255  # Mask a horizontal strip at Y=70
    motion = MogMotion(history=1, motion_mask=mask)

    frame_raw = np.zeros((100, 100), dtype=np.uint8)
    # Use a slightly larger block to ensure a proper contour is found
    frame_raw[60:71, 30:71] = 255

    prop = motion.get_motion_proportion(frame_raw)

    assert prop == 0.0

def test_motion_mask_intersects_bottom_edge():
    # Initialize motion handler
    motion = MogMotion(history=1)

    # Create a mask that only covers the MIDDLE of the bottom edge
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[70, 45:55] = 255 # Small mask in the center of the bottom edge
    motion.motion_mask = mask

    # Create a blob whose bottom edge is at Y=70 from X=30 to X=70
    # The endpoints (30, 70) and (70, 70) are NOT masked
    # But the middle (45-55, 70) IS masked
    frame_raw = np.zeros((100, 100), dtype=np.uint8)
    frame_raw[60:71, 30:71] = 255

    prop = motion.get_motion_proportion(frame_raw)

    # The contour should be removed because the mask intersects the bottom edge
    assert prop == 0.0
