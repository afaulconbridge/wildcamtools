from collections.abc import Generator

import numpy as np

from wildcamtools.lib.motion import MogMotion


def test_motion_mog(video_frame_generator: Generator[np.ndarray]):

    motion_mog = MogMotion(history=1, threshold=16, detect_shadows=False, kernel_size=3)

    for frame in video_frame_generator():
        frame = motion_mog.handle(frame)
        if frame.frame_no > 10:
            prop = frame.motion_proportion
            print(f"Frame {frame.frame_no} : {prop * 100:.2f}%")
            assert 0.0 <= prop <= 1.0
