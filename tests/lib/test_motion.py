from collections.abc import Generator

import numpy as np

from wildcamtools.lib.motion import MogMotion


def test_motion_mog(video_frame_generator: Generator[np.ndarray]):

    motion_mog = MogMotion(history=500, threshold=16, detect_shadows=False, kernel_size=3)

    for frame in video_frame_generator():
        motion_mog.handle(frame)

        prop = motion_mog.get_motion_proportion()
        print(f"Frame {frame.frame_no} : {prop * 100:.2f}%")
        assert 0.0 <= prop <= 1.0
