from pathlib import Path

from wildcamtools.lib.stats import (
    Colourspace,
    get_video_stats,
)


def test_get_video_stats(video_path: Path) -> None:
    stats = get_video_stats(video_path)

    assert stats.x == 3840
    assert stats.y == 2160
    assert stats.colourspace == Colourspace.RGB
    assert stats.fps == 30
    assert stats.frame_count == 150
    assert stats.shape == (2160, 3840, 3)
    assert stats.nbytes == 2160 * 3840 * 3
    assert stats.frame_duration == int(1000 / 30)
