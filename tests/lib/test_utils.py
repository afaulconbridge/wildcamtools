from pathlib import Path

from wildcamtools.lib.utils import is_stream_url


def test_is_stream_url_rtsp() -> None:
    assert is_stream_url("rtsp://localhost:8554/stream") is True
    assert is_stream_url("rtsp://192.168.1.100:554/camera1") is True
    assert is_stream_url("rtsp://user:pass@hostname:554/stream") is True
    assert is_stream_url("rtsp://example.com/live/feed1") is True
    assert is_stream_url("rtsp://localhost/stream") is True


def test_is_stream_url_rtmp() -> None:
    assert is_stream_url("rtmp://localhost/live/stream") is True
    assert is_stream_url("rtmp://streaming.example.com/app/key") is True
    assert is_stream_url("rtmp://192.168.1.50:1935/live") is True
    assert is_stream_url("rtmp://cdn.service.tv/live/camera1") is True


def test_is_stream_url_http() -> None:
    assert is_stream_url("http://localhost:8080/stream") is True
    assert is_stream_url("http://example.com/video.m3u8") is True
    assert is_stream_url("http://192.168.1.100:8080/hls/stream.m3u8") is True


def test_is_stream_url_https() -> None:
    assert is_stream_url("https://localhost:8443/stream") is True
    assert is_stream_url("https://cdn.example.com/live/feed.m3u8") is True
    assert is_stream_url("https://streaming.service.com/hls/playlist.m3u8") is True


def test_is_stream_url_file_paths() -> None:
    assert is_stream_url("/home/user/videos/camera1.mp4") is False
    assert is_stream_url("/var/lib/wildcam/recordings/clip.avi") is False
    assert is_stream_url("./recordings/test.mkv") is False
    assert is_stream_url("../data/video.mov") is False
    assert is_stream_url("video.mp4") is False
    assert is_stream_url("clip.webm") is False
    assert is_stream_url(Path("/absolute/path/to/video.mp4")) is False
    assert is_stream_url(Path("relative/path/video.mkv")) is False


def test_is_stream_url_file_extensions() -> None:
    assert is_stream_url("/path/to/file.mp4") is False
    assert is_stream_url("/path/to/file.avi") is False
    assert is_stream_url("/path/to/file.mkv") is False
    assert is_stream_url("/path/to/file.mov") is False
    assert is_stream_url("/path/to/file.webm") is False
    assert is_stream_url("/path/to/file.m3u8") is False
    assert is_stream_url("/path/to/file.ts") is False


def test_is_stream_url_edge_cases() -> None:
    assert is_stream_url("rtsp_not_a_protocol://file.mp4") is False
    assert is_stream_url("http_not_a_protocol://file.mp4") is False
    assert is_stream_url("my_rtsp_stream.mp4") is False
    assert is_stream_url("camera1-rtsp-backup.mp4") is False
    assert is_stream_url("https-backup-video.mkv") is False
    assert is_stream_url("") is False
    assert is_stream_url("just-a-filename") is False
    assert is_stream_url("RTSP://uppercase.mp4") is False
    assert is_stream_url("RtSp://mixedcase.mp4") is False
