from pathlib import Path


def is_stream_url(input_: str | Path) -> bool:
    """Check if input is a stream URL (RTSP, HTTP, etc.) vs a file path.

    Stream URLs are identified by protocol prefixes. Inputs without a recognized
    protocol prefix are treated as file paths.

    Currently supported protocols:
    - rtsp:// (Real Time Streaming Protocol)
    - rtmp:// (Real Time Messaging Protocol)
    - http:// (HTTP streaming)
    - https:// (HTTPS streaming)

    TODO: Add support for additional protocols as needed:
    - udp:// (UDP streaming)
    - srt:// (Secure Reliable Transport)
    - webrtc:// (WebRTC)
    - rtp:// (Real-time Transport Protocol)

    Args:
        input_: Path or URL string to check

    Returns:
        True if input is a stream URL, False if it's a file path
    """
    input_str = str(input_)
    return input_str.startswith(("rtsp://", "rtmp://", "http://", "https://"))
