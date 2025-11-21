class FrameError(Exception):
    """Exception raised when related to video frames."""


class CannotSeekVideoError(FrameError):
    """Exception raised when trying to seek in a video  that does not suppport seeking."""

    def __init__(self) -> None:
        super().__init__("Unable to seek video file")


class RTSPError(Exception):
    """Exception raised when related to RTSP streaming."""


class RTSPOpenError(RTSPError):
    """Exception raised when unable to open an RTSP stream."""

    def __init__(self, url: str) -> None:
        super().__init__(f"Unable to open stream: {url}")


class RTSPCloseTimeoutError(RTSPError):
    """Exception raised when unable to close an RTSP stream within the specified timeout."""

    def __init__(self, url: str) -> None:
        super().__init__(f"Unable to close stream: {url}")
