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


class MotionDetectionError(Exception):
    """Exception raised when motion detection fails."""


class CannotCombineOpenWindowError(MotionDetectionError):
    """Exception raised when trying to combine an open motion window."""

    def __init__(self) -> None:
        super().__init__("Cannot combine open window")


class ProcessError(Exception):
    """Exception raised when background process operations fail."""


class ProcessTypeMismatchError(ProcessError, TypeError):
    """Exception raised when a process operation returns an unexpected type."""

    def __init__(self) -> None:
        super().__init__("ffmpeg.run_async did not return a Popen object")


class ProcessNotInitializedError(ProcessError):
    """Exception raised when process has not been initialized."""

    def __init__(self) -> None:
        super().__init__("Process must exist to be checked")


class ProcessExitedPrematurelyError(ProcessError):
    """Exception raised when process exits before becoming ready."""

    def __init__(self, returncode: int | None) -> None:
        super().__init__(f"Process exited prematurely with code {returncode}")


class ProcessReadyTimeoutError(ProcessError):
    """Exception raised when process does not become ready within timeout."""

    def __init__(self) -> None:
        super().__init__("Process did not become ready within timeout")


class VideoError(Exception):
    """Exception raised when video operations fail."""


class VideoProbeError(VideoError):
    """Exception raised when video probe fails to find a stream."""

    def __init__(self) -> None:
        super().__init__("No video stream found in probe")


class VideoNotInContextError(VideoError):
    """Exception raised when video source is used outside context manager."""

    def __init__(self) -> None:
        super().__init__("Must be used in context")


class VideoSizeNotSetError(VideoError):
    """Exception raised when video dimensions are not set."""

    def __init__(self) -> None:
        super().__init__("Must have size")


class MotionFlowError(Exception):
    """Exception raised related to motion flow."""


class InvalidAlphaError(MotionFlowError):
    """Exception raised when alpha is not between 0 and 1."""

    def __init__(self, alpha: float) -> None:
        super().__init__(f"Alpha must be between 0.0 and 1.0, got {alpha}")


class InvalidMaxMagnitudeError(MotionFlowError):
    """Exception raised when max_magnitude is not positive."""

    def __init__(self, max_magnitude: float) -> None:
        super().__init__(f"Max magnitude must be positive, got {max_magnitude}")


class FFmpegPipeClosedError(VideoError):
    """Exception raised when FFmpeg process pipe is closed."""

    def __init__(self) -> None:
        super().__init__("FFmpeg process pipe is closed")


class BoundingBoxWidthError(ValueError):
    def __init__(self) -> None:
        super().__init__("Width is not valid")


class BoundingBoxHeightError(ValueError):
    def __init__(self) -> None:
        super().__init__("Height is not valid")


class MotionMaskNotCreatedError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Motion Mask not created")


class InertiaValueError(ValueError):
    def __init__(self) -> None:
        super().__init__("Inertia is not valid")


class ExpansionValueError(ValueError):
    def __init__(self) -> None:
        super().__init__("Expansion is not valid")
