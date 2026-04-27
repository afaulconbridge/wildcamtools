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


class PyAVError(Exception):
    """Base exception for PyAV/FFmpeg-related errors."""


class VideoReadError(PyAVError):
    """Exception raised when reading video frames fails."""

    def __init__(self, filename: str, operation: str, details: str | None = None) -> None:
        message = f"Failed to read from '{filename}' during {operation}"
        if details:
            message += f": {details}"
        super().__init__(message)


class VideoWriteError(PyAVError):
    """Exception raised when writing video frames fails."""

    def __init__(self, filename: str, operation: str, details: str | None = None) -> None:
        message = f"Failed to write to '{filename}' during {operation}"
        if details:
            message += f": {details}"
        super().__init__(message)


class ContainerError(PyAVError):
    """Exception raised when container operations fail."""

    def __init__(self, filename: str, operation: str, details: str | None = None) -> None:
        message = f"Container operation failed for '{filename}' during {operation}"
        if details:
            message += f": {details}"
        super().__init__(message)


class CodecError(PyAVError):
    """Exception raised when codec operations fail."""

    def __init__(self, filename: str, codec_name: str, details: str | None = None) -> None:
        message = f"Codec '{codec_name}' failed for '{filename}'"
        if details:
            message += f": {details}"
        super().__init__(message)


class StreamNotFoundError(PyAVError):
    """Exception raised when a stream is not found in a container."""

    def __init__(self, filename: str, stream_type: str = "video") -> None:
        super().__init__(f"No {stream_type} stream found in '{filename}'")


def translate_av_error(error: Exception, filename: str = "", operation: str = "operation") -> PyAVError:  # noqa: C901
    """Translate a PyAV/FFmpeg exception to a domain-specific exception.

    Maps av.error.FFmpegError and related exceptions to appropriate
    domain-specific PyAVError subclasses based on error context.

    Args:
        error: The original exception from PyAV/FFmpeg
        filename: The file being operated on
        operation: Description of the operation being performed

    Returns:
        A PyAVError subclass with contextual information
    """
    import av.error

    error_name = type(error).__name__
    error_str = str(error)

    if isinstance(error, (av.error.EOFError, EOFError)):
        return VideoReadError(filename, operation, "End of file reached")

    if isinstance(error, (av.error.TimeoutError, TimeoutError)):
        return VideoReadError(filename, operation, "Operation timed out")

    if isinstance(error, av.error.FFmpegError):
        if "No such file" in error_str or "Operation not permitted" in error_str:
            return ContainerError(filename, operation, error_str)

        if "Invalid data format" in error_str or "Invalid argument" in error_str:
            return ContainerError(filename, operation, error_str)

        if "Stream not found" in error_str or "Stream does not exist" in error_str:
            return StreamNotFoundError(filename, "video")

        if "Decoder not found" in error_str or "Encoder not found" in error_str:
            return CodecError(filename, "unknown", error_str)

        if "Error while decoding" in error_str:
            return VideoReadError(filename, operation, error_str)

        if "Error while encoding" in error_str:
            return VideoWriteError(filename, operation, error_str)

        return ContainerError(filename, operation, f"{error_name}: {error_str}")

    if isinstance(error, ValueError):
        return CodecError(filename, "unknown", f"{error_str}")

    return PyAVError(f"{error_name}: {error_str} (file: {filename}, operation: {operation})")
