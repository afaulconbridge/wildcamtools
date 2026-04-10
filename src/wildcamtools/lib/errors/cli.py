import click
import typer


class MotionMaskError(typer.BadParameter):
    """Exception raised when motion mask validation fails."""


class MotionMaskNotExistsError(MotionMaskError):
    """Exception raised when motion mask file does not exist."""

    def __init__(self, path: str) -> None:
        super().__init__(f"Motion mask file does not exist: {path}")


class MotionMaskNotFileError(MotionMaskError):
    """Exception raised when motion mask path is not a file."""

    def __init__(self, path: str) -> None:
        super().__init__(f"Motion mask path is not a file: {path}")


class MotionMaskNotReadableError(MotionMaskError):
    """Exception raised when motion mask file is not readable."""

    def __init__(self, path: str) -> None:
        super().__init__(f"Motion mask file is not readable: {path}")


class SegmentError(click.BadArgumentUsage):
    """Exception raised when segment operations fail."""


class OutputNotDirectoryError(SegmentError):
    """Exception raised when output path is not a directory."""

    def __init__(self) -> None:
        super().__init__("Output must be a directory that can be created")
