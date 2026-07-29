import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from wildcamtools.lib.states import WatcherTransitionMetrics


class MotionDetectionConfig(BaseModel):
    """Configuration for motion detection parameters.

    Attributes:
        history: MOG2 background subtractor history in seconds (converted to frames at target FPS)
        threshold: Motion detection threshold (< 128)
        kernel_size: Kernel size as proportion (0.0 to 1.0)
        scale: Frame scale factor (< 1.0)
        fps: Target frames per second (>= 1.0)
        segment_duration: Segment duration in seconds

    """

    history: Annotated[float, Field(strict=True, ge=0.0, description="MOG2 history in seconds")] = 10.0
    threshold: Annotated[int, Field(strict=True, ge=0, lt=128, description="Motion detection threshold")] = 16
    kernel_size: Annotated[float, Field(strict=True, ge=0.0, le=1.0, description="Kernel size as proportion")] = 0.005
    scale: Annotated[float, Field(strict=True, gt=0.0, le=1.0, description="Frame scale factor")] = 0.25
    fps: Annotated[float, Field(strict=True, ge=1.0, description="Target frames per second")] = 5.0
    segment_duration: Annotated[int, Field(strict=True, gt=0, description="Segment duration in seconds")] = 15


class WatcherTransitionMetricsConfig(BaseModel):
    """Configuration for state machine transition thresholds.

    All duration fields are in seconds (wall-clock time of the frame).

    Note: Default values here are more conservative/sensitive than the base
    WatcherTransitionMetrics defaults. These config defaults are intended for
    production wildlife monitoring where false positives should be minimized.
    The base class defaults are more permissive for testing/development.

    Attributes:
        preparing_duration: Initial warm-up duration in seconds
        green_to_amber_motion_min: Minimum motion proportion to transition GREEN -> AMBER
        amber_to_green_proportion_max: Maximum motion proportion to transition AMBER -> GREEN
        amber_to_red_duration: Duration in AMBER state before transitioning to RED
        red_to_red_amber_proportion_max: Maximum motion proportion to transition RED -> RED_AMBER
        red_amber_to_red_proportion_min: Minimum motion proportion to transition RED_AMBER -> RED
        red_amber_to_green_duration: Duration in RED_AMBER state before transitioning to GREEN

    """

    preparing_duration: Annotated[float, Field(strict=True, ge=0.0)] = 10.0
    green_to_amber_motion_min: Annotated[float, Field(strict=True, ge=0.0, le=1.0)] = 0.01
    amber_to_green_proportion_max: Annotated[float, Field(strict=True, ge=0.0, le=1.0)] = 0.0075
    amber_to_red_duration: Annotated[float, Field(strict=True, ge=0.0)] = 5.0
    red_to_red_amber_proportion_max: Annotated[float, Field(strict=True, ge=0.0, le=1.0)] = 0.0075
    red_amber_to_red_proportion_min: Annotated[float, Field(strict=True, ge=0.0, le=1.0)] = 0.01
    red_amber_to_green_duration: Annotated[float, Field(strict=True, ge=0.0)] = 5.0

    def to_transition_metrics(self) -> "WatcherTransitionMetrics":
        """Convert to WatcherTransitionMetrics instance."""
        from wildcamtools.lib.states import WatcherTransitionMetrics

        return WatcherTransitionMetrics(
            preparing_duration=self.preparing_duration,
            green_to_amber_motion_min=self.green_to_amber_motion_min,
            amber_to_green_proportion_max=self.amber_to_green_proportion_max,
            amber_to_red_duration=self.amber_to_red_duration,
            red_to_red_amber_proportion_max=self.red_to_red_amber_proportion_max,
            red_amber_to_red_proportion_min=self.red_amber_to_red_proportion_min,
            red_amber_to_green_duration=self.red_amber_to_green_duration,
        )


class WatchConfig(BaseModel):
    """Configuration for the watch command.

    This configuration file contains motion detection and state machine parameters.
    Deployment-specific paths (segments_dir, output_dir) should be provided via
    command-line arguments, not in this config file.

    Attributes:
        rtsp_stream: RTSP URL or file path to process
        keep_count: Number of segments to keep in segments directory
        offset_start: Lookback duration in seconds before motion detected
        offset_end: Lookahead duration in seconds after motion ends
        motion_detection: Motion detection parameters
        transition_metrics: State machine transition thresholds
        motion_mask: Optional path to motion mask image file

    Example:
        ```json
        {
            "rtsp_stream": "${WCT_RTSP}",
            "keep_count": 4,
            "offset_start": 10.0,
            "offset_end": 10.0,
            "motion_detection": {
                "history": 10.0,
                "threshold": 16,
                "kernel_size": 0.005,
                "scale": 0.25,
                "fps": 5.0,
                "segment_duration": 15
            },
            "transition_metrics": {
                "preparing_duration": 10.0,
                "green_to_amber_motion_min": 0.01,
                "amber_to_green_proportion_max": 0.0075,
                "amber_to_red_duration": 5.0,
                "red_to_red_amber_proportion_max": 0.0075,
                "red_amber_to_red_proportion_min": 0.01,
                "red_amber_to_green_duration": 5.0
            }
        }
        ```

    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rtsp_stream": "${WCT_RTSP}",
                "keep_count": 4,
                "offset_start": 10.0,
                "offset_end": 10.0,
                "motion_detection": {
                    "history": 10.0,
                    "threshold": 16,
                    "kernel_size": 0.005,
                    "scale": 0.25,
                    "fps": 5.0,
                    "segment_duration": 15,
                },
                "transition_metrics": {
                    "preparing_duration": 10.0,
                    "green_to_amber_motion_min": 0.01,
                    "amber_to_green_proportion_max": 0.0075,
                    "amber_to_red_duration": 5.0,
                    "red_to_red_amber_proportion_max": 0.0075,
                    "red_amber_to_red_proportion_min": 0.01,
                    "red_amber_to_green_duration": 5.0,
                },
            },
        },
    )

    rtsp_stream: Annotated[str, Field(min_length=1, description="RTSP URL or file path to process")]
    keep_count: Annotated[int, Field(strict=True, ge=0, description="Number of segments to keep")] = 4
    offset_start: Annotated[float, Field(strict=True, ge=0.0, description="Lookback duration in seconds")] = 10.0
    offset_end: Annotated[float, Field(strict=True, ge=0.0, description="Lookahead duration in seconds")] = 10.0
    motion_detection: MotionDetectionConfig = Field(default_factory=MotionDetectionConfig)
    transition_metrics: WatcherTransitionMetricsConfig = Field(default_factory=WatcherTransitionMetricsConfig)
    motion_mask: Annotated[Path | None, Field(description="Optional path to motion mask image file")] = None

    @classmethod
    def _resolve_env_vars(cls, v: str) -> str:
        """Resolve all ${VAR} patterns in a string.

        Args:
            v: String potentially containing ${VAR} patterns.

        Returns:
            String with all environment variables resolved.

        Raises:
            ValueError: If an environment variable is not set.

        """

        def replacer(match: re.Match[str]) -> str:
            env_var = match.group(1)
            resolved = os.environ.get(env_var)
            if resolved is None:
                raise ValueError(f"Environment variable {env_var} is not set")
            return resolved

        return re.sub(r"\$\{([^}]+)\}", replacer, v)

    @field_validator("rtsp_stream", mode="before")
    @classmethod
    def resolve_rtsp_stream(cls, v: object) -> str:
        """Resolve environment variable references in rtsp_stream."""
        if not isinstance(v, str):
            raise TypeError(f"rtsp_stream must be a string, got {type(v).__name__}")
        return cls._resolve_env_vars(v)

    @field_validator("motion_mask", mode="before")
    @classmethod
    def resolve_motion_mask(cls, v: object) -> Path | None:
        """Convert motion_mask to Path if provided."""
        if v is None:
            return None
        if not isinstance(v, str):
            raise TypeError(f"motion_mask must be a string or None, got {type(v).__name__}")
        resolved = cls._resolve_env_vars(v)
        return Path(resolved).resolve()

    @classmethod
    def from_json(cls, path: Path) -> Self:
        """Load configuration from a JSON file.

        Args:
            path: Path to the JSON configuration file.

        Returns:
            WatchConfig instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            pydantic.ValidationError: If the config file contains invalid data.

        """
        content = path.read_text()
        return cls.model_validate_json(content)

    def to_json(self, path: Path, indent: int = 2) -> None:
        """Save configuration to a JSON file.

        Args:
            path: Path to the JSON configuration file.
            indent: JSON indentation level.

        """
        json_str = self.model_dump_json(indent=indent)
        path.write_text(json_str)

    def get_mog_history(self) -> int:
        """Get MOG2 history as frame count.

        The MOG2 background subtractor's history parameter is in number of frames.
        Convert from seconds to frames at the target FPS.

        Returns:
            Frame count for MOG2 history (minimum 1).

        """
        return max(1, round(self.motion_detection.history * self.motion_detection.fps))
