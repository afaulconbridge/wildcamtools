from datetime import datetime

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel


class VideoStat(SQLModel, table=True):
    """Video metadata statistics.

    Normalized to allow multiple videos to share the same statistics.
    """

    __tablename__ = "videostat"

    id: int | None = Field(default=None, primary_key=True)
    width: int
    height: int
    fps: float
    total_frames: int

    __table_args__ = (UniqueConstraint("width", "height", "fps", "total_frames", name="uq_videostat_metadata"),)


class Video(SQLModel, table=True):
    """Video file entity.

    References VideoStat for metadata.
    """

    __tablename__ = "video"

    filename: str = Field(primary_key=True)
    stat_id: int | None = Field(default=None, foreign_key="videostat.id", ondelete="CASCADE")
    recorded_at: datetime | None = Field(default=None, index=True)

    stat: VideoStat | None = Relationship()
    runs: list["PipelineRun"] = Relationship(back_populates="video")


class ClassificationResult(SQLModel, table=True):
    """AI classification result.

    Normalized to allow multiple runs/batches to share identical results.
    """

    __tablename__ = "classificationresult"

    id: int | None = Field(default=None, primary_key=True)
    species_name: str
    confidence: str
    is_animal_present: bool
    is_animal_unknown: bool

    __table_args__ = (
        UniqueConstraint(
            "species_name",
            "confidence",
            "is_animal_present",
            "is_animal_unknown",
            name="uq_classification_result",
        ),
    )


class PipelineRun(SQLModel, table=True):
    """Pipeline execution run.

    Links a video file to its configuration and results.
    """

    __tablename__ = "pipelinerun"

    id: int | None = Field(default=None, primary_key=True)
    video_filename: str = Field(foreign_key="video.filename", ondelete="CASCADE")
    config_json: str
    timestamp: datetime = Field(default_factory=datetime.now)
    final_result_id: int | None = Field(default=None, foreign_key="classificationresult.id", ondelete="CASCADE")

    video: Video | None = Relationship(back_populates="runs")
    final_result: ClassificationResult | None = Relationship()
    batches: list["PipelineBatch"] = Relationship(back_populates="run")


class PipelineBatch(SQLModel, table=True):
    """Batch of frames processed in a pipeline run.

    Contains frame numbers and the classification result for that batch.
    """

    __tablename__ = "pipelinebatch"

    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="pipelinerun.id", ondelete="CASCADE")
    frame_numbers: str
    result_id: int | None = Field(default=None, foreign_key="classificationresult.id", ondelete="CASCADE")

    run: PipelineRun | None = Relationship(back_populates="batches")
    result: ClassificationResult | None = Relationship()
