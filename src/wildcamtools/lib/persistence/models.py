from typing import Any

from sqlalchemy import JSON, Column, UniqueConstraint
from sqlalchemy.orm import Mapped
from sqlmodel import Field, Relationship, SQLModel


class Resolution(SQLModel, table=True):
    __tablename__ = "resolution"

    id: int | None = Field(default=None, primary_key=True)
    width: int = Field(ge=1)
    height: int = Field(ge=1)

    __table_args__ = (UniqueConstraint("width", "height"),)

    videos: Mapped[list["Video"]] = Relationship(back_populates="resolution", cascade_delete=True)
    processing_steps: Mapped[list["ProcessingStep"]] = Relationship(
        back_populates="output_resolution", cascade_delete=True
    )


class ProcessingStep(SQLModel, table=True):
    __tablename__ = "processing_step"

    id: int | None = Field(default=None, primary_key=True)
    output_resolution_id: int | None = Field(default=None, foreign_key="resolution.id")
    step_type: str = Field(max_length=100)
    config_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    output_resolution: Mapped["Resolution"] = Relationship(back_populates="processing_steps")
    processing_nodes: Mapped[list["ProcessingNode"]] = Relationship(back_populates="step", cascade_delete=True)


class ProcessingNode(SQLModel, table=True):
    __tablename__ = "processing_node"

    id: int | None = Field(default=None, primary_key=True)
    step_id: int = Field(foreign_key="processing_step.id")
    parent_node_id: int | None = Field(default=None, foreign_key="processing_node.id")
    run_id: int = Field(ge=1)

    step: Mapped["ProcessingStep"] = Relationship(back_populates="processing_nodes")
    parent: Mapped["ProcessingNode"] = Relationship(
        back_populates="children", sa_relationship_kwargs={"remote_side": "ProcessingNode.id"}
    )
    children: Mapped[list["ProcessingNode"]] = Relationship(back_populates="parent", cascade_delete=True)
    frame_versions: Mapped[list["FrameVersion"]] = Relationship(back_populates="node", cascade_delete=True)


class FrameVersion(SQLModel, table=True):
    __tablename__ = "frame_version"

    id: int | None = Field(default=None, primary_key=True)
    node_id: int = Field(foreign_key="processing_node.id", index=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    frame_set_id: int | None = Field(default=None, foreign_key="frame_set.id")
    frame_no: int = Field(ge=0)
    is_kept: bool = Field(default=False)
    motion_proportion: float = Field(ge=0.0, le=1.0)
    crop_x1: int = Field(default=0)
    crop_y1: int = Field(default=0)
    crop_x2: int = Field(default=0)
    crop_y2: int = Field(default=0)

    __table_args__ = (UniqueConstraint("node_id", "video_id", "frame_no"),)

    node: Mapped["ProcessingNode"] = Relationship(back_populates="frame_versions")
    video: Mapped["Video"] = Relationship(back_populates="frame_versions")
    frame_set: Mapped["FrameSet"] = Relationship(back_populates="frame_versions")


class Video(SQLModel, table=True):
    __tablename__ = "video"

    id: int | None = Field(default=None, primary_key=True)
    resolution_id: int = Field(foreign_key="resolution.id", index=True)
    filename: str = Field(max_length=255)
    filepath: str = Field(max_length=500, index=True, unique=True)

    __table_args__ = (UniqueConstraint("resolution_id", "filename"),)

    resolution: Mapped["Resolution"] = Relationship(back_populates="videos")
    frame_versions: Mapped[list["FrameVersion"]] = Relationship(back_populates="video", cascade_delete=True)
    human_labels: Mapped[list["HumanLabel"]] = Relationship(back_populates="video", cascade_delete=True)


class FrameSet(SQLModel, table=True):
    __tablename__ = "frame_set"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(max_length=255, index=True, unique=True)

    frame_versions: Mapped[list["FrameVersion"]] = Relationship(back_populates="frame_set", cascade_delete=True)
    ai_labels: Mapped[list["AILabel"]] = Relationship(back_populates="frame_set", cascade_delete=True)


class AILabel(SQLModel, table=True):
    __tablename__ = "ai_label"

    id: int | None = Field(default=None, primary_key=True)
    frame_set_id: int = Field(foreign_key="frame_set.id", index=True)
    label_text: str = Field(max_length=500)

    __table_args__ = (UniqueConstraint("frame_set_id", "label_text"),)

    frame_set: Mapped["FrameSet"] = Relationship(back_populates="ai_labels")


class HumanLabel(SQLModel, table=True):
    __tablename__ = "human_label"

    id: int | None = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    label_text: str = Field(max_length=500)

    __table_args__ = (UniqueConstraint("video_id", "label_text"),)

    video: Mapped["Video"] = Relationship(back_populates="human_labels")
