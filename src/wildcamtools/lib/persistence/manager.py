import logging
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload
from sqlmodel import select

from wildcamtools.lib.persistence.database import get_session
from wildcamtools.lib.persistence.models import (
    AILabel,
    FrameSet,
    FrameVersion,
    HumanLabel,
    ProcessingNode,
    ProcessingStep,
    Resolution,
    Video,
)

logger = logging.getLogger(__name__)


class PersistenceManager:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def create_video(self, filepath: str, filename: str, resolution_id: int) -> Video:
        if not filepath:
            msg = "filepath cannot be empty"
            raise ValueError(msg)
        if not filename:
            msg = "filename cannot be empty"
            raise ValueError(msg)
        if resolution_id < 1:
            msg = f"resolution_id must be positive, got {resolution_id}"
            raise ValueError(msg)
        with get_session(self._engine) as session:
            try:
                video = Video(filepath=filepath, filename=filename, resolution_id=resolution_id)
                session.add(video)
                session.commit()
                session.refresh(video)
                logger.info("Created video: %s", video.filepath)
            except Exception:
                session.rollback()
                raise
            return video

    def get_video_by_path(self, filepath: str) -> Video | None:
        if not filepath:
            msg = "filepath cannot be empty"
            raise ValueError(msg)
        with get_session(self._engine) as session:
            statement = (
                select(Video)
                .where(Video.filepath == filepath)
                .options(
                    selectinload(Video.resolution),
                    selectinload(Video.frame_versions),
                    selectinload(Video.human_labels),
                )
            )
            video = session.exec(statement).first()
            return video

    def get_or_create_resolution(self, width: int, height: int) -> Resolution:
        if width < 1 or height < 1:
            msg = f"width and height must be positive, got {width}x{height}"
            raise ValueError(msg)
        with get_session(self._engine) as session:
            try:
                statement = select(Resolution).where(Resolution.width == width, Resolution.height == height)
                resolution = session.exec(statement).first()
                if resolution is None:
                    resolution = Resolution(width=width, height=height)
                    session.add(resolution)
                    session.commit()
                    session.refresh(resolution)
                    logger.info("Created resolution: %dx%d", width, height)
            except IntegrityError:
                session.rollback()
                statement = select(Resolution).where(Resolution.width == width, Resolution.height == height)
                resolution = session.exec(statement).first()
                if resolution is None:
                    raise
            except Exception:
                session.rollback()
                raise
            return resolution

    def create_processing_step(
        self, output_resolution_id: int, step_type: str, config: dict[str, Any]
    ) -> ProcessingStep:
        if output_resolution_id < 1:
            msg = f"output_resolution_id must be positive, got {output_resolution_id}"
            raise ValueError(msg)
        if not step_type:
            msg = "step_type cannot be empty"
            raise ValueError(msg)
        with get_session(self._engine) as session:
            try:
                step = ProcessingStep(
                    output_resolution_id=output_resolution_id,
                    step_type=step_type,
                    config_json=config,
                )
                session.add(step)
                session.commit()
                session.refresh(step)
                logger.info("Created processing step: %s", step.step_type)
            except Exception:
                session.rollback()
                raise
            return step

    def create_processing_node(self, step_id: int, parent_node_id: int | None, run_id: int) -> ProcessingNode:
        if step_id < 1:
            msg = f"step_id must be positive, got {step_id}"
            raise ValueError(msg)
        if parent_node_id is not None and parent_node_id < 1:
            msg = f"parent_node_id must be positive, got {parent_node_id}"
            raise ValueError(msg)
        if run_id < 1:
            msg = f"run_id must be positive, got {run_id}"
            raise ValueError(msg)
        with get_session(self._engine) as session:
            try:
                node = ProcessingNode(
                    step_id=step_id,
                    parent_node_id=parent_node_id,
                    run_id=run_id,
                )
                session.add(node)
                session.commit()
                session.refresh(node)
                logger.info("Created processing node: %d", node.id)
            except Exception:
                session.rollback()
                raise
            return node

    def create_frame_version(
        self,
        node_id: int,
        video_id: int,
        frame_no: int,
        is_kept: bool,
        motion_proportion: float,
        crop_coords: tuple[int, int, int, int] | None = None,
    ) -> FrameVersion:
        if node_id < 1:
            msg = f"node_id must be positive, got {node_id}"
            raise ValueError(msg)
        if video_id < 1:
            msg = f"video_id must be positive, got {video_id}"
            raise ValueError(msg)
        if frame_no < 0:
            msg = f"frame_no must be non-negative, got {frame_no}"
            raise ValueError(msg)
        if motion_proportion < 0.0 or motion_proportion > 1.0:
            msg = f"motion_proportion must be between 0 and 1, got {motion_proportion}"
            raise ValueError(msg)
        if crop_coords is not None:
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
            if crop_x1 < 0 or crop_y1 < 0 or crop_x2 < 0 or crop_y2 < 0:
                msg = "crop coordinates must be non-negative"
                raise ValueError(msg)
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                msg = "crop_x2 must be greater than crop_x1 and crop_y2 must be greater than crop_y1"
                raise ValueError(msg)
        with get_session(self._engine) as session:
            try:
                if crop_coords is not None:
                    crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
                else:
                    crop_x1 = crop_y1 = crop_x2 = crop_y2 = 0
                frame_version = FrameVersion(
                    node_id=node_id,
                    video_id=video_id,
                    frame_no=frame_no,
                    is_kept=is_kept,
                    motion_proportion=motion_proportion,
                    crop_x1=crop_x1,
                    crop_y1=crop_y1,
                    crop_x2=crop_x2,
                    crop_y2=crop_y2,
                )
                session.add(frame_version)
                session.commit()
                session.refresh(frame_version)
                logger.info("Created frame version: node=%d, frame=%d", node_id, frame_no)
            except Exception:
                session.rollback()
                raise
            return frame_version

    def get_frame_versions_by_node(self, node_id: int) -> list[FrameVersion]:
        if node_id < 1:
            msg = f"node_id must be positive, got {node_id}"
            raise ValueError(msg)
        with get_session(self._engine) as session:
            statement = (
                select(FrameVersion)
                .where(FrameVersion.node_id == node_id)
                .options(
                    selectinload(FrameVersion.node),
                    selectinload(FrameVersion.video),
                    selectinload(FrameVersion.frame_set),
                )
            )
            results = session.exec(statement)
            return list(results)

    def create_human_label(self, video_id: int, label_text: str) -> HumanLabel:
        if video_id < 1:
            msg = f"video_id must be positive, got {video_id}"
            raise ValueError(msg)
        if not label_text:
            msg = "label_text cannot be empty"
            raise ValueError(msg)
        with get_session(self._engine) as session:
            try:
                label = HumanLabel(video_id=video_id, label_text=label_text)
                session.add(label)
                session.commit()
                session.refresh(label)
                logger.info("Created human label for video: %d", video_id)
            except Exception:
                session.rollback()
                raise
            return label

    def create_frame_set(self, name: str) -> FrameSet:
        if not name:
            msg = "name cannot be empty"
            raise ValueError(msg)
        with get_session(self._engine) as session:
            try:
                frame_set = FrameSet(name=name)
                session.add(frame_set)
                session.commit()
                session.refresh(frame_set)
                logger.info("Created frame set: %s", name)
            except Exception:
                session.rollback()
                raise
            return frame_set

    def create_ai_label(self, frame_set_id: int, label_text: str) -> AILabel:
        if frame_set_id < 1:
            msg = f"frame_set_id must be positive, got {frame_set_id}"
            raise ValueError(msg)
        if not label_text:
            msg = "label_text cannot be empty"
            raise ValueError(msg)
        with get_session(self._engine) as session:
            try:
                label = AILabel(frame_set_id=frame_set_id, label_text=label_text)
                session.add(label)
                session.commit()
                session.refresh(label)
                logger.info("Created AI label for frame set: %d", frame_set_id)
            except Exception:
                session.rollback()
                raise
            return label
