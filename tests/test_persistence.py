import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import SQLModel

from wildcamtools.lib.persistence.database import create_engine_and_tables, get_session
from wildcamtools.lib.persistence.manager import PersistenceManager
from wildcamtools.lib.persistence.models import (
    AILabel,
    FrameVersion,
    HumanLabel,
)


@pytest.fixture
def engine():
    engine = create_engine_and_tables("sqlite:///:memory:")
    yield engine
    SQLModel.metadata.drop_all(engine)


@pytest.fixture
def pm(engine):
    return PersistenceManager(engine)


class TestResolution:
    def test_create_resolution(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        assert resolution.width == 1920
        assert resolution.height == 1080
        assert resolution.id is not None

    def test_get_existing_resolution(self, pm):
        resolution1 = pm.get_or_create_resolution(1920, 1080)
        resolution2 = pm.get_or_create_resolution(1920, 1080)
        assert resolution1.id == resolution2.id

    def test_invalid_resolution_dimensions(self, pm):
        with pytest.raises(ValueError, match="width and height must be positive"):
            pm.get_or_create_resolution(0, 1080)
        with pytest.raises(ValueError, match="width and height must be positive"):
            pm.get_or_create_resolution(1920, -1)


class TestVideo:
    def test_create_video(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        video = pm.create_video("/path/to/video.mp4", "video.mp4", resolution.id)
        assert video.filepath == "/path/to/video.mp4"
        assert video.filename == "video.mp4"
        assert video.resolution_id == resolution.id
        assert video.id is not None

    def test_get_video_by_path(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        video = pm.create_video("/path/to/video.mp4", "video.mp4", resolution.id)
        retrieved = pm.get_video_by_path("/path/to/video.mp4")
        assert retrieved is not None
        assert retrieved.id == video.id

    def test_create_video_invalid_filepath(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        with pytest.raises(ValueError, match="filepath cannot be empty"):
            pm.create_video("", "video.mp4", resolution.id)

    def test_create_video_invalid_filename(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        with pytest.raises(ValueError, match="filename cannot be empty"):
            pm.create_video("/path/to/video.mp4", "", resolution.id)

    def test_create_video_invalid_resolution_id(self, pm):
        with pytest.raises(ValueError, match="resolution_id must be positive"):
            pm.create_video("/path/to/video.mp4", "video.mp4", 0)


class TestProcessingStep:
    def test_create_processing_step(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        step = pm.create_processing_step(resolution.id, "motion_detect", {"threshold": 0.5})
        assert step.step_type == "motion_detect"
        assert step.output_resolution_id == resolution.id
        assert step.config_json == {"threshold": 0.5}
        assert step.id is not None

    def test_create_processing_step_invalid_resolution_id(self, pm):
        with pytest.raises(ValueError, match="output_resolution_id must be positive"):
            pm.create_processing_step(0, "motion_detect", {})

    def test_create_processing_step_invalid_step_type(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        with pytest.raises(ValueError, match="step_type cannot be empty"):
            pm.create_processing_step(resolution.id, "", {})


class TestProcessingNode:
    def test_create_processing_node(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        step = pm.create_processing_step(resolution.id, "motion_detect", {})
        node = pm.create_processing_node(step.id, parent_node_id=None, run_id=1)
        assert node.step_id == step.id
        assert node.parent_node_id is None
        assert node.run_id == 1
        assert node.id is not None

    def test_create_processing_node_with_parent(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        step = pm.create_processing_step(resolution.id, "motion_detect", {})
        parent_node = pm.create_processing_node(step.id, parent_node_id=None, run_id=1)
        child_node = pm.create_processing_node(step.id, parent_node_id=parent_node.id, run_id=1)
        assert child_node.parent_node_id == parent_node.id

    def test_create_processing_node_invalid_step_id(self, pm):
        with pytest.raises(ValueError, match="step_id must be positive"):
            pm.create_processing_node(0, parent_node_id=None, run_id=1)

    def test_create_processing_node_invalid_run_id(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        step = pm.create_processing_step(resolution.id, "motion_detect", {})
        with pytest.raises(ValueError, match="run_id must be positive"):
            pm.create_processing_node(step.id, parent_node_id=None, run_id=0)


class TestFrameVersion:
    def test_create_frame_version(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        video = pm.create_video("/path/to/video.mp4", "video.mp4", resolution.id)
        step = pm.create_processing_step(resolution.id, "motion_detect", {})
        node = pm.create_processing_node(step.id, parent_node_id=None, run_id=1)

        frame = pm.create_frame_version(
            node_id=node.id,
            video_id=video.id,
            frame_no=42,
            is_kept=True,
            motion_proportion=0.75,
            crop_coords=(100, 100, 800, 600),
        )

        assert frame.node_id == node.id
        assert frame.video_id == video.id
        assert frame.frame_no == 42
        assert frame.is_kept is True
        assert frame.motion_proportion == 0.75
        assert frame.crop_x1 == 100
        assert frame.crop_y1 == 100
        assert frame.crop_x2 == 800
        assert frame.crop_y2 == 600

    def test_get_frame_versions_by_node(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        video = pm.create_video("/path/to/video.mp4", "video.mp4", resolution.id)
        step = pm.create_processing_step(resolution.id, "motion_detect", {})
        node = pm.create_processing_node(step.id, parent_node_id=None, run_id=1)

        pm.create_frame_version(node.id, video.id, 1, True, 0.5)
        pm.create_frame_version(node.id, video.id, 2, True, 0.6)

        frames = pm.get_frame_versions_by_node(node.id)
        assert len(frames) == 2
        assert {f.frame_no for f in frames} == {1, 2}

    def test_create_frame_version_invalid_motion_proportion(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        video = pm.create_video("/path/to/video.mp4", "video.mp4", resolution.id)
        step = pm.create_processing_step(resolution.id, "motion_detect", {})
        node = pm.create_processing_node(step.id, parent_node_id=None, run_id=1)

        with pytest.raises(ValueError, match="motion_proportion must be between 0 and 1"):
            pm.create_frame_version(node.id, video.id, 1, True, 1.5)

    def test_create_frame_version_invalid_crop_coords(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        video = pm.create_video("/path/to/video.mp4", "video.mp4", resolution.id)
        step = pm.create_processing_step(resolution.id, "motion_detect", {})
        node = pm.create_processing_node(step.id, parent_node_id=None, run_id=1)

        with pytest.raises(ValueError, match="crop_x2 must be greater than crop_x1"):
            pm.create_frame_version(node.id, video.id, 1, True, 0.5, crop_coords=(800, 600, 100, 100))


class TestFrameSet:
    def test_create_frame_set(self, pm):
        frame_set = pm.create_frame_set("event_001")
        assert frame_set.name == "event_001"
        assert frame_set.id is not None

    def test_create_frame_set_invalid_name(self, pm):
        with pytest.raises(ValueError, match="name cannot be empty"):
            pm.create_frame_set("")


class TestAILabel:
    def test_create_ai_label(self, pm):
        frame_set = pm.create_frame_set("event_001")
        label = pm.create_ai_label(frame_set.id, "deer, night")
        assert label.frame_set_id == frame_set.id
        assert label.label_text == "deer, night"
        assert label.id is not None

    def test_create_ai_label_invalid_frame_set_id(self, pm):
        with pytest.raises(ValueError, match="frame_set_id must be positive"):
            pm.create_ai_label(0, "deer")

    def test_create_ai_label_invalid_label_text(self, pm):
        frame_set = pm.create_frame_set("event_001")
        with pytest.raises(ValueError, match="label_text cannot be empty"):
            pm.create_ai_label(frame_set.id, "")


class TestHumanLabel:
    def test_create_human_label(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        video = pm.create_video("/path/to/video.mp4", "video.mp4", resolution.id)
        label = pm.create_human_label(video.id, "winter footage")
        assert label.video_id == video.id
        assert label.label_text == "winter footage"
        assert label.id is not None

    def test_create_human_label_invalid_video_id(self, pm):
        with pytest.raises(ValueError, match="video_id must be positive"):
            pm.create_human_label(0, "winter footage")

    def test_create_human_label_invalid_label_text(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        video = pm.create_video("/path/to/video.mp4", "video.mp4", resolution.id)
        with pytest.raises(ValueError, match="label_text cannot be empty"):
            pm.create_human_label(video.id, "")


class TestCascadeDelete:
    def test_delete_video_cascades_to_frame_versions(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        video = pm.create_video("/path/to/video.mp4", "video.mp4", resolution.id)
        step = pm.create_processing_step(resolution.id, "motion_detect", {})
        node = pm.create_processing_node(step.id, parent_node_id=None, run_id=1)

        frame = pm.create_frame_version(node.id, video.id, 1, True, 0.5)
        frame_id = frame.id

        with get_session(pm._engine) as session:
            session.delete(video)
            session.commit()

        with get_session(pm._engine) as session:
            deleted_frame = session.get(FrameVersion, frame_id)
            assert deleted_frame is None

    def test_delete_video_cascades_to_human_labels(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        video = pm.create_video("/path/to/video.mp4", "video.mp4", resolution.id)
        label = pm.create_human_label(video.id, "winter footage")
        label_id = label.id

        with get_session(pm._engine) as session:
            session.delete(video)
            session.commit()

        with get_session(pm._engine) as session:
            deleted_label = session.get(HumanLabel, label_id)
            assert deleted_label is None

    def test_delete_frame_set_cascades_to_ai_labels(self, pm):
        frame_set = pm.create_frame_set("event_001")
        label = pm.create_ai_label(frame_set.id, "deer, night")
        label_id = label.id

        with get_session(pm._engine) as session:
            session.delete(frame_set)
            session.commit()

        with get_session(pm._engine) as session:
            deleted_label = session.get(AILabel, label_id)
            assert deleted_label is None


class TestUniqueConstraints:
    def test_frame_version_unique_constraint(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        video = pm.create_video("/path/to/video.mp4", "video.mp4", resolution.id)
        step = pm.create_processing_step(resolution.id, "motion_detect", {})
        node = pm.create_processing_node(step.id, parent_node_id=None, run_id=1)

        pm.create_frame_version(node.id, video.id, 1, True, 0.5)

        with pytest.raises(IntegrityError):
            pm.create_frame_version(node.id, video.id, 1, True, 0.6)

    def test_human_label_unique_constraint(self, pm):
        resolution = pm.get_or_create_resolution(1920, 1080)
        video = pm.create_video("/path/to/video.mp4", "video.mp4", resolution.id)

        pm.create_human_label(video.id, "winter footage")

        with pytest.raises(IntegrityError):
            pm.create_human_label(video.id, "winter footage")

    def test_ai_label_unique_constraint(self, pm):
        frame_set = pm.create_frame_set("event_001")

        pm.create_ai_label(frame_set.id, "deer, night")

        with pytest.raises(IntegrityError):
            pm.create_ai_label(frame_set.id, "deer, night")
