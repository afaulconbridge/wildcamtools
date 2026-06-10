import json
import logging
from pathlib import Path
from typing import Any

import streamlit as st
from sqlalchemy import desc, select
from sqlalchemy.engine import Engine
from sqlmodel import Session

from wildcamtools.lib.persistence.database import create_engine_and_tables
from wildcamtools.lib.persistence.models import ClassificationResult, PipelineBatch, PipelineRun, Video

logger = logging.getLogger(__name__)

# mypy: disable-error-code="arg-type,call-overload,no-any-return"
# SQLModel's Session.exec() has type issues with select() that are well-known


def get_engine_and_session(db_path: Path) -> tuple[Engine, Session]:
    """Create database engine and session."""
    engine = create_engine_and_tables(f"sqlite:///{db_path.absolute()}")
    session = Session(engine)
    return engine, session


def _get_confidence_value(confidence: Any) -> str:
    """Safely extract confidence value as string."""
    if hasattr(confidence, "value"):
        return str(confidence.value)
    return str(confidence)


def get_all_videos(session: Session) -> list[str]:
    """Get all unique video filenames from the database."""
    stmt = select(Video.filename).order_by(Video.filename)
    results = session.exec(stmt)
    return list(results.scalars().all())


def get_runs_for_video(session: Session, filename: str) -> list[PipelineRun]:
    """Get all pipeline runs for a specific video, ordered by timestamp."""
    stmt = select(PipelineRun).where(PipelineRun.video_filename == filename).order_by(desc(PipelineRun.timestamp))
    results = session.exec(stmt)
    return list(results.scalars().all())


def get_classification_result(session: Session, result_id: int | None) -> ClassificationResult | None:
    """Get a classification result by ID."""
    if result_id is None:
        return None
    stmt = select(ClassificationResult).where(ClassificationResult.id == result_id)
    result = session.exec(stmt).scalars().first()
    return result


def display_run_info(run: PipelineRun, classification: ClassificationResult | None) -> None:
    """Display information about a pipeline run."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Run ID", run.id)
        timestamp_str = run.timestamp.strftime("%Y-%m-%d %H:%M:%S") if run.timestamp else "N/A"
        st.metric("Timestamp", timestamp_str)

    with col2:
        if classification:
            st.metric("Species", classification.species_name)
            st.metric("Confidence", _get_confidence_value(classification.confidence))

    with col3:
        if classification:
            animal_status = "Present" if classification.is_animal_present else "Absent"
            unknown_status = "Unknown" if classification.is_animal_unknown else "Identified"
            st.metric("Animal", animal_status)
            st.metric("Status", unknown_status)


def _display_config_summary(config: dict[str, Any]) -> None:
    """Display pipeline configuration summary."""
    col1, col2 = st.columns(2)
    with col1:
        if "frame_selector" in config:
            fs = config["frame_selector"]
            st.json({"frame_selector": fs})
        if "frame_extractor" in config:
            fe = config["frame_extractor"]
            st.json({
                "frame_extractor": {"extractor_type": fe.get("extractor_type"), "resolution": fe.get("resolution")}
            })
    with col2:
        if "llm" in config:
            llm = config["llm"]
            st.json({"llm": {"backend": llm.get("backend"), "model": llm.get("model")}})
        if "query" in config:
            query = config["query"]
            st.json({"query": {"query_type": query.get("query_type")}})


def _display_batch_summary(session: Session, batches: list[PipelineBatch]) -> None:
    """Display batch processing summary."""
    batch_data = []
    for batch in batches:
        try:
            frame_numbers = json.loads(batch.frame_numbers)
        except Exception:
            frame_numbers = []
        batch_result = get_classification_result(session, batch.result_id)
        batch_data.append({
            "Batch ID": batch.id,
            "Frames": len(frame_numbers),
            "Species": batch_result.species_name if batch_result else "N/A",
            "Confidence": _get_confidence_value(batch_result.confidence) if batch_result else "N/A",
        })
    st.dataframe(batch_data, use_container_width=True)


def _init_database(db_path: Path) -> Session | None:
    """Initialize database session with caching via session_state."""
    if "db_session" not in st.session_state:
        st.session_state.db_session = None
    if "last_db_path" not in st.session_state:
        st.session_state.last_db_path = None

    if st.session_state.last_db_path != str(db_path.absolute()):
        st.session_state.last_db_path = str(db_path.absolute())
        st.session_state.db_session = None

    if st.session_state.db_session is None:
        try:
            engine, session = get_engine_and_session(db_path)
            st.session_state.db_engine = engine
            st.session_state.db_session = session
        except Exception as e:
            st.sidebar.error(f"Failed to connect to database: {e}")
            logger.exception("Failed to connect to database")
            return None
    return st.session_state.db_session


def _display_video_and_results(session: Session, selected_video: str, video_dir: Path) -> None:
    """Display video player and pipeline results."""
    video_path = video_dir / selected_video
    if video_path.exists():
        st.video(str(video_path))
    else:
        st.warning(f"Video file not found: {video_path.absolute()}")
        st.info(f"Expected video at: {video_path}")
        st.caption("You can set the video directory in the sidebar configuration.")

    try:
        runs = get_runs_for_video(session, selected_video)
    except Exception as e:
        st.error(f"Failed to load runs: {e}")
        logger.exception("Failed to load runs for video")
        return

    st.header(f"Pipeline Results ({len(runs)} runs)")

    for run in runs:
        timestamp_str = run.timestamp.strftime("%Y-%m-%d %H:%M:%S") if run.timestamp else "N/A"
        with st.expander(f"Run ID: {run.id} ({timestamp_str})"):
            classification = get_classification_result(session, run.final_result_id)
            display_run_info(run, classification)

            st.subheader("Configuration")
            try:
                config = json.loads(run.config_json)
                _display_config_summary(config)
            except Exception:
                display_text = run.config_json[:500] + "..." if len(run.config_json) > 500 else run.config_json
                st.code(display_text)

            if run.batches:
                st.subheader(f"Batches ({len(run.batches)})")
                _display_batch_summary(session, run.batches)


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(page_title="WildCam Video Browser", layout="wide")
    st.title("WildCam Video Browser")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    db_path_str = st.sidebar.text_input(
        "Database Path",
        value="wildcamtools.db",
        help="Path to the SQLite database file",
    )
    video_dir_str = st.sidebar.text_input(
        "Video Directory",
        value=".",
        help="Base directory where video files are located",
    )

    db_path = Path(db_path_str)
    video_dir = Path(video_dir_str)

    if not db_path.exists():
        st.sidebar.error(f"Database file not found: {db_path.absolute()}")
        st.warning("Please provide a valid database path.")
        return

    session = _init_database(db_path)
    if session is None:
        return

    try:
        videos = get_all_videos(session)
    except Exception as e:
        st.sidebar.error(f"Failed to load videos: {e}")
        logger.exception("Failed to load videos")
        return

    if not videos:
        st.sidebar.info("No videos found in the database.")
        st.info("Import pipeline results using: `wildcamtools db import-result <result.json> <video.mp4>`")
        return

    st.sidebar.header("Videos")
    selected_video = st.sidebar.selectbox(
        "Select a video",
        videos,
        placeholder="Choose a video...",
    )

    if not selected_video:
        st.info("Select a video from the sidebar to view results.")
        return

    _display_video_and_results(session, selected_video, video_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    logging.getLogger("wildcamtools").setLevel(logging.DEBUG)
    main()
