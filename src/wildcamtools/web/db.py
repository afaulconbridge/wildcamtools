import json
import logging
from pathlib import Path
from typing import Any

import plotly.express as px
import streamlit as st
from sqlalchemy.engine import Engine
from sqlmodel import Session

from wildcamtools.lib.persistence.database import create_engine_and_tables
from wildcamtools.lib.persistence.models import ClassificationResult, PipelineBatch, PipelineRun
from wildcamtools.lib.persistence.repository import (
    StatisticsSummary,
    aggregate_statistics,
    count_pipeline_runs_filtered,
    get_classification_result,
    list_all_video_filenames,
    list_pipeline_runs_filtered,
    list_recent_pipeline_runs,
    list_runs_for_video,
    list_species_with_counts,
)
from wildcamtools.web.lib.thumbnails import cached_thumbnail

logger = logging.getLogger(__name__)

# mypy: disable-error-code="arg-type,call-overload,no-any-return"
# SQLModel's Session.exec() has type issues with select() that are well-known

CONFIDENCE_LEVELS = ["high", "medium", "low"]
PAGE_SIZE = 25
SESSION_KEYS = (
    "db_session",
    "db_engine",
    "last_db_path",
    "selected_video",
    "browse_page",
    "browse_filters",
    "stats_filters",
)


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


def display_run_info(run: PipelineRun, classification: ClassificationResult | None) -> None:
    """Display information about a pipeline run."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Run ID", run.id)
        timestamp_str = run.timestamp.strftime("%Y-%m-%d %H:%M:%S")
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
                "frame_extractor": {"extractor_type": fe.get("extractor_type"), "resolution": fe.get("resolution")},
            })
    with col2:
        if "llm" in config:
            llm = config["llm"]
            st.json({"llm": {"backend": llm.get("backend"), "model": llm.get("model")}})
        if "query" in config:
            query = config["query"]
            st.json({"query": {"query_type": query.get("query_type")}})


def _display_batch_summary(batches: list[PipelineBatch]) -> None:
    """Display batch processing summary."""
    batch_data = []
    for batch in batches:
        try:
            frame_numbers = json.loads(batch.frame_numbers)
        except Exception:
            frame_numbers = []
        batch_result = batch.result
        batch_data.append({
            "Batch ID": batch.id,
            "Frames": len(frame_numbers),
            "Species": batch_result.species_name if batch_result else "N/A",
            "Confidence": _get_confidence_value(batch_result.confidence) if batch_result else "N/A",
        })
    st.dataframe(batch_data, width="stretch")


def _init_database(db_path: Path) -> Session | None:
    """Initialize database session with caching via session_state."""
    for key in SESSION_KEYS:
        st.session_state.setdefault(key, None)

    if st.session_state.last_db_path != str(db_path.absolute()):
        if st.session_state.db_session is not None:
            st.session_state.db_session.close()
        if st.session_state.db_engine is not None:
            st.session_state.db_engine.dispose()
        st.session_state.last_db_path = str(db_path.absolute())
        st.session_state.db_session = None
        st.session_state.db_engine = None
        st.session_state.browse_filters = None
        st.session_state.stats_filters = None
        st.session_state.selected_video = None
        st.session_state.browse_page = 0

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
        runs = list_runs_for_video(session, selected_video)
    except Exception as e:
        st.error(f"Failed to load runs: {e}")
        logger.exception("Failed to load runs for video")
        return

    st.header(f"Pipeline Results ({len(runs)} runs)")

    for run in runs:
        timestamp_str = run.timestamp.strftime("%Y-%m-%d %H:%M:%S")
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
                _display_batch_summary(run.batches)


def _default_filters() -> dict[str, Any]:
    return {
        "confidences": list(CONFIDENCE_LEVELS),
        "species": [],
        "animal_present_only": False,
    }


def _ensure_filters(key: str) -> dict[str, Any]:
    if not st.session_state.get(key):
        st.session_state[key] = _default_filters()
    return st.session_state[key]


def _resolve_video_path(video_dir: Path, filename: str) -> Path:
    """Best-effort resolution of a video file given its stored filename.

    The CLI imports results with absolute paths so the stored filename is
    already absolute. Fall back to joining with ``video_dir`` for relative
    filenames.
    """
    candidate = Path(filename)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    joined = video_dir / filename
    if joined.exists():
        return joined
    return candidate


def _render_stats_filters(
    filters: dict[str, Any],
    species_options: list[tuple[str, int]],
) -> dict[str, Any]:
    with st.form(key="stats_filters_form", border=False):
        confidences = st.multiselect(
            "Confidence",
            options=CONFIDENCE_LEVELS,
            default=filters.get("confidences") or list(CONFIDENCE_LEVELS),
            key="stats_confidences",
        )
        species_labels = [name for name, _ in species_options]
        selected_species = st.multiselect(
            "Species",
            options=species_labels,
            default=[s for s in filters.get("species", []) if s in species_labels],
            key="stats_species",
            help="Empty = all species",
        )
        if species_options:
            with st.expander("Species counts", expanded=False):
                for name, count in species_options:
                    st.caption(f"{name}: {count}")
        animal_present_only = st.checkbox(
            "Animal present only",
            value=bool(filters.get("animal_present_only", False)),
            key="stats_animal_present",
        )
        st.form_submit_button("Apply filters", width="stretch")
    return {
        "confidences": confidences,
        "species": selected_species,
        "animal_present_only": animal_present_only,
    }


def _format_run_caption(run: PipelineRun) -> str:
    parts = []
    result = run.final_result
    if result is not None:
        parts.append(result.species_name)
        parts.append(_get_confidence_value(result.confidence))
        if result.is_animal_present:
            parts.append("animal present")
        if result.is_animal_unknown:
            parts.append("unknown")
    else:
        parts.append("no result")
    video = run.video
    if video is not None and video.recorded_at is not None:
        parts.append(f"recorded {video.recorded_at.strftime('%Y-%m-%d %H:%M:%S')}")
    timestamp_str = run.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    parts.append(timestamp_str)
    return " · ".join(parts)


def _render_browse_filters(
    filters: dict[str, Any],
    species_options: list[tuple[str, int]],
) -> tuple[dict[str, Any], bool]:
    """Render the filter panel and return (new_filters, changed)."""
    with st.form(key="browse_filters_form", border=False):
        confidences = st.multiselect(
            "Confidence",
            options=CONFIDENCE_LEVELS,
            default=filters.get("confidences") or list(CONFIDENCE_LEVELS),
            key="browse_confidences",
        )
        species_labels = [name for name, _ in species_options]
        selected_species = st.multiselect(
            "Species",
            options=species_labels,
            default=[s for s in filters.get("species", []) if s in species_labels],
            key="browse_species",
            help="Empty = all species",
        )
        if species_options:
            with st.expander("Species counts", expanded=False):
                for name, count in species_options:
                    st.caption(f"{name}: {count}")
        animal_present_only = st.checkbox(
            "Animal present only",
            value=bool(filters.get("animal_present_only", False)),
            key="browse_animal_present",
        )
        st.form_submit_button("Apply filters", width="stretch")

    new_filters = {
        "confidences": confidences,
        "species": selected_species,
        "animal_present_only": animal_present_only,
    }
    return new_filters, new_filters != filters


def _render_browse_pagination(total: int) -> int:
    max_page = max(0, (total - 1) // PAGE_SIZE) if total > 0 else 0
    st.session_state.browse_page = min(st.session_state.browse_page, max_page)
    page = st.session_state.browse_page

    nav_l, nav_m, nav_r = st.columns([1, 2, 1])
    with nav_l:
        if st.button("◀ Prev", disabled=page == 0, width="stretch", key="browse_prev"):
            st.session_state.browse_page = max(0, page - 1)
            st.rerun()
    with nav_m:
        st.write(f"Page {page + 1} of {max_page + 1}")
    with nav_r:
        if st.button("Next ▶", disabled=page >= max_page, width="stretch", key="browse_next"):
            st.session_state.browse_page = min(max_page, page + 1)
            st.rerun()
    return page


def _render_browse_tab(session: Session, video_dir: Path) -> None:
    """Render the filterable list of pipeline runs."""
    filters = _ensure_filters("browse_filters")

    try:
        species_options = list_species_with_counts(session, confidences=filters.get("confidences") or CONFIDENCE_LEVELS)
    except Exception as e:
        logger.exception("Failed to load species options")
        st.error(f"Failed to load species: {e}")
        species_options = []

    col_filter, col_results = st.columns([1, 3])

    with col_filter:
        new_filters, changed = _render_browse_filters(filters, species_options)
        st.session_state.browse_filters = new_filters
        if changed:
            st.session_state.browse_page = 0
        if st.button("Reset filters", width="stretch", key="browse_reset"):
            st.session_state.browse_filters = _default_filters()
            st.session_state.browse_page = 0
            st.session_state.browse_confidences = list(CONFIDENCE_LEVELS)
            st.session_state.browse_species = []
            st.session_state.browse_animal_present = False
            st.rerun()

    with col_results:
        try:
            total = count_pipeline_runs_filtered(
                session,
                confidences=new_filters["confidences"] or CONFIDENCE_LEVELS,
                species=new_filters["species"] or None,
                animal_present_only=new_filters["animal_present_only"],
            )
        except Exception as e:
            logger.exception("Failed to count runs")
            st.error(f"Failed to count runs: {e}")
            return

        st.caption(f"{total} matching run{'s' if total != 1 else ''}")

        if total == 0:
            st.info("No runs match these filters.")
            return

        page = _render_browse_pagination(total)

        try:
            runs = list_pipeline_runs_filtered(
                session,
                confidences=new_filters["confidences"] or CONFIDENCE_LEVELS,
                species=new_filters["species"] or None,
                animal_present_only=new_filters["animal_present_only"],
                limit=PAGE_SIZE,
                offset=page * PAGE_SIZE,
            )
        except Exception as e:
            logger.exception("Failed to load runs")
            st.error(f"Failed to load runs: {e}")
            return

        for run in runs:
            _render_run_row(run, video_dir, key_suffix=f"browse_{run.id}")


def _render_run_row(
    run: PipelineRun,
    video_dir: Path,
    *,
    key_suffix: str,
) -> None:
    """Render a single result row with thumbnail and brief info."""
    video_path = _resolve_video_path(video_dir, run.video_filename)
    thumb_col, info_col = st.columns([1, 4])
    with thumb_col:
        st.image(cached_thumbnail(str(video_path), 160), width=160)
    with info_col:
        st.markdown(f"**{run.video_filename}**")
        st.caption(_format_run_caption(run))
        result = run.final_result
        if result is not None and result.species_name:
            st.caption(f"Run ID: {run.id}")
        if st.button("View details", key=f"open_{key_suffix}"):
            st.session_state.selected_video = run.video_filename
            st.rerun()
    st.divider()


def _render_details_tab(session: Session, video_dir: Path) -> None:
    """Render the per-video details view."""
    selected = st.session_state.get("selected_video")

    if not selected:
        st.info("Select a video from the Browse tab to view its details.")
        try:
            recent = list_recent_pipeline_runs(session, limit=10)
        except Exception as e:
            logger.exception("Failed to load recent videos")
            st.error(f"Failed to load recent videos: {e}")
            return

        if not recent:
            st.caption("No pipeline runs available.")
            return

        st.subheader("Recent videos")
        for run in recent:
            _render_run_row(
                run,
                video_dir,
                key_suffix=f"recent_{run.id}",
            )
        return

    if st.button("◀ Back to Browse", key="details_back"):
        st.session_state.selected_video = None
        st.rerun()

    st.header(selected)
    _display_video_and_results(session, selected, video_dir)


def _render_statistics_tab(session: Session) -> None:
    """Render the aggregate statistics view."""
    filters = _ensure_filters("stats_filters")

    try:
        species_options = list_species_with_counts(session, confidences=filters.get("confidences") or CONFIDENCE_LEVELS)
    except Exception as e:
        logger.exception("Failed to load species options")
        st.error(f"Failed to load species: {e}")
        species_options = []

    col_filter, col_charts = st.columns([1, 3])

    with col_filter:
        new_filters = _render_stats_filters(filters, species_options)
        st.session_state.stats_filters = new_filters
        if st.button("Reset filters", width="stretch", key="stats_reset"):
            st.session_state.stats_filters = _default_filters()
            st.session_state.stats_confidences = list(CONFIDENCE_LEVELS)
            st.session_state.stats_species = []
            st.session_state.stats_animal_present = False
            st.rerun()

    with col_charts:
        try:
            summary: StatisticsSummary = aggregate_statistics(
                session,
                confidences=new_filters["confidences"] or CONFIDENCE_LEVELS,
                species=new_filters["species"] or None,
                animal_present_only=new_filters["animal_present_only"],
            )
        except Exception as e:
            logger.exception("Failed to aggregate statistics")
            st.error(f"Failed to aggregate statistics: {e}")
            return

        if summary.total_runs == 0:
            st.info("No data matches these filters.")
            return

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Runs", summary.total_runs)
        m2.metric("Videos", summary.total_videos)
        m3.metric("Species", summary.distinct_species)
        present_pct = summary.animal_present_count / summary.total_runs if summary.total_runs else 0.0
        unknown_pct = summary.animal_unknown_count / summary.total_runs if summary.total_runs else 0.0
        m4.metric("Animal present", f"{present_pct:.0%}", delta=f"{summary.animal_present_count}")
        m5.metric("Unknown", f"{unknown_pct:.0%}", delta=f"{summary.animal_unknown_count}")

        st.subheader("Species frequency")
        if summary.species_counts:
            species_df = {
                "Species": list(summary.species_counts.keys()),
                "Runs": list(summary.species_counts.values()),
            }
            fig = px.bar(species_df, x="Species", y="Runs", title="Pipeline runs by species")
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("No species data.")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Confidence distribution")
            if summary.confidence_counts:
                ordered = {lvl: summary.confidence_counts.get(lvl, 0) for lvl in CONFIDENCE_LEVELS}
                ordered = {k: v for k, v in ordered.items() if v > 0}
                conf_df = {"Confidence": list(ordered.keys()), "Runs": list(ordered.values())}
                fig = px.bar(conf_df, x="Confidence", y="Runs", title="Runs by confidence level")
                st.plotly_chart(fig, width="stretch")
            else:
                st.caption("No confidence data.")

        with c2:
            st.subheader("Animal status")
            status_df = {
                "Status": ["Present", "Absent", "Unknown"],
                "Runs": [summary.animal_present_count, summary.animal_absent_count, summary.animal_unknown_count],
            }
            fig = px.bar(status_df, x="Status", y="Runs", title="Runs by animal status")
            st.plotly_chart(fig, width="stretch")


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(page_title="WildCam Video Browser", layout="wide")
    st.title("WildCam Video Browser")

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
        videos = list_all_video_filenames(session)
    except Exception as e:
        st.sidebar.error(f"Failed to load videos: {e}")
        logger.exception("Failed to load videos")
        return

    if not videos:
        st.sidebar.info("No videos found in the database.")
        st.info("Import pipeline results using: `wildcamtools db import-result <result.json> <video.mp4>`")
        return

    tab1, tab2, tab3 = st.tabs(["Browse", "Video Details", "Statistics"])
    with tab1:
        _render_browse_tab(session, video_dir)
    with tab2:
        _render_details_tab(session, video_dir)
    with tab3:
        _render_statistics_tab(session)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    logging.getLogger("wildcamtools").setLevel(logging.DEBUG)
    main()
