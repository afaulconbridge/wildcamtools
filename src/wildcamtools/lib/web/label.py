import logging
import os
from pathlib import Path

import streamlit as st

from wildcamtools.lib.labels import (
    get_video_list,
    load_labels,
    save_label,
)

logger = logging.getLogger(__name__)


def handle_navigation(videos: list[str]) -> None:
    """Handle previous and skip buttons."""

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous") and st.session_state.vid_idx > 0:
            st.session_state.vid_idx -= 1
            st.session_state.pop("label_input", None)
            st.rerun()
    with col2:
        if st.button("Skip") and st.session_state.vid_idx < len(videos) - 1:
            st.session_state.vid_idx += 1
            st.session_state.pop("label_input", None)
            st.rerun()


def save_and_next(output_path: Path, current_vid_name: str, label: str, videos: list[str]) -> None:
    """Save label and advance to next video."""

    if label:
        save_label(output_path, current_vid_name, label)
        st.session_state.labels[current_vid_name] = label
        logger.info("Labeled %s as %s", current_vid_name, label)

        if st.session_state.vid_idx < len(videos) - 1:
            st.session_state.vid_idx += 1
            st.session_state.pop("label_input", None)
            st.rerun()
        else:
            st.success("All videos labeled!")
            logger.info("Finished labeling all videos")
    else:
        st.error("Please enter a label before saving.")
        logger.warning("Attempted to save label for %s without providing a label", current_vid_name)


def main() -> None:
    st.set_page_config(layout="wide")
    st.title("WildCam Video Labeler")

    vid_dir = st.sidebar.text_input("Video Directory", value=".")
    output_file = st.sidebar.text_input("Output JSONL File", value="labels.jsonl")
    output_path = Path(output_file)

    if not vid_dir or not os.path.isdir(vid_dir):
        st.warning("Please provide a valid video directory.")
        logger.warning("Invalid video directory provided: %s", vid_dir)
        return

    videos = get_video_list(vid_dir)

    if "labels" not in st.session_state:
        st.session_state.labels = load_labels(output_path)

    if not videos:
        st.info("No videos found in the specified directory.")
        logger.info("No videos found in directory: %s", vid_dir)
        return

    if "vid_idx" not in st.session_state:
        st.session_state.vid_idx = 0

    st.sidebar.checkbox("Skip labeled videos", value=False, key="skip_labeled")
    if st.session_state.skip_labeled:
        while st.session_state.vid_idx < len(videos) and videos[st.session_state.vid_idx] in st.session_state.labels:
            st.session_state.vid_idx += 1

        if st.session_state.vid_idx >= len(videos):
            st.success("All remaining videos have been labeled!")
            return

    current_vid_name = videos[st.session_state.vid_idx]
    vid_path = os.path.join(vid_dir, current_vid_name)

    st.subheader(f"Video: {current_vid_name}")

    st.video(vid_path)

    current_label = st.session_state.labels.get(current_vid_name, "")
    label = st.text_input("Label", value=current_label, key="label_input")

    if st.button("Save and Next"):
        save_and_next(output_path, current_vid_name, label, videos)

    handle_navigation(videos)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
