import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def load_labels(output: Path) -> dict[str, str]:
    """Load labels from JSONL file."""
    labels = {}
    if output.exists():
        try:
            with open(output, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    if "video" in data and "label" in data:
                        labels[data["video"]] = data["label"]
        except OSError:
            logger.exception("Failed to load existing labels")
        except json.JSONDecodeError:
            logger.exception("Failed to decode existing labels")
    return labels


def save_label(output: Path, video_name: str, label: str) -> None:
    """Save label to JSONL file, replacing any existing label for the video."""
    labels = load_labels(output)
    labels[video_name] = label

    with tempfile.NamedTemporaryFile(
        dir=str(output.resolve().parent), prefix=output.stem, delete=False, encoding="utf-8", mode="w"
    ) as f:
        f.write("")
        temp_name = f.name
        for vid, lbl in labels.items():
            f.write(json.dumps({"video": vid, "label": lbl}) + "\n")
    os.replace(temp_name, str(output))


def get_video_list(vid_dir: str) -> list[str]:
    """Get sorted list of video files in a directory."""
    valid_extensions = (".mp4", ".mov", ".avi", ".mkv")
    return sorted([f for f in os.listdir(vid_dir) if f.lower().endswith(valid_extensions)])
