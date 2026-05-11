import json
import logging
from collections import defaultdict
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)


def load_results(results_file: Path) -> list[dict]:
    """Load results from JSONL file."""
    results = []
    if results_file.exists():
        try:
            with open(results_file, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    results.append(data)
        except OSError:
            logger.exception("Failed to load results file")
        except json.JSONDecodeError:
            logger.exception("Failed to decode results file")
    return results


def calculate_accuracy_by_label(results: list[dict]) -> dict[str, tuple[int, int]]:
    """Calculate accuracy statistics grouped by label.

    Returns dict mapping label -> (correct_count, total_count)
    """
    stats: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for result in results:
        if len(result) == 2:
            _, result_dict = result
        else:
            continue
        label = result_dict.get("label", "unknown")
        is_correct = result_dict.get("result", False)
        correct, total = stats[label]
        stats[label] = (correct + (1 if is_correct else 0), total + 1)
    return dict(stats)


def main() -> None:
    st.set_page_config(layout="wide")
    st.title("WildCam Results Viewer")

    results_file = st.sidebar.text_input("Results JSONL File", value="result.jsonl")
    results_path = Path(results_file)

    if not results_path.exists():
        st.warning("Results file not found. Please provide a valid path.")
        logger.warning("Results file not found: %s", results_file)
        return

    results = load_results(results_path)

    if not results:
        st.info("No results found in the file.")
        logger.info("No results found in file: %s", results_file)
        return

    st.subheader(f"Loaded {len(results)} results from {results_file}")

    accuracy_stats = calculate_accuracy_by_label(results)

    total_correct = sum(s[0] for s in accuracy_stats.values())
    total_count = sum(s[1] for s in accuracy_stats.values())
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0
    st.metric("Overall Accuracy", value=f"{overall_accuracy:.2%}", delta=f"{total_correct}/{total_count}")

    if not accuracy_stats:
        st.error("Could not calculate accuracy statistics.")
        return

    labels = list(accuracy_stats.keys())
    accuracies = []
    totals = []

    for label in labels:
        correct, total = accuracy_stats[label]
        accuracy = correct / total if total > 0 else 0.0
        accuracies.append(accuracy)
        totals.append(total)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accuracy by Label")
        chart_data = {
            "Label": labels,
            "Accuracy": accuracies,
            "Total": totals,
        }
        st.bar_chart(chart_data, x="Label", y="Accuracy")

    with col2:
        st.subheader("Summary Statistics")
        for label in labels:
            correct, total = accuracy_stats[label]
            accuracy = correct / total if total > 0 else 0.0
            st.metric(
                label=label,
                value=f"{accuracy:.2%}",
                delta=f"{correct}/{total}",
            )

    st.subheader("Detailed Results")
    st.dataframe({
        "Label": labels,
        "Correct": [accuracy_stats[label][0] for label in labels],
        "Total": [accuracy_stats[label][1] for label in labels],
        "Accuracy": [f"{a:.2%}" for a in accuracies],
    })


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
