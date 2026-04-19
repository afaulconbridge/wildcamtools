"""Tests verifying that AGENTS.md content accurately reflects the repository state.

These tests act as documentation conformance checks: they ensure the developer
guide claims about paths, binaries, conventions, and dependencies match reality.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
AGENTS_MD = REPO_ROOT / "AGENTS.md"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"


# ---------------------------------------------------------------------------
# File existence & structure
# ---------------------------------------------------------------------------


def test_agents_md_exists() -> None:
    """AGENTS.md must be present at the repository root."""
    assert AGENTS_MD.exists(), "AGENTS.md not found at repository root"
    assert AGENTS_MD.is_file(), "AGENTS.md is not a regular file"


def test_agents_md_is_non_empty() -> None:
    """AGENTS.md must contain meaningful content."""
    content = AGENTS_MD.read_text(encoding="utf-8")
    assert len(content.strip()) > 0, "AGENTS.md is empty"


def test_agents_md_starts_with_title() -> None:
    """AGENTS.md must start with a level-1 heading."""
    content = AGENTS_MD.read_text(encoding="utf-8")
    first_line = content.splitlines()[0]
    assert first_line.startswith("# "), f"First line is not a level-1 heading: {first_line!r}"


# ---------------------------------------------------------------------------
# Required sections
# ---------------------------------------------------------------------------


REQUIRED_SECTIONS = [
    "## High-Signal Commands",
    "## Architecture & Flow",
    "## Conventions & Quirks",
    "## Best practices",
]


@pytest.mark.parametrize("section", REQUIRED_SECTIONS)
def test_agents_md_has_required_section(section: str) -> None:
    """Each required section heading must be present in AGENTS.md."""
    content = AGENTS_MD.read_text(encoding="utf-8")
    assert section in content, f"Required section missing from AGENTS.md: {section!r}"


# ---------------------------------------------------------------------------
# Documented paths exist in the repository
# ---------------------------------------------------------------------------


DOCUMENTED_DIRECTORIES = [
    "src/wildcamtools/lib",
    "src/wildcamtools/cli",
    "deployment",
    "tests",
    "tests/bin",
]


@pytest.mark.parametrize("rel_path", DOCUMENTED_DIRECTORIES)
def test_documented_directory_exists(rel_path: str) -> None:
    """Every directory mentioned in the Architecture section must exist."""
    path = REPO_ROOT / rel_path
    assert path.exists(), f"Documented directory does not exist: {rel_path}"
    assert path.is_dir(), f"Documented path is not a directory: {rel_path}"


def test_mediamtx_binary_exists() -> None:
    """tests/bin must contain the mediamtx binary referenced in AGENTS.md."""
    mediamtx = REPO_ROOT / "tests" / "bin" / "mediamtx"
    assert mediamtx.exists(), "tests/bin/mediamtx binary not found"
    assert mediamtx.is_file(), "tests/bin/mediamtx is not a regular file"


def test_uv_lock_exists() -> None:
    """uv.lock must exist – AGENTS.md states dependencies are checked via it."""
    uv_lock = REPO_ROOT / "uv.lock"
    assert uv_lock.exists(), "uv.lock not found at repository root"
    assert uv_lock.is_file(), "uv.lock is not a regular file"


# ---------------------------------------------------------------------------
# Documented conventions match pyproject.toml
# ---------------------------------------------------------------------------


def test_python_version_constraint_matches_docs() -> None:
    """pyproject.toml must require Python ==3.13.*, as stated in AGENTS.md."""
    content = PYPROJECT_TOML.read_text(encoding="utf-8")
    assert 'requires-python = "==3.13.*"' in content, (
        "pyproject.toml does not require Python ==3.13.* (AGENTS.md Conventions claim)"
    )


def test_opencv_headless_dependency_matches_docs() -> None:
    """pyproject.toml must depend on opencv-contrib-python-headless, as stated in AGENTS.md."""
    content = PYPROJECT_TOML.read_text(encoding="utf-8")
    assert "opencv-contrib-python-headless" in content, (
        "opencv-contrib-python-headless not found in pyproject.toml (AGENTS.md Conventions claim)"
    )


def test_opencv_gui_version_is_excluded() -> None:
    """uv override must prevent opencv-python (GUI) from being installed, as AGENTS.md describes."""
    content = PYPROJECT_TOML.read_text(encoding="utf-8")
    assert "opencv-python" in content, "Expected opencv-python override entry in pyproject.toml"
    # The override sets an impossible python_version constraint so it is never installed.
    assert "python_version < '0'" in content, (
        "Expected impossible python_version constraint for opencv-python override"
    )


def test_wildcamtools_cli_script_defined() -> None:
    """pyproject.toml must define the wildcamtools CLI entrypoint, as stated in AGENTS.md."""
    content = PYPROJECT_TOML.read_text(encoding="utf-8")
    assert "wildcamtools" in content, "wildcamtools script not found in pyproject.toml"
    assert "wildcamtools.cli" in content, (
        "wildcamtools CLI module reference not found in pyproject.toml"
    )


# ---------------------------------------------------------------------------
# High-Signal Commands section content
# ---------------------------------------------------------------------------


DOCUMENTED_COMMANDS = [
    "uv run pytest",
    "uv run ruff check",
    "uv run ruff format",
    "uv run mypy src",
    "uv run wildcamtools",
]


@pytest.mark.parametrize("command", DOCUMENTED_COMMANDS)
def test_high_signal_command_present(command: str) -> None:
    """Every documented high-signal command must appear in AGENTS.md."""
    content = AGENTS_MD.read_text(encoding="utf-8")
    assert command in content, f"Documented command missing from AGENTS.md: {command!r}"


# ---------------------------------------------------------------------------
# Best Practices section content
# ---------------------------------------------------------------------------


def test_best_practices_format_after_edit() -> None:
    """AGENTS.md must document the 'format after edit' best practice."""
    content = AGENTS_MD.read_text(encoding="utf-8")
    assert "ruff format" in content, "Best practice 'ruff format' not mentioned in AGENTS.md"


def test_best_practices_before_commit() -> None:
    """AGENTS.md must document the 'before commit' best practice using prek."""
    content = AGENTS_MD.read_text(encoding="utf-8")
    assert "prek" in content, "Best practice 'prek' not mentioned in AGENTS.md"
    assert "before committing" in content or "Before commit" in content, (
        "AGENTS.md does not mention pre-commit guidance"
    )


# ---------------------------------------------------------------------------
# Negative / boundary tests
# ---------------------------------------------------------------------------


def test_agents_md_has_multiple_lines() -> None:
    """AGENTS.md must have more than one line (boundary: not a one-liner stub)."""
    lines = AGENTS_MD.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 5, f"AGENTS.md has too few lines ({len(lines)}); expected a full guide"


def test_agents_md_does_not_reference_nonexistent_src_module() -> None:
    """src/wildcamtools/__main__.py must exist (supports 'python -m wildcamtools' entrypoint)."""
    main_module = REPO_ROOT / "src" / "wildcamtools" / "__main__.py"
    assert main_module.exists(), (
        "src/wildcamtools/__main__.py missing; 'python -m wildcamtools' entrypoint would fail"
    )


def test_python_m_wildcamtools_entrypoint_documented() -> None:
    """AGENTS.md must document 'python -m wildcamtools' as an alternative entrypoint."""
    content = AGENTS_MD.read_text(encoding="utf-8")
    assert "python -m wildcamtools" in content, (
        "AGENTS.md does not document 'python -m wildcamtools' alternative entrypoint"
    )