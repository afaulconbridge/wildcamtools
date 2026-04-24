import logging
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def run() -> None:
    """Run the Streamlit video labeling application."""
    # We need to point streamlit to the correct file path
    script_path = Path(__file__).parent.parent / "lib" / "label.py"
    python_exe = sys.executable

    # Streamlit must be run via 'streamlit run ...'
    # We can use subprocess to launch it.
    cmd = [python_exe, "-m", "streamlit", "run", str(script_path)]

    try:
        subprocess.run(cmd, check=True)  # noqa: S603
    except subprocess.CalledProcessError as e:
        logger.exception("Error in streamlit application")
        raise typer.Exit(code=1) from e
    except KeyboardInterrupt:
        logger.info("Stopping labeling app...")
        raise


if __name__ == "__main__":
    app()
