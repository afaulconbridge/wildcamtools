import logging
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def run() -> None:
    """Run the Streamlit results viewer application."""
    script_path = Path(__file__).parent.parent / "lib" / "web" / "results.py"
    python_exe = sys.executable

    cmd = [python_exe, "-m", "streamlit", "run", str(script_path)]

    try:
        subprocess.run(cmd, check=True)  # noqa: S603
    except subprocess.CalledProcessError as e:
        logger.exception("Error in streamlit application")
        raise typer.Exit(code=1) from e
    except KeyboardInterrupt:
        logger.info("Stopping results viewer...")
        raise


if __name__ == "__main__":
    app()
