import logging

import typer

from wildcamtools.cli.motion_mog2 import app as motion_mog2_app
from wildcamtools.cli.perftest import app as perftest_app
from wildcamtools.cli.rescale import app as rescale_app
from wildcamtools.cli.rtsp import app as rtsp_app

app = typer.Typer()
app.add_typer(rescale_app)
app.add_typer(motion_mog2_app)
app.add_typer(perftest_app)
app.add_typer(rtsp_app)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
