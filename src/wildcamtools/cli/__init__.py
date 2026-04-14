import logging

import typer

from wildcamtools.cli.frames import app as frames_app
from wildcamtools.cli.motion_mog2 import app as motion_mog2_app
from wildcamtools.cli.perftest import app as perftest_app
from wildcamtools.cli.rescale import app as rescale_app
from wildcamtools.cli.rtsp import app as rtsp_app
from wildcamtools.cli.segment import app as segment_app
from wildcamtools.cli.watch import app as watch_app

# generally, imports have a high level logger object
# those loggers need to be re-configured as they already exist
# TODO make log levels configurable
logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("wildcamtools").setLevel(logging.DEBUG)

app = typer.Typer()
app.add_typer(rescale_app)
app.add_typer(motion_mog2_app, name="motion")
app.add_typer(perftest_app)
app.add_typer(rtsp_app)
app.add_typer(segment_app)
app.add_typer(watch_app)
app.add_typer(frames_app)


if __name__ == "__main__":
    app()
