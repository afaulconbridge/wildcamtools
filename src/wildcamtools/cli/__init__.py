import logging

# have to setup logging _before_ imports
# generally, imports have a high level logger object
# those loggers need to have config created _before_ they exist
logging.basicConfig(level=logging.INFO, force=False)

import typer

from wildcamtools.cli.motion_mog2 import app as motion_mog2_app
from wildcamtools.cli.perftest import app as perftest_app
from wildcamtools.cli.rescale import app as rescale_app
from wildcamtools.cli.rtsp import app as rtsp_app
from wildcamtools.cli.segment import app as segment_app
from wildcamtools.cli.states import app as states_app
from wildcamtools.cli.watch import app as watch_app

app = typer.Typer()
app.add_typer(rescale_app)
app.add_typer(motion_mog2_app)
app.add_typer(perftest_app)
app.add_typer(rtsp_app)
app.add_typer(states_app)
app.add_typer(segment_app)
app.add_typer(watch_app)


if __name__ == "__main__":
    pass
