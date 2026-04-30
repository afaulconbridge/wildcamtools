from pathlib import Path
from time import sleep

import typer

from wildcamtools.lib.rtsp import BackgroundMediaMTX, RTSPBroadcaster

app = typer.Typer()


@app.command()
def serve(path: Path) -> None:
    with BackgroundMediaMTX(), RTSPBroadcaster(path, "rtsp://localhost:8554/stream"):
        typer.secho("RTSP stream ready at rtsp://localhost:8554/stream")
        try:
            while True:
                sleep(1)
        except KeyboardInterrupt:
            pass
    typer.secho("Cleanup complete")
