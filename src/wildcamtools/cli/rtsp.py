from time import sleep

import typer

from wildcamtools.lib.rtsp import BackgroundFFMPEGBroadcast, BackgroundMediaMTX

app = typer.Typer()


@app.command()
def serve():
    with BackgroundMediaMTX(), BackgroundFFMPEGBroadcast():
        typer.secho("RTSP stream ready at rtsp://localhost:8554/stream")
        try:
            while True:
                sleep(1)
        except KeyboardInterrupt:
            pass
    typer.secho("Cleanup complete")
