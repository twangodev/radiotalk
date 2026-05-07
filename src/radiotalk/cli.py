from __future__ import annotations

import typer

from . import __version__
from .data.cli import data_app
from .voices.cli import voices_app

app = typer.Typer(add_completion=False, help="radiotalk — ATC language tooling.")
app.add_typer(data_app, name="data")
app.add_typer(voices_app, name="voices")


def _add_audio_typer() -> None:
    """Audio subcommand requires the `tts` extra (heavy: torch, transformers,
    hume-tada). Import lazily so `radiotalk data` / `radiotalk voices` work
    without the extra installed.
    """
    try:
        from .audio.cli import audio_app
    except ImportError:
        return
    app.add_typer(audio_app, name="audio")


_add_audio_typer()


@app.command()
def version() -> None:
    """Print the installed radiotalk version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
