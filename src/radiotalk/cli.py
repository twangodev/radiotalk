from __future__ import annotations

import typer

from . import __version__
from .data.cli import data_app
from .voices.cli import voices_app

app = typer.Typer(add_completion=False, help="radiotalk — ATC language tooling.")
app.add_typer(data_app, name="data")
app.add_typer(voices_app, name="voices")


@app.command()
def version() -> None:
    """Print the installed radiotalk version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
