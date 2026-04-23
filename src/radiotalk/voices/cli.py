from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .._progress import ProgressLogger
from .builder import BuildStats, build as run_build
from .filter import FilterParams
from .notice import render_notice
from .pool import VoicePoolWriter
from .source import get_source, get_sub_pool

voices_app = typer.Typer(
    add_completion=False,
    help="Reference voice pool aggregation for radiotalk TTS.",
)
console = Console()


@voices_app.callback()
def _voices_callback() -> None:
    """Voice pool aggregation commands."""


@voices_app.command()
def build(
    out: Annotated[Path, typer.Option(help="Output directory for the voice pool.")],
    target: Annotated[int, typer.Option(help="Total voices to collect.")] = 100,
    sources: Annotated[
        str, typer.Option(help="Comma-separated source names. Pilot supports: libritts-r")
    ] = "libritts-r",
    seed: Annotated[int, typer.Option(help="PRNG seed (reserved for future shuffling).")] = 42,
    min_duration_s: Annotated[float, typer.Option()] = 10.0,
    max_duration_s: Annotated[float, typer.Option()] = 30.0,
    min_rms_dbfs: Annotated[float, typer.Option()] = -40.0,
    shard_size: Annotated[int, typer.Option(help="Rows per parquet shard.")] = 500,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    log_file: Annotated[
        Path | None,
        typer.Option(help="Periodic structured progress log. Defaults to <out>/run.log."),
    ] = None,
    log_every: Annotated[
        float,
        typer.Option(help="Min seconds between log lines."),
    ] = 5.0,
) -> None:
    """Stream sources, filter, and write a deterministic reference voice pool."""
    source_names = tuple(s.strip() for s in sources.split(",") if s.strip())
    if not source_names:
        raise typer.BadParameter("--sources must list at least one source")
    try:
        src_objs = [get_source(n) for n in source_names]
    except KeyError as e:
        raise typer.BadParameter(str(e)) from e

    params = FilterParams(
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
        min_rms_dbfs=min_rms_dbfs,
    )
    writer = VoicePoolWriter.open(
        out_dir=out, seed=seed, target=target,
        sources=source_names, shard_size=shard_size, resume=resume,
    )

    log_path = log_file if log_file is not None else out / "run.log"
    logger = ProgressLogger(log_path, total=target, log_every=log_every)

    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("kept={task.fields[kept]} skipped={task.fields[skipped]}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            pbar = progress.add_task(
                "voices", total=target, kept=writer.total_voices, skipped=0,
            )

            def _on_progress(stats: BuildStats) -> None:
                progress.update(
                    pbar, completed=stats.kept,
                    kept=stats.kept, skipped=stats.skipped,
                )
                logger.log(
                    done=stats.kept,
                    kept=stats.kept, skipped=stats.skipped,
                )

            stats = run_build(
                sources=src_objs, writer=writer, params=params,
                target=target, on_progress=_on_progress,
            )
        logger.log(
            done=stats.kept, force=True,
            kept=stats.kept, skipped=stats.skipped,
        )
    finally:
        logger.close()

    (out / "NOTICE.md").write_text(render_notice(writer.records))
    console.print(
        f"[green]done:[/] {stats.kept} voices in [bold]{out}[/]. "
        f"NOTICE.md updated. Log: [dim]{log_path}[/]"
    )
