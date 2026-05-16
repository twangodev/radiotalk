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
    TimeRemainingColumn,
)

from .._progress import ProgressLogger
from .presets import PIPELINE_VERSION
from .schema import SAMPLE_RATE
from .source import count_clean_rows, list_clean_shards, snapshot_clean_repo
from .synth import (
    append_failure,
    iter_jobs,
    run_synthesis,
    synth_config_fingerprint,
)
from .writer import RadioShardWriter


radio_app = typer.Typer(
    add_completion=False,
    help="Apply VHF AM channel artifacts to clean radiotalk audio.",
)
console = Console()


@radio_app.callback()
def _radio_callback() -> None:
    """Radio channel-simulation commands."""


@radio_app.command()
def synthesize(
    clean_repo: Annotated[
        str, typer.Option(help="HF clean audio dataset repo id."),
    ] = "twangodev/radiotalk-us-audio-tada-clean",
    out: Annotated[
        Path, typer.Option(help="Output directory for parquet shards."),
    ] = Path("out/radio-100k"),
    variants: Annotated[int, typer.Option(help="Noisy variants per clean row.")] = 3,
    workers: Annotated[
        int, typer.Option(help="Worker processes (default: physical core count)."),
    ] = 24,
    chunksize: Annotated[int, typer.Option(help="Jobs per worker batch.")] = 8,
    limit: Annotated[
        int | None, typer.Option(help="Cap on clean rows to process (debug)."),
    ] = None,
    shard_size: Annotated[int, typer.Option(help="Rows per output parquet shard.")] = 5000,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    log_file: Annotated[
        Path | None,
        typer.Option(help="Periodic structured progress log. Defaults to <out>/run.log."),
    ] = None,
    log_every: Annotated[float, typer.Option(help="Min seconds between log lines.")] = 5.0,
) -> None:
    """Stream clean audio rows → parallel channel-degrade → parquet shards."""
    out.mkdir(parents=True, exist_ok=True)
    fp = synth_config_fingerprint()

    console.print(f"[cyan]resolving clean audio[/]: {clean_repo}")
    snap_dir = snapshot_clean_repo(clean_repo)
    shards = list_clean_shards(snap_dir)
    n_clean_total = count_clean_rows(shards)
    n_clean = min(limit, n_clean_total) if limit is not None else n_clean_total
    n_jobs_total = n_clean * variants
    console.print(
        f"  {len(shards)} input shards, {n_clean:,} clean rows × {variants} variants "
        f"= {n_jobs_total:,} noisy rows"
    )
    console.print(f"  {workers} workers, chunksize={chunksize}")
    console.print(f"  pipeline fingerprint: [bold]{fp}[/]")

    extra_meta = {
        "clean_repo": clean_repo,
        "variants": variants,
        "pipeline_version": PIPELINE_VERSION,
        "sample_rate": SAMPLE_RATE,
    }
    writer = RadioShardWriter.open(
        out_dir=out, shard_size=shard_size, pipeline_fingerprint=fp,
        resume=resume, extra_meta=extra_meta,
    )

    already_done_variants = writer.total_rows
    skip_clean_rows = already_done_variants // variants if resume else 0
    if skip_clean_rows > 0:
        console.print(
            f"[yellow]resuming[/]: writer has {already_done_variants:,} rows "
            f"(skipping first {skip_clean_rows:,} clean rows)"
        )
    remaining_jobs = n_jobs_total - skip_clean_rows * variants

    log_path = log_file if log_file is not None else out / "run.log"
    logger = ProgressLogger(log_path, total=remaining_jobs, log_every=log_every)

    jobs = iter_jobs(
        shards, variants,
        skip_clean_rows=skip_clean_rows,
        limit_clean_rows=(n_clean - skip_clean_rows),
    )

    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("({task.fields[rate]:.1f} var/s)"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            pbar = progress.add_task("rows", total=remaining_jobs, rate=0.0)

            def _on_variant_ok(done: int) -> None:
                progress.update(
                    pbar, advance=1,
                    rate=done / max(progress.tasks[pbar].elapsed or 1e-6, 1e-6),
                )
                logger.log(done)

            stats = run_synthesis(
                jobs, writer,
                workers=workers, chunksize=chunksize,
                on_variant_ok=_on_variant_ok,
                log_failure=lambda rid, reason: append_failure(out, rid, reason),
            )

        logger.log(stats.variants_ok, force=True)
        console.print(
            f"  {stats.variants_ok:,} variants ok, "
            f"{stats.variants_failed:,} failed"
        )
    finally:
        writer.close()
        logger.close()

    console.print(
        f"[green]done[/]: shards in [bold]{out}[/]. log: [dim]{log_path}[/]"
    )