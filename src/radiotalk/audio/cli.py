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
from .model import LoadedTada, TADA_ENCODER_ID, TADA_MODEL_ID, make_locked_inference_options
from .staging import Staging
from .synth import (
    append_failure,
    assemble_shards,
    encode_voice_pool,
    gather_voice_queue,
    synth_config_fingerprint,
    synthesize_voice_queue,
)
from .writer import AudioShardWriter


audio_app = typer.Typer(
    add_completion=False,
    help="Synthesize the radiotalk-us-audio-100k dataset from transcripts + voices.",
)
console = Console()


@audio_app.callback()
def _audio_callback() -> None:
    """Audio synthesis commands."""


@audio_app.command()
def build(
    out: Annotated[Path, typer.Option(help="Output directory for parquet shards.")],
    transcripts_repo: Annotated[
        str, typer.Option(help="HF transcripts dataset repo id."),
    ] = "twangodev/radiotalk-us-transcripts-100k",
    voices_repo: Annotated[
        str, typer.Option(help="HF voices dataset repo id."),
    ] = "twangodev/radiotalk-voices-2k",
    model_id: Annotated[
        str, typer.Option(help="TADA causal LM model id."),
    ] = TADA_MODEL_ID,
    encoder_id: Annotated[
        str, typer.Option(help="TADA encoder/codec model id."),
    ] = TADA_ENCODER_ID,
    limit: Annotated[
        int | None, typer.Option(help="Cap on transcripts to process (debug)."),
    ] = None,
    max_batch: Annotated[
        int, typer.Option(help="Max texts per model.generate() call (per voice)."),
    ] = 256,
    compile_model: Annotated[
        bool, typer.Option("--compile/--no-compile", help="Apply torch.compile to TADA."),
    ] = True,
    shard_size: Annotated[
        int, typer.Option(help="Turns per parquet shard."),
    ] = 5000,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    log_file: Annotated[
        Path | None,
        typer.Option(help="Periodic structured progress log. Defaults to <out>/run.log."),
    ] = None,
    log_every: Annotated[
        float, typer.Option(help="Min seconds between log lines."),
    ] = 5.0,
) -> None:
    """Synthesize transcripts × voices → audio parquet shards (per-turn rows)."""
    from datasets import load_dataset

    out.mkdir(parents=True, exist_ok=True)
    staging_path = out / ".staging" / "audio.sqlite"

    console.print(f"[cyan]loading transcripts[/]: {transcripts_repo}")
    transcripts_ds = load_dataset(transcripts_repo, split="train")
    if limit is not None:
        transcripts_ds = transcripts_ds.select(range(min(limit, len(transcripts_ds))))
    total_transcripts = len(transcripts_ds)
    console.print(f"  {total_transcripts:,} transcripts")

    console.print(f"[cyan]loading voices[/]:      {voices_repo}")
    voices_ds = load_dataset(voices_repo, split="train")
    voice_ids = list(voices_ds["voice_id"])
    console.print(f"  {len(voice_ids)} voices")

    console.print(f"[cyan]loading TADA[/]:        {model_id}")
    loaded = LoadedTada.load(model_id=model_id, encoder_id=encoder_id)
    if compile_model:
        console.print("compiling TADA (first batch will pay compile cost)...")
        loaded.compile()

    opts = make_locked_inference_options()
    fp = synth_config_fingerprint(loaded.model_id, opts)
    console.print(f"synth fingerprint: [bold]{fp}[/]")

    extra_meta = {
        "tts_model": loaded.model_id,
        "voices_repo": voices_repo,
        "transcripts_repo": transcripts_repo,
        "max_batch": max_batch,
        "compile": compile_model,
    }
    writer = AudioShardWriter.open(
        out_dir=out, shard_size=shard_size, config_fingerprint=fp,
        resume=resume, extra_meta=extra_meta,
    )
    staging = Staging(staging_path)

    log_path = log_file if log_file is not None else out / "run.log"

    try:
        console.print("[bold]phase 1[/]: assigning voices and queueing turns")
        assignments, voice_queue = gather_voice_queue(transcripts_ds, voice_ids)
        total_turns = sum(len(items) for items in voice_queue.values())
        unique_voices = len(voice_queue)
        console.print(
            f"  {total_turns:,} turns across {unique_voices:,} unique voices "
            f"(avg {total_turns / max(unique_voices, 1):.1f} turns/voice)"
        )

        already_done = staging.voices_done()
        remaining_turns = sum(
            len(items) for vid, items in voice_queue.items() if vid not in already_done
        )

        console.print(f"  encoding {unique_voices:,} reference voices once...")
        voice_cache = encode_voice_pool(loaded, voices_ds, voice_queue.keys())

        logger = ProgressLogger(log_path, total=remaining_turns, log_every=log_every)
        console.print(
            f"[bold]phase 2[/]: synthesizing per voice (max_batch={max_batch})"
        )
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("turns_ok={task.fields[turns]} fail={task.fields[fail]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            pbar = progress.add_task(
                "turns", total=remaining_turns, turns=0, fail=0,
            )

            def _on_turn_ok(turns_done: int) -> None:
                progress.update(pbar, advance=1, turns=turns_done)
                logger.log(turns_done)

            stats = synthesize_voice_queue(
                loaded, voice_cache, voice_queue, staging,
                max_batch=max_batch,
                on_turn_ok=_on_turn_ok,
                log_failure=lambda sid, reason: append_failure(out, sid, reason),
            )
            progress.update(
                pbar, completed=remaining_turns,
                turns=stats.turns_ok, fail=stats.turns_failed,
            )

        attempts = stats.turns_by_attempt or {}
        attempts_summary = ", ".join(
            f"attempt {a}: {n:,}" for a, n in sorted(attempts.items())
        ) or "(none)"
        console.print(
            f"  synthesized {stats.turns_ok:,} turns "
            f"({stats.turns_failed:,} failed) across "
            f"{stats.voices_processed:,} voices  [{attempts_summary}]"
        )

        console.print("[bold]phase 3[/]: assembling parquet shards")
        emitted, failed = assemble_shards(
            transcripts_ds, staging, writer,
            tts_model_id=loaded.model_id,
            log_failure=lambda sid, reason: append_failure(out, sid, reason),
        )
        console.print(f"  {emitted:,} scenarios emitted, {failed:,} dropped")
        logger.log(
            done=stats.turns_ok, force=True,
            turns_failed=stats.turns_failed,
            scenarios_emitted=emitted, scenarios_failed=failed,
        )
    finally:
        writer.close()
        staging.close()
        logger.close()

    console.print(
        f"[green]done[/]: shards in [bold]{out}[/]. "
        f"Staging DB: [dim]{staging_path}[/] (safe to delete after upload). "
        f"Log: [dim]{log_path}[/]"
    )
