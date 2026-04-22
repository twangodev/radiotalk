from __future__ import annotations

import asyncio
import json
import os
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

from . import config as regionconfig
from .runtime import RuntimeConfig
from .generator import GenStats, run
from .scenario import (
    AirportWeighter,
    CustomWeighter,
    ScenarioSampler,
    UniformWeighter,
)
from .writer import MANIFEST_NAME, Manifest, ParquetShardWriter

data_app = typer.Typer(
    add_completion=False, help="Synthetic ATC transcript generation + packaging."
)
console = Console()


def _pick_weighter(
    name: str, path: Path | None, cfg: regionconfig.RegionConfig
) -> AirportWeighter | None:
    """Returns None to signal 'use the config's default TierWeighter'."""
    if name == "tier":
        return None   # sampler will build TierWeighter from cfg
    if name == "uniform":
        return UniformWeighter()
    if name == "custom":
        if path is None:
            raise typer.BadParameter(
                "--airports-weights is required when --weighter=custom"
            )
        return CustomWeighter(path)
    raise typer.BadParameter(f"unknown weighter: {name}")


def _parse_regions(regions: str) -> frozenset[str]:
    out = frozenset(r.strip() for r in regions.split(",") if r.strip())
    if not out:
        raise typer.BadParameter("--regions must list at least one region")
    return out


@data_app.command()
def sample(
    seed: Annotated[int, typer.Option(help="PRNG seed.")] = 42,
    n: Annotated[int, typer.Option(help="How many scenarios to print.")] = 5,
    config: Annotated[
        str,
        typer.Option(help=f"Region config (YAML). Available: {regionconfig.available()}"),
    ] = regionconfig.DEFAULT,
    weighter: Annotated[
        str, typer.Option(help="tier (from config) | uniform | custom")
    ] = "tier",
    airports_weights: Annotated[
        Path | None, typer.Option(help="CSV (icao,weight) when --weighter=custom.")
    ] = None,
    regions: Annotated[
        str | None,
        typer.Option(help="Override the config's allowed_regions (comma-separated)."),
    ] = None,
) -> None:
    """Print sampled scenarios as JSON (debugging aid)."""
    cfg = regionconfig.load(config)
    w = _pick_weighter(weighter, airports_weights, cfg)
    region_set = _parse_regions(regions) if regions else None
    sampler = ScenarioSampler(
        seed=seed, config=cfg, weighter=w, allowed_regions=region_set,  # type: ignore[arg-type]
    )
    for sc in sampler.iter(n):
        console.print_json(sc.model_dump_json())


@data_app.command()
def generate(
    count: Annotated[int, typer.Option(help="How many transcripts to generate.")],
    out: Annotated[Path, typer.Option(help="Output directory for Parquet shards.")],
    base_url: Annotated[
        str, typer.Option(help="SGLang OpenAI-compatible endpoint, e.g. http://localhost:30000/v1")
    ] = "http://localhost:30000/v1",
    api_key: Annotated[
        str, typer.Option(help="API key (local servers usually accept anything).")
    ] = "",
    model: Annotated[str, typer.Option(help="Model name.")] = "Qwen/Qwen3-32B-NVFP4",
    concurrency: Annotated[int, typer.Option(help="Concurrent in-flight requests.")] = 64,
    shard_size: Annotated[int, typer.Option(help="Rows per Parquet shard.")] = 10_000,
    seed: Annotated[int, typer.Option(help="PRNG seed.")] = 42,
    temperature: Annotated[float, typer.Option()] = 0.9,
    max_tokens: Annotated[int, typer.Option()] = 1024,
    config: Annotated[
        str,
        typer.Option(help=f"Region config (YAML). Available: {regionconfig.available()}"),
    ] = regionconfig.DEFAULT,
    weighter: Annotated[
        str, typer.Option(help="tier (from config) | uniform | custom")
    ] = "tier",
    airports_weights: Annotated[
        Path | None, typer.Option(help="CSV (icao,weight) when --weighter=custom.")
    ] = None,
    regions: Annotated[
        str | None,
        typer.Option(help="Override the config's allowed_regions (comma-separated)."),
    ] = None,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    max_retries: Annotated[int, typer.Option()] = 5,
    max_parse_retries: Annotated[
        int,
        typer.Option(help="Re-sample attempts when output fails parse/validation."),
    ] = 5,
) -> None:
    """Generate transcripts against a local OpenAI-compatible server."""
    region_cfg = regionconfig.load(config)
    w = _pick_weighter(weighter, airports_weights, region_cfg)
    region_set = _parse_regions(regions) if regions else frozenset(region_cfg.allowed_regions)
    weighter_name = w.name if w is not None else "tier_v1"
    cfg = RuntimeConfig(
        base_url=base_url,
        api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
        model=model,
        count=count,
        out_dir=out,
        concurrency=concurrency,
        temperature=temperature,
        max_tokens=max_tokens,
        shard_size=shard_size,
        seed=seed,
        weighter_name=weighter_name,
        regions=tuple(sorted(region_set)),
        config_name=config,
        max_retries=max_retries,
        max_parse_retries=max_parse_retries,
    )

    writer = ParquetShardWriter.open(
        out_dir=out,
        shard_size=shard_size,
        config_fingerprint=cfg.fingerprint(),
        seed=seed,
        resume=resume,
        overwrite=overwrite,
    )
    sampler = ScenarioSampler(
        seed=seed, config=region_cfg, weighter=w,
        allowed_regions=region_set,  # type: ignore[arg-type]
    )
    if writer.total_rows > 0:
        console.print(
            f"[yellow]resuming:[/] fast-forwarding sampler past "
            f"{writer.total_rows:,} already-generated rows..."
        )
        sampler.fast_forward(writer.total_rows)

    remaining = max(0, count - writer.total_rows)
    if remaining == 0:
        console.print("[green]nothing to do:[/] requested count already generated.")
        return

    console.print(
        f"generating [bold]{remaining:,}[/] transcripts "
        f"(seed={seed}, model={model}, concurrency={concurrency})"
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("ok={task.fields[ok]} fail={task.fields[fail]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        pbar = progress.add_task("transcripts", total=remaining, ok=0, fail=0)

        def _on_progress(stats: GenStats) -> None:
            progress.update(
                pbar, completed=stats.ok + stats.failed, ok=stats.ok, fail=stats.failed
            )

        stats = asyncio.run(
            run(
                cfg=cfg,
                scenarios=sampler.iter(remaining),
                writer=writer,
                on_progress=_on_progress,
            )
        )

    console.print(
        f"[green]done:[/] wrote {stats.ok:,} transcripts, "
        f"{stats.failed:,} failures. Shards in [bold]{out}[/]."
    )


@data_app.command()
def inspect(out: Annotated[Path, typer.Argument(help="Output directory.")]) -> None:
    """Print manifest summary for an existing output directory."""
    manifest_path = out / MANIFEST_NAME
    if not manifest_path.exists():
        typer.secho(f"no manifest at {manifest_path}", fg=typer.colors.RED)
        raise typer.Exit(1)
    m = Manifest.load(manifest_path)
    failures = out / "failures.jsonl"
    failure_count = 0
    if failures.exists():
        with failures.open() as f:
            failure_count = sum(1 for _ in f)
    shards = sorted(out.glob("shard-*.parquet"))
    console.print_json(
        json.dumps(
            {
                "out_dir": str(out),
                "seed": m.seed,
                "config_fingerprint": m.config_fingerprint,
                "total_rows": m.total_rows,
                "last_shard_index": m.last_shard_index,
                "shard_files": len(shards),
                "failures": failure_count,
                "started_at": m.started_at,
                "updated_at": m.updated_at,
            }
        )
    )


@data_app.command()
def push(
    repo_id: Annotated[str, typer.Argument(help="e.g. 'your-user/radiotalk-atc'")],
    out: Annotated[Path, typer.Option(help="Output directory with parquet shards.")],
    private: Annotated[bool, typer.Option("--private/--public")] = True,
) -> None:
    """Upload shards to the Hugging Face Hub. Requires `radiotalk[hf]`."""
    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # pragma: no cover
        raise typer.BadParameter(
            "huggingface_hub not installed. Install with: pip install 'radiotalk[hf]'"
        ) from e
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(out),
    )
    console.print(f"[green]uploaded[/] to https://huggingface.co/datasets/{repo_id}")
