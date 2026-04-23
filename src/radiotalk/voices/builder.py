from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from .._pa import now_iso
from .encode import encode_flac
from .filter import FilterParams, accept_clip
from .manifest import VoiceRecord, voice_id_for
from .pool import VoicePoolWriter
from .source import Source, SourceCandidate, get_sub_pool


@dataclass
class BuildStats:
    kept: int = 0
    skipped: int = 0


ProgressCallback = Callable[[BuildStats], None]


def _record_for(
    cand: SourceCandidate,
    *,
    voice_id: str,
    sub_pool: str,
    selected_at: str,
) -> VoiceRecord:
    duration = cand.audio.shape[0] / cand.sample_rate
    return VoiceRecord(
        voice_id=voice_id,
        source=cand.source,
        source_speaker_id=cand.source_speaker_id,
        source_clip_id=cand.source_clip_id,
        duration_s=float(duration),
        sample_rate=cand.sample_rate,
        license=cand.license,
        attribution=cand.attribution,
        selected_at=selected_at,
        sub_pool=sub_pool,
        gender_hint=cand.gender_hint,
        language=cand.language,
    )


def build(
    sources: Iterable[Source],
    writer: VoicePoolWriter,
    params: FilterParams,
    target: int,
    *,
    on_progress: ProgressCallback | None = None,
) -> BuildStats:
    stats = BuildStats(kept=writer.total_voices)
    seen_speakers: set[tuple[str, str]] = set()
    selected_at = now_iso()

    for src in sources:
        if writer.total_voices >= target:
            break
        for cand in src.candidates():
            if writer.total_voices >= target:
                break
            speaker_key = (cand.source, cand.source_speaker_id)
            if speaker_key in seen_speakers:
                stats.skipped += 1
                if on_progress:
                    on_progress(stats)
                continue
            if not accept_clip(cand.audio, cand.sample_rate, params):
                stats.skipped += 1
                if on_progress:
                    on_progress(stats)
                continue
            vid = voice_id_for(
                source=cand.source,
                source_speaker_id=cand.source_speaker_id,
            )
            if writer.contains(vid):
                seen_speakers.add(speaker_key)
                continue
            blob = encode_flac(cand.audio, cand.sample_rate)
            rec = _record_for(
                cand,
                voice_id=vid,
                sub_pool=get_sub_pool(cand.source),
                selected_at=selected_at,
            )
            writer.add(rec, blob)
            seen_speakers.add(speaker_key)
            stats.kept = writer.total_voices
            if on_progress:
                on_progress(stats)

    writer.close()
    return stats
