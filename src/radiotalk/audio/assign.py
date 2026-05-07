from __future__ import annotations

import random
from collections.abc import Iterable, Sequence


class VoicePoolTooSmall(ValueError):
    """The voice pool has fewer voices than the scenario has unique speakers."""


def unique_speakers(turns: Iterable[dict]) -> list[str]:
    return sorted({t["speaker"] for t in turns})


def assign(
    scenario_id: str,
    speakers: Sequence[str],
    voice_ids: Sequence[str],
) -> dict[str, str]:
    unique = sorted(dict.fromkeys(speakers))
    if len(unique) > len(voice_ids):
        raise VoicePoolTooSmall(
            f"scenario {scenario_id} has {len(unique)} unique speakers but "
            f"voice pool only has {len(voice_ids)}"
        )
    rng = random.Random(scenario_id)
    chosen = rng.sample(list(voice_ids), len(unique))
    return dict(zip(unique, chosen))
