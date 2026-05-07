from __future__ import annotations

from typing import Sequence

from .schema import TokenSpan


FPS = 50


def extract_token_spans(
    text: str,
    time_before: Sequence[int],
    tokenizer,
    audio_duration_s: float,
) -> list[TokenSpan]:
    text_token_ids = tokenizer.encode(text, add_special_tokens=False)
    tb_full = list(time_before)
    n_real = min(len(text_token_ids), len(tb_full))
    if n_real == 0:
        return []
    text_token_ids = text_token_ids[:n_real]
    tb = tb_full[:n_real]
    leading_trim = tb[0]

    starts: list[float] = []
    pos = 0
    for j in range(n_real):
        pos += tb[j]
        starts.append((pos - leading_trim) / FPS)
        pos += 1

    spans: list[TokenSpan] = []
    for k in range(n_real):
        end = starts[k + 1] if k + 1 < n_real else audio_duration_s
        spans.append(
            TokenSpan(
                text=tokenizer.decode([text_token_ids[k]]),
                start_s=max(0.0, starts[k]),
                end_s=max(starts[k], end),
            )
        )
    return spans
