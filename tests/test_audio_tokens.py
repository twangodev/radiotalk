from __future__ import annotations

from radiotalk.audio.tokens import extract_token_spans


class FakeTokenizer:
    """Tiny tokenizer: each lowercase letter is one token, ' ' is one token."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_empty_text_returns_no_spans():
    spans = extract_token_spans("", [], FakeTokenizer(), 0.0)
    assert spans == []


def test_first_token_starts_at_zero_after_trim():
    # 5 silence frames before token 0 -> trimmed; token 0 should be at 0.0s.
    tb = [5, 1, 1]  # silence-before for tokens "a", "b", "c"
    spans = extract_token_spans("abc", tb, FakeTokenizer(), audio_duration_s=0.10)
    assert spans[0].start_s == 0.0
    # Token 1 starts after token 0's content (1 frame) + token 1's silence (1 frame).
    # In trimmed audio: (5 + 1 + 1 - 5) / 50 = 2/50 = 0.04s
    assert abs(spans[1].start_s - 0.04) < 1e-9
    # Token 2: (5 + 1 + 1 + 1 + 1 - 5) / 50 = 4/50 = 0.08s
    assert abs(spans[2].start_s - 0.08) < 1e-9
    # Last token end = audio_duration
    assert spans[2].end_s == 0.10


def test_end_s_chained_to_next_start():
    tb = [0, 2, 2]
    spans = extract_token_spans("abc", tb, FakeTokenizer(), audio_duration_s=0.20)
    for i in range(len(spans) - 1):
        assert spans[i].end_s == spans[i + 1].start_s


def test_token_text_round_trips():
    spans = extract_token_spans("hi", [0, 0], FakeTokenizer(), 0.04)
    assert [s.text for s in spans] == ["h", "i"]


def test_extra_time_before_clipped_to_text_length():
    # If the model emitted extra time_before slots (EOS / padding), we
    # should only emit spans up to the text-token count.
    spans = extract_token_spans("ab", [0, 0, 0, 0, 0], FakeTokenizer(), 0.04)
    assert len(spans) == 2
