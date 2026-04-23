from __future__ import annotations

from radiotalk.voices.manifest import VoiceRecord
from radiotalk.voices.notice import render_notice


def _rec(source: str, license_: str, attribution: str, vid: str) -> VoiceRecord:
    return VoiceRecord(
        voice_id=vid,
        source=source,
        source_speaker_id="x",
        source_clip_id="x",
        duration_s=4.0,
        sample_rate=24000,
        license=license_,
        attribution=attribution,
        clip_path=f"clips/{source}/{vid}.flac",
        selected_at="2026-04-22T15:00:00+00:00",
    )


def test_notice_groups_by_source_and_lists_counts() -> None:
    records = [
        _rec("libritts-r", "CC-BY-4.0", "LibriTTS-R, CC BY 4.0", "a" * 12),
        _rec("libritts-r", "CC-BY-4.0", "LibriTTS-R, CC BY 4.0", "b" * 12),
        _rec("vctk", "CC-BY-4.0", "VCTK, CC BY 4.0", "c" * 12),
    ]
    notice = render_notice(records)
    assert "libritts-r" in notice
    assert "vctk" in notice
    assert "2 voice(s)" in notice
    assert "1 voice(s)" in notice
    assert "CC-BY-4.0" in notice


def test_notice_dedupes_attribution_strings() -> None:
    records = [
        _rec("libritts-r", "CC-BY-4.0", "LibriTTS-R, CC BY 4.0", "a" * 12),
        _rec("libritts-r", "CC-BY-4.0", "LibriTTS-R, CC BY 4.0", "b" * 12),
    ]
    notice = render_notice(records)
    assert notice.count("LibriTTS-R, CC BY 4.0") == 1
