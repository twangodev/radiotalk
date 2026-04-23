from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .manifest import VoiceRecord


def render_notice(records: Iterable[VoiceRecord]) -> str:
    by_source: dict[str, list[VoiceRecord]] = defaultdict(list)
    for r in records:
        by_source[r.source].append(r)

    lines: list[str] = [
        "# radiotalk-voices — Source Attribution",
        "",
        "This voice pool aggregates clips from the following upstream datasets.",
        "Each source's license terms apply to the clips originating from it.",
        "",
    ]
    for source in sorted(by_source):
        recs = by_source[source]
        attributions = sorted({r.attribution for r in recs})
        licenses = sorted({r.license for r in recs})
        lines.append(f"## {source}")
        lines.append("")
        lines.append(f"- {len(recs)} voice(s)")
        lines.append(f"- License(s): {', '.join(licenses)}")
        lines.append("- Attribution:")
        for a in attributions:
            lines.append(f"  - {a}")
        lines.append("")
    return "\n".join(lines)