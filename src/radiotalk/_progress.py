from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import IO


class ProgressLogger:
    """Rate-limited progress logger that writes structured lines to a file.

    Each line is a timestamp + progress metrics + caller-provided key=value
    fields. Intended for long-running batch jobs where the interactive
    progress bar isn't visible (backgrounded, redirected, or remote).
    """

    def __init__(
        self,
        path: Path,
        total: int,
        *,
        log_every: float = 5.0,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.total = total
        self.log_every = log_every
        self._fh: IO = path.open("a", buffering=1)
        self._start = time.monotonic()
        self._last = 0.0

    def log(self, done: int, *, force: bool = False, **extras: object) -> None:
        now = time.monotonic()
        if not force and now - self._last < self.log_every:
            return
        self._last = now
        elapsed = now - self._start
        rate = done / elapsed if elapsed > 0 else 0.0
        left = max(0, self.total - done)
        eta = (left / rate) if rate > 0 else float("inf")
        eta_str = str(timedelta(seconds=int(eta))) if eta != float("inf") else "inf"
        pct = (done / self.total * 100) if self.total else 100.0
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        extra_str = (
            "  " + "  ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""
        )
        self._fh.write(
            f"{ts}  done={done}  total={self.total}  rate={rate:.2f}/s  "
            f"elapsed={timedelta(seconds=int(elapsed))}  eta={eta_str}  "
            f"pct={pct:.1f}%{extra_str}\n"
        )

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
