"""Keep the compose log to 7 days (review of api#97: it holds model text
about the photos).

The log is one JSON line per job (compose._write_log). A line is dropped
only when its `ts` is readable and older than keep_days; a line we cannot
date is never deleted. The purge shares compose.LOG_LOCK with the writer, so
a line written during a purge is not lost.
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import compose

DAY_SECONDS = 24 * 3600


def _too_old(line: str, cutoff: datetime) -> bool:
    """True only for a line with a readable timestamp before the cutoff."""
    try:
        entry = json.loads(line)
        ts = entry["ts"]
        when = datetime.fromisoformat(ts)
    except (ValueError, TypeError, KeyError):
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when < cutoff


def purge(path: Path, now: datetime, keep_days: int = 7) -> int:
    """Rewrite `path` without lines older than keep_days; returns lines removed."""
    path = Path(path)
    cutoff = now - timedelta(days=keep_days)
    with compose.LOG_LOCK:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return 0
        # Split on "\n" only: splitlines() also breaks on U+2028 and friends,
        # which ensure_ascii=False leaves raw inside a single JSON line.
        lines = [x + "\n" for x in text.split("\n")[:-1]]
        if not text.endswith("\n"):
            lines.append(text.rsplit("\n", 1)[-1])
        kept = [x for x in lines if not _too_old(x, cutoff)]
        removed = len(lines) - len(kept)
        if removed == 0:
            return 0
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("".join(kept))
        os.replace(tmp, path)
    return removed


def run_daily(path: Path, clock: Callable[[], datetime], sleep: Callable[[float], None]) -> None:
    """Purge now, then every 24 hours, until `sleep` raises."""
    while True:
        try:
            removed = purge(path, clock())
            if removed:
                print("[collections] compose log: removed {} old line(s)".format(removed))
        except Exception:  # noqa: BLE001 -- a purge failure must not stop the loop
            traceback.print_exc()
        sleep(DAY_SECONDS)
