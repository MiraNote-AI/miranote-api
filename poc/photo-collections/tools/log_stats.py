"""Summarise the compose log: counts by status and reason, mean latency,
total cost. Entries whose group_key starts with cc_ are test submissions and
are skipped.

    python tools/log_stats.py [path/to/compose.jsonl]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

TEST_PREFIX = "cc_"


def summarize(path: Path) -> dict:
    by_status: Counter = Counter()
    by_reason: Counter = Counter()
    latencies = []
    cost = 0.0
    skipped = 0
    for raw in Path(path).read_text(encoding="utf-8").split("\n"):
        try:
            entry = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        if str(entry.get("group_key") or "").startswith(TEST_PREFIX):
            skipped += 1
            continue
        by_status[entry.get("status", "")] += 1
        if entry.get("reason"):
            by_reason[entry["reason"]] += 1
        if isinstance(entry.get("latencyMs"), (int, float)):
            latencies.append(entry["latencyMs"])
        if isinstance(entry.get("costCny"), (int, float)):
            cost += entry["costCny"]
    return {
        "count": sum(by_status.values()),
        "skipped_cc": skipped,
        "by_status": dict(by_status),
        "by_reason": dict(by_reason),
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "total_cost_cny": cost,
    }


def main(argv) -> int:
    path = Path(argv[1]) if len(argv) > 1 else config.LOG_PATH
    if not path.exists():
        print("no log at {}".format(path))
        return 1
    s = summarize(path)
    print("entries: {} (skipped {} cc_ test entries)".format(s["count"], s["skipped_cc"]))
    for name in ("by_status", "by_reason"):
        print("{}:".format(name))
        for key, n in sorted(s[name].items()):
            print("  {}: {}".format(key, n))
    print("mean latency: {:.0f} ms".format(s["mean_latency_ms"]))
    print("total cost: {:.4f} CNY".format(s["total_cost_cny"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
