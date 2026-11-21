"""The month range in a repeat theme's title (handoff 6.9, E1)."""

from __future__ import annotations

import repeat


def test_one_month():
    assert repeat.month_range(["2026-11-03T18:10:00-06:00", "2026-11-20T17:40:00-06:00"]) == "11月"


def test_months_in_one_year():
    assert repeat.month_range(["2026-11-20T17:40:00-06:00", "2026-09-02T18:30:00-05:00",
                               "2026-10-11T18:00:00-05:00"]) == "9月–11月"


def test_months_across_years():
    assert repeat.month_range(["2025-12-30T17:00:00-06:00", "2026-02-01T17:20:00-06:00"]) \
        == "2025.12–2026.2"


def test_uses_the_offset_in_each_string_not_utc():
    # 23:30 on Oct 31 in Chicago is already Nov 1 in UTC; Nov 30 22:00 in
    # Chicago is Dec 1 in UTC.
    assert repeat.month_range(["2026-10-31T23:30:00-05:00", "2026-11-30T22:00:00-06:00"]) \
        == "10月–11月"
