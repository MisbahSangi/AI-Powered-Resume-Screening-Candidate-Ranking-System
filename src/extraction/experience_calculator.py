from __future__ import annotations

import datetime
import re
from typing import List, Tuple

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_single_date(token: str, *, end_of_period: bool) -> datetime.date:
    """Parse a date token like 'Jan 2022', '2022', or 'Present'."""
    token = token.strip()
    if token.lower() in ("present", "current", "now"):
        return datetime.date.today()

    month_match = re.match(r"([A-Za-z]+)\.?\s+(\d{4})", token)
    if month_match:
        month_str, year_str = month_match.groups()
        month = _MONTH_MAP.get(month_str[:3].lower(), 1)
        return datetime.date(int(year_str), month, 1)

    year_match = re.match(r"^(\d{4})$", token)
    if year_match:
        year = int(year_match.group(1))
        # Year-only: assume start of year for a start date, end of year for an end date
        return datetime.date(year, 12 if end_of_period else 1, 1)

    raise ValueError(f"Unrecognized date token: {token!r}")


def _merge_intervals(
    intervals: List[Tuple[datetime.date, datetime.date]]
) -> List[Tuple[datetime.date, datetime.date]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda iv: iv[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def calculate_total_experience_years(
    date_ranges: List[Tuple[str, str]]
) -> float:
    intervals: List[Tuple[datetime.date, datetime.date]] = []
    for start_str, end_str in date_ranges:
        try:
            start = _parse_single_date(start_str, end_of_period=False)
            end = _parse_single_date(end_str, end_of_period=True)
            if end > start:
                intervals.append((start, end))
        except ValueError:
            continue

    merged = _merge_intervals(intervals)
    total_days = sum((end - start).days for start, end in merged)
    return round(total_days / 365.25, 1)
