
from __future__ import annotations

import re
from typing import Optional, Tuple

LEVEL_LABELS = {
    0: "Not specified",
    1: "Diploma / Associate",
    2: "Bachelor's",
    3: "Master's",
    4: "PhD / Doctorate",
}

# Ordered highest-confidence-first; first match wins for a given level.
# NOTE: every abbreviation token has \b on BOTH sides. A pattern like
# r"m\.?s\.?\b" (missing the leading boundary) will false-positive match
# inside ordinary words like "Systems" (...sy-ste-MS) or "Comments" — this
# bit us during testing, which is exactly why this module has a test below.
_PATTERNS: list[Tuple[int, str]] = [
    (4, r"\bph\.?d\b|\bdoctorate\b|\bd\.?phil\b"),
    (3, r"\bmaster'?s?\b|\bm\.?sc\.?\b|\bm\.?s\.?\b|\bmba\b|\bm\.?eng\b|\bm\.?tech\b|\bm\.?phil\b"),
    (2, r"\bbachelor'?s?\b|\bb\.?sc\.?\b|\bb\.?s\.?\b|\bbba\b|\bb\.?eng\b|\bb\.?tech\b|\bbse\b"),
    (1, r"\bdiploma\b|\bassociate'?s?\s+degree\b|\ba\.?a\.?\b"),
]

_COMPILED = [(level, re.compile(pat, re.IGNORECASE)) for level, pat in _PATTERNS]


def extract_education_level(text: str) -> int:
    best = 0
    for level, pattern in _COMPILED:
        if pattern.search(text):
            best = max(best, level)
    return best


def level_label(level: int) -> str:
    return LEVEL_LABELS.get(level, "Unknown")
