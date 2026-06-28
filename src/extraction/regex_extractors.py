"""
regex_extractors.py
--------------------
Deterministic, zero-ML extraction for fields that are unambiguous patterns:
email, phone, LinkedIn/GitHub/portfolio links, and date ranges.

Using regex here instead of an LLM isn't a compromise — it's the *better*
engineering choice. An email address is a precise pattern; asking a model to
"find the email" adds cost, latency, and a (small but real) chance of error
for something regex gets right 100% of the time.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# Handles formats like: +92 300 1234567, (021) 1234567, 0300-1234567, 555-123-4567
PHONE_RE = re.compile(
    r"(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}"
)

LINKEDIN_RE = re.compile(r"(https?://)?(www\.)?linkedin\.com/in/[A-Za-z0-9\-_/]+", re.IGNORECASE)
GITHUB_RE = re.compile(r"(https?://)?(www\.)?github\.com/[A-Za-z0-9\-_/]+", re.IGNORECASE)
GENERIC_URL_RE = re.compile(r"https?://[^\s,)]+", re.IGNORECASE)

# Date range patterns, e.g. "Jan 2022 - Present", "2020-2023", "March 2021 – June 2022"
MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
DATE_TOKEN = rf"(?:{MONTH}\s+\d{{4}}|\d{{4}})"
DATE_RANGE_RE = re.compile(
    rf"({DATE_TOKEN})\s*(?:-|–|—|to)\s*(Present|Current|Now|{DATE_TOKEN})",
    re.IGNORECASE,
)


def extract_email(text: str) -> Optional[str]:
    match = EMAIL_RE.search(text)
    return match.group(0) if match else None


def extract_phone(text: str) -> Optional[str]:
    """Find the first plausible phone number (at least 7 digits total)."""
    for match in PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) >= 7:
            return match.group(0).strip()
    return None


def extract_links(text: str) -> dict:
    """Return a dict with 'linkedin', 'github', and 'other' URLs found."""
    linkedin = LINKEDIN_RE.search(text)
    github = GITHUB_RE.search(text)
    all_urls = set(GENERIC_URL_RE.findall(text)) if False else set(
        m.group(0) for m in GENERIC_URL_RE.finditer(text)
    )
    other = [
        u for u in all_urls
        if "linkedin.com" not in u.lower() and "github.com" not in u.lower()
    ]
    return {
        "linkedin": linkedin.group(0) if linkedin else None,
        "github": github.group(0) if github else None,
        "other": other,
    }


def extract_date_ranges(text: str) -> List[Tuple[str, str]]:
    """Return a list of (start, end) date string tuples found in the text.

    `end` may be the literal string 'Present' (case-normalized) to indicate
    an ongoing role — this is handled downstream in experience_calculator.
    """
    ranges = []
    for match in DATE_RANGE_RE.finditer(text):
        start, end = match.group(1), match.group(2)
        if end.lower() in ("present", "current", "now"):
            end = "Present"
        ranges.append((start.strip(), end.strip()))
    return ranges
