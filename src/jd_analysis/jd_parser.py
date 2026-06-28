from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set

from src.extraction.skill_matcher import SkillTaxonomy, find_skills, skill_names

_PREFERRED_HEADER_RE = re.compile(
    r"(preferred|nice.to.have|bonus|good.to.have|plus)", re.IGNORECASE
)
_REQUIRED_HEADER_RE = re.compile(
    r"(required|requirements|must.have|qualifications|responsibilities)",
    re.IGNORECASE,
)

_EXPERIENCE_YEARS_RE = re.compile(
    r"(\d+)\+?\s*(?:-\s*\d+\s*)?\s*years?", re.IGNORECASE
)


@dataclass
class JobRequirements:
    raw_text: str
    required_skills: Set[str] = field(default_factory=set)
    preferred_skills: Set[str] = field(default_factory=set)
    required_experience_years: float = 0.0
    required_education_level: int = 0


def _split_required_preferred(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    required_lines: List[str] = []
    preferred_lines: List[str] = []
    current = "required"

    for line in lines:
        stripped = line.strip()
        if _PREFERRED_HEADER_RE.search(stripped) and len(stripped) < 60:
            current = "preferred"
            continue
        if _REQUIRED_HEADER_RE.search(stripped) and len(stripped) < 60:
            current = "required"
            continue
        (preferred_lines if current == "preferred" else required_lines).append(line)

    return "\n".join(required_lines), "\n".join(preferred_lines)


def _extract_required_years(text: str) -> float:
    matches = _EXPERIENCE_YEARS_RE.findall(text)
    if not matches:
        return 0.0
    return float(max(int(m) for m in matches))


def parse_job_description(text: str, taxonomy: SkillTaxonomy) -> JobRequirements:
    from src.extraction.education_classifier import extract_education_level

    required_block, preferred_block = _split_required_preferred(text)

    required_matches = find_skills(required_block, taxonomy)
    preferred_matches = find_skills(preferred_block, taxonomy) if preferred_block.strip() else set()

    required = skill_names(required_matches)
    preferred = skill_names(preferred_matches) if preferred_matches else set()
    # A skill shouldn't be double-counted as both required and preferred
    preferred -= required

    return JobRequirements(
        raw_text=text,
        required_skills=required,
        preferred_skills=preferred,
        required_experience_years=_extract_required_years(text),
        required_education_level=extract_education_level(text),
    )
