"""
skill_matcher.py
-----------------
This is the explainable backbone of "skill matching" as an AI feature.

Instead of asking a model "what skills does this person have" (which can
hallucinate skills that aren't really there, or phrase real ones
differently every time you ask), we match resume text against a curated
skill taxonomy:
  1. Exact / multi-word phrase matching (skill names + known synonyms)
  2. Fuzzy matching via RapidFuzz, to tolerate typos and minor variants

The payoff: for every skill we claim a candidate has, we can point to the
exact line of text that proves it. That traceability is what "explainable
scoring methodology" actually means in practice.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

from rapidfuzz import fuzz


@dataclass
class SkillMatch:
    canonical_skill: str
    matched_text: str
    match_type: str  # "exact" or "fuzzy"
    confidence: float  # 1.0 for exact, 0.0-1.0 for fuzzy


class SkillTaxonomy:
    """Loads the skill taxonomy JSON and builds fast lookup structures."""

    def __init__(self, taxonomy_path: str | Path):
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # alias (lowercase) -> canonical skill name
        self.alias_to_canonical: Dict[str, str] = {}
        self.all_canonical: Set[str] = set()

        for category, skills in raw.items():
            if category.startswith("_"):
                continue
            for canonical, aliases in skills.items():
                self.all_canonical.add(canonical)
                self.alias_to_canonical[canonical.lower()] = canonical
                for alias in aliases:
                    self.alias_to_canonical[alias.lower().strip()] = canonical

        # Sort aliases longest-first so multi-word skills match before
        # shorter substrings of themselves (e.g. "react native" before "react")
        self._sorted_aliases = sorted(self.alias_to_canonical.keys(), key=len, reverse=True)


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9.#+\s/-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_skills(
    text: str,
    taxonomy: SkillTaxonomy,
    fuzzy_threshold: int = 88,
) -> List[SkillMatch]:
    """Find every taxonomy skill present in `text`.

    Returns one SkillMatch per *canonical* skill found (deduplicated), with
    the matched substring kept for auditability.
    """
    normalized = _normalize(text)
    found: Dict[str, SkillMatch] = {}

    # --- Pass 1: exact phrase matching (word-boundary aware) ---
    for alias in taxonomy._sorted_aliases:
        canonical = taxonomy.alias_to_canonical[alias]
        if canonical in found:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        match = re.search(pattern, normalized)
        if match:
            found[canonical] = SkillMatch(
                canonical_skill=canonical,
                matched_text=match.group(0),
                match_type="exact",
                confidence=1.0,
            )

    # --- Pass 2: fuzzy matching on remaining tokens, for typos/variants ---
    # Only run against short n-grams to keep this fast and avoid matching
    # unrelated long sentences to a short skill name.
    tokens = normalized.split()
    candidate_ngrams = set(tokens)
    for n in (2, 3):
        candidate_ngrams.update(
            " ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
        )

    for canonical in taxonomy.all_canonical:
        if canonical in found:
            continue
        canonical_norm = canonical.lower()
        best_score = 0.0
        best_ngram = ""
        for ngram in candidate_ngrams:
            if abs(len(ngram) - len(canonical_norm)) > 4:
                continue  # skip wildly different lengths, saves time
            score = fuzz.ratio(canonical_norm, ngram)
            if score > best_score:
                best_score = score
                best_ngram = ngram
        if best_score >= fuzzy_threshold:
            found[canonical] = SkillMatch(
                canonical_skill=canonical,
                matched_text=best_ngram,
                match_type="fuzzy",
                confidence=round(best_score / 100, 2),
            )

    return list(found.values())


def skill_names(matches: List[SkillMatch]) -> Set[str]:
    return {m.canonical_skill for m in matches}
