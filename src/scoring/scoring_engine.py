from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set

from src.jd_analysis.jd_parser import JobRequirements
from src.scoring.semantic_similarity import compute_similarity


@dataclass
class ScoringWeights:
    skill: float = 0.40
    semantic: float = 0.25
    experience: float = 0.20
    education: float = 0.15

    def validate(self) -> None:
        total = self.skill + self.semantic + self.experience + self.education
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total}")


@dataclass
class ScoreBreakdown:
    final_score: float

    skill_score: float
    semantic_score: float
    experience_score: float
    education_score: float

    matched_required_skills: List[str] = field(default_factory=list)
    missing_required_skills: List[str] = field(default_factory=list)
    matched_preferred_skills: List[str] = field(default_factory=list)

    candidate_experience_years: float = 0.0
    required_experience_years: float = 0.0

    candidate_education_level: int = 0
    required_education_level: int = 0

    weights_used: ScoringWeights = field(default_factory=ScoringWeights)

    def explanation(self) -> str:
        lines = []

        n_req = len(self.matched_required_skills) + len(self.missing_required_skills)
        lines.append(
            f"Skill match: {len(self.matched_required_skills)}/{n_req} required "
            f"skills found ({self.skill_score * 100:.0f}% weighted score)."
        )
        if self.missing_required_skills:
            lines.append(f"Missing required skills: {', '.join(self.missing_required_skills)}.")
        if self.matched_preferred_skills:
            lines.append(f"Bonus — also has preferred skills: {', '.join(self.matched_preferred_skills)}.")

        lines.append(
            f"Semantic similarity to job description: {self.semantic_score:.2f} "
            f"(0=unrelated, 1=very closely matched wording/context)."
        )

        exp_verdict = "meets" if self.candidate_experience_years >= self.required_experience_years else "below"
        lines.append(
            f"Experience: {self.candidate_experience_years:.1f} years vs "
            f"{self.required_experience_years:.1f} years required ({exp_verdict} requirement)."
        )

        from src.extraction.education_classifier import level_label
        edu_verdict = "meets" if self.candidate_education_level >= self.required_education_level else "below"
        lines.append(
            f"Education: {level_label(self.candidate_education_level)} vs "
            f"{level_label(self.required_education_level)} required ({edu_verdict} requirement)."
        )

        lines.append(
            f"Final weighted score: {self.final_score * 100:.1f}/100 "
            f"(weights: skills {self.weights_used.skill:.0%}, semantic {self.weights_used.semantic:.0%}, "
            f"experience {self.weights_used.experience:.0%}, education {self.weights_used.education:.0%})."
        )
        return " ".join(lines)


def _skill_score(
    candidate_skills: Set[str], required: Set[str], preferred: Set[str]
) -> tuple[float, List[str], List[str], List[str]]:
    matched_required = sorted(candidate_skills & required)
    missing_required = sorted(required - candidate_skills)
    matched_preferred = sorted(candidate_skills & preferred)

    if not required:
        # No required skills detected in the JD at all — don't punish the
        # candidate for something the JD itself didn't specify.
        base_score = 1.0 if matched_preferred or not preferred else 0.5
    else:
        base_score = len(matched_required) / len(required)

    # Small bonus for preferred skills, capped so it can't exceed 1.0
    if preferred:
        bonus = 0.1 * (len(matched_preferred) / len(preferred))
        base_score = min(1.0, base_score + bonus)

    return base_score, matched_required, missing_required, matched_preferred


def _experience_score(candidate_years: float, required_years: float) -> float:
    if required_years <= 0:
        return 1.0  # JD didn't specify a bar — don't penalize
    ratio = candidate_years / required_years
    return round(min(1.0, ratio), 4)


def _education_score(candidate_level: int, required_level: int) -> float:
    if required_level <= 0:
        return 1.0
    if candidate_level >= required_level:
        return 1.0
    # Partial credit for being one level below (e.g. Bachelor's when
    # Master's preferred) rather than an all-or-nothing cliff.
    gap = required_level - candidate_level
    return max(0.0, 1.0 - 0.4 * gap)


def score_candidate(
    *,
    candidate_skills: Set[str],
    candidate_resume_text: str,
    candidate_experience_years: float,
    candidate_education_level: int,
    jd: JobRequirements,
    weights: ScoringWeights | None = None,
) -> ScoreBreakdown:
    weights = weights or ScoringWeights()
    weights.validate()

    skill_score, matched_req, missing_req, matched_pref = _skill_score(
        candidate_skills, jd.required_skills, jd.preferred_skills
    )
    semantic_score = compute_similarity(candidate_resume_text, jd.raw_text)
    experience_score = _experience_score(candidate_experience_years, jd.required_experience_years)
    education_score = _education_score(candidate_education_level, jd.required_education_level)

    final = (
        weights.skill * skill_score
        + weights.semantic * semantic_score
        + weights.experience * experience_score
        + weights.education * education_score
    )

    return ScoreBreakdown(
        final_score=round(final, 4),
        skill_score=round(skill_score, 4),
        semantic_score=round(semantic_score, 4),
        experience_score=round(experience_score, 4),
        education_score=round(education_score, 4),
        matched_required_skills=matched_req,
        missing_required_skills=missing_req,
        matched_preferred_skills=matched_pref,
        candidate_experience_years=candidate_experience_years,
        required_experience_years=jd.required_experience_years,
        candidate_education_level=candidate_education_level,
        required_education_level=jd.required_education_level,
        weights_used=weights,
    )
