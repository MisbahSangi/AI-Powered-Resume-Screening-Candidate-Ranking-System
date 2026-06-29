from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from src.extraction.education_classifier import extract_education_level
from src.extraction.experience_calculator import calculate_total_experience_years
from src.extraction.ner_extractor import extract_name, extract_organizations
from src.extraction.regex_extractors import extract_date_ranges, extract_email, extract_links, extract_phone
from src.extraction.skill_matcher import SkillTaxonomy, find_skills, skill_names
from src.jd_analysis.jd_parser import JobRequirements, parse_job_description
from src.parsing.resume_parser import ParsedResume, parse_resume
from src.scoring.scoring_engine import ScoreBreakdown, ScoringWeights, score_candidate
from src.summarization.summarizer import generate_recommendation, summarize

DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "data" / "skills_taxonomy.json"


@dataclass
class CandidateProfile:
    source_path: str
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    linkedin: Optional[str]
    github: Optional[str]
    skills: List[str]
    experience_years: float
    education_level: int
    raw_text: str
    summary: str
    parse_warnings: List[str]


def build_candidate_profile(resume_path: str | Path, taxonomy: SkillTaxonomy) -> CandidateProfile:
    parsed: ParsedResume = parse_resume(resume_path)

    header_text = parsed.section("header")
    experience_text = parsed.section("experience") or parsed.raw_text
    education_text = parsed.section("education") or parsed.raw_text
    skills_text = parsed.section("skills")
    # Search skills across the whole resume, not just the Skills section —
    # plenty of real skills show up only inside project/experience bullets.
    skill_search_text = parsed.raw_text

    name = extract_name(header_text, parsed.raw_text)
    email = extract_email(parsed.raw_text)
    phone = extract_phone(parsed.raw_text)
    links = extract_links(parsed.raw_text)

    matched_skills = find_skills(skill_search_text, taxonomy)
    skills = sorted(skill_names(matched_skills))

    date_ranges = extract_date_ranges(experience_text)
    experience_years = calculate_total_experience_years(date_ranges)

    education_level = extract_education_level(education_text)

    summary_source = parsed.section("summary") or parsed.section("experience") or parsed.raw_text
    summary = summarize(summary_source, num_sentences=3)

    return CandidateProfile(
        source_path=str(resume_path),
        name=name,
        email=email,
        phone=phone,
        linkedin=links["linkedin"],
        github=links["github"],
        skills=skills,
        experience_years=experience_years,
        education_level=education_level,
        raw_text=parsed.raw_text,
        summary=summary,
        parse_warnings=parsed.parse_warnings,
    )


def build_job_requirements(jd_text: str, taxonomy: SkillTaxonomy) -> JobRequirements:
    return parse_job_description(jd_text, taxonomy)


@dataclass
class CandidateResult:
    profile: CandidateProfile
    breakdown: ScoreBreakdown
    recommendation: str


def score_candidate_against_job(
    profile: CandidateProfile,
    jd: JobRequirements,
    weights: ScoringWeights | None = None,
) -> CandidateResult:
    breakdown = score_candidate(
        candidate_skills=set(profile.skills),
        candidate_resume_text=profile.raw_text,
        candidate_experience_years=profile.experience_years,
        candidate_education_level=profile.education_level,
        jd=jd,
        weights=weights,
    )
    recommendation = generate_recommendation(breakdown)
    return CandidateResult(profile=profile, breakdown=breakdown, recommendation=recommendation)


def rank_candidates(
    resume_paths: List[str | Path],
    jd_text: str,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    weights: ScoringWeights | None = None,
) -> List[CandidateResult]:
    taxonomy = SkillTaxonomy(taxonomy_path)
    jd = build_job_requirements(jd_text, taxonomy)

    results: List[CandidateResult] = []
    for path in resume_paths:
        profile = build_candidate_profile(path, taxonomy)
        result = score_candidate_against_job(profile, jd, weights)
        results.append(result)

    results.sort(key=lambda r: r.breakdown.final_score, reverse=True)
    return results
