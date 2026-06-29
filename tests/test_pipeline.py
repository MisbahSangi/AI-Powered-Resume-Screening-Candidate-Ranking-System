import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.extraction.education_classifier import extract_education_level
from src.extraction.experience_calculator import calculate_total_experience_years
from src.extraction.ner_extractor import extract_organizations
from src.extraction.regex_extractors import extract_date_ranges, extract_email, extract_phone
from src.extraction.skill_matcher import SkillTaxonomy, find_skills, skill_names
from src.jd_analysis.jd_parser import parse_job_description
from src.pipeline import rank_candidates
from src.scoring.scoring_engine import ScoringWeights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = PROJECT_ROOT / "data" / "skills_taxonomy.json"
SAMPLE_RESUMES = PROJECT_ROOT / "data" / "sample_resumes"
SAMPLE_JD = PROJECT_ROOT / "data" / "sample_job_descriptions" / "backend_engineer_jd.txt"


@pytest.fixture(scope="module")
def taxonomy():
    return SkillTaxonomy(TAXONOMY_PATH)


def test_email_extraction():
    assert extract_email("Contact me at jane.doe@example.com please") == "jane.doe@example.com"
    assert extract_email("no email here") is None


def test_phone_extraction():
    assert extract_phone("Call +92 300 1234567 anytime") is not None
    assert extract_phone("no phone number in this text") is None


def test_education_level_does_not_false_positive_on_systems():
    # Regression test: "Systems Ltd" previously matched the Master's regex
    # because "ms" appears inside "Systems" without a leading word boundary.
    text = "Software Engineer at Systems Ltd, Bachelor of Science in Computer Science"
    assert extract_education_level(text) == 2  # Bachelor's, not Master's


def test_education_level_detects_phd():
    assert extract_education_level("PhD in Machine Learning") == 4


def test_experience_calculation_merges_overlap():
    ranges = [("Jan 2020", "Dec 2021"), ("Jun 2021", "Jun 2022")]  # overlapping
    years = calculate_total_experience_years(ranges)
    # Should be ~2.5 years (merged), NOT 1.99 + 1.0 = ~3 years (double-counted)
    assert 2.3 <= years <= 2.7


def test_org_extraction_ignores_contact_lines():
    # Regression test: lines with emails/URLs containing "|" previously
    # got misread as "Company | Title" pairs.
    text = "jane@example.com | +1 555 1234\nSoftware Engineer at Acme Corp\nJan 2022 - Present"
    orgs = extract_organizations(text)
    assert "Acme Corp" in orgs
    assert not any("@" in org for org in orgs)


def test_skill_matcher_exact_and_fuzzy(taxonomy):
    matches = find_skills("Experienced in Python, FastAPI, and Reactt Native (typo)", taxonomy)
    names = skill_names(matches)
    assert "Python" in names
    assert "FastAPI" in names
    assert "React Native" in names  # caught despite the typo


def test_jd_parser_splits_required_and_preferred(taxonomy):
    jd_text = SAMPLE_JD.read_text()
    jd = parse_job_description(jd_text, taxonomy)
    assert "Python" in jd.required_skills
    assert "Docker" in jd.preferred_skills
    assert jd.required_experience_years == 3.0


def test_end_to_end_ranking_orders_candidates_sensibly():
    """The strongest-matching candidate must outrank an unrelated-field
    candidate. This is the one test that would catch a totally broken
    scoring engine even if every individual unit test passed.
    """
    resumes = sorted(SAMPLE_RESUMES.glob("*.txt"))  # txt only, for speed/determinism
    jd_text = SAMPLE_JD.read_text()
    results = rank_candidates(resumes, jd_text)

    names_in_rank_order = [r.profile.name for r in results]
    assert names_in_rank_order[0] == "Ahmed Raza"  # strong backend match
    assert results[0].breakdown.final_score > results[-1].breakdown.final_score


def test_docx_parsing_with_real_file(taxonomy):
    """Regression test: DOCX parsing was previously unverified (code existed
    but was never run against an actual .docx file). Confirmed working as
    of this test — extracts name, contact info, skills, experience, and
    education correctly from a real python-docx-generated file.
    """
    from src.pipeline import build_candidate_profile

    docx_path = SAMPLE_RESUMES / "hina_tariq.docx"
    assert docx_path.exists(), "Sample DOCX fixture is missing"

    profile = build_candidate_profile(docx_path, taxonomy)
    assert profile.name == "Hina Tariq"
    assert profile.email == "hina.tariq@email.com"
    assert "Python" in profile.skills
    assert "FastAPI" in profile.skills
    assert profile.education_level == 2  # Bachelor's
    assert profile.experience_years > 0


def test_scoring_weights_must_sum_to_one():
    bad_weights = ScoringWeights(skill=0.5, semantic=0.5, experience=0.5, education=0.5)
    with pytest.raises(ValueError):
        bad_weights.validate()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
