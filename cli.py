from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from src.extraction.education_classifier import level_label
from src.pipeline import rank_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-powered resume screening & ranking (CLI)")
    parser.add_argument("--jd", required=True, help="Path to a job description .txt file")
    parser.add_argument(
        "--resumes", required=True, nargs="+", help="One or more resume file paths (.pdf/.docx/.txt)"
    )
    parser.add_argument("--output", default="results.csv", help="Where to write the ranked CSV")
    parser.add_argument(
        "--taxonomy",
        default=str(Path(__file__).parent / "data" / "skills_taxonomy.json"),
        help="Path to the skill taxonomy JSON (default: bundled taxonomy)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    jd_path = Path(args.jd)
    if not jd_path.exists():
        print(f"Job description file not found: {jd_path}", file=sys.stderr)
        return 1

    jd_text = jd_path.read_text(encoding="utf-8", errors="ignore")
    resume_paths = [Path(p) for p in args.resumes]

    missing = [p for p in resume_paths if not p.exists()]
    if missing:
        print(f"Resume file(s) not found: {missing}", file=sys.stderr)
        return 1

    print(f"Processing {len(resume_paths)} resume(s) against job description '{jd_path.name}'...\n")
    results = rank_candidates(resume_paths, jd_text, taxonomy_path=args.taxonomy)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank", "name", "email", "phone", "final_score", "skill_score",
                "semantic_score", "experience_score", "education_score",
                "experience_years", "education_level", "matched_required_skills",
                "missing_required_skills", "recommendation",
            ]
        )
        for rank, result in enumerate(results, start=1):
            p, b = result.profile, result.breakdown
            writer.writerow(
                [
                    rank, p.name, p.email, p.phone, round(b.final_score * 100, 1),
                    b.skill_score, b.semantic_score, b.experience_score, b.education_score,
                    p.experience_years, level_label(p.education_level),
                    "; ".join(b.matched_required_skills), "; ".join(b.missing_required_skills),
                    result.recommendation,
                ]
            )

    print(f"{'Rank':<5}{'Name':<22}{'Score':<8}{'Recommendation'}")
    print("-" * 80)
    for rank, result in enumerate(results, start=1):
        p, b = result.profile, result.breakdown
        print(f"{rank:<5}{(p.name or '—'):<22}{b.final_score*100:<8.1f}{result.recommendation}")

    print(f"\nFull results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
