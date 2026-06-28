from __future__ import annotations

import re
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    raw = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in raw if len(s.strip()) > 15]


def summarize(text: str, num_sentences: int = 3) -> str:
    sentences = _split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:num_sentences])

    sentence_scores = tfidf_matrix.sum(axis=1).A1  # sum of weights per sentence
    top_indices = sentence_scores.argsort()[::-1][:num_sentences]
    top_indices_sorted = sorted(top_indices)  # restore original order

    return " ".join(sentences[i] for i in top_indices_sorted)


def generate_recommendation(breakdown) -> str:
    
    if breakdown.final_score >= 0.75:
        verdict = "Strong match — recommended to shortlist."
    elif breakdown.final_score >= 0.5:
        verdict = "Moderate match — worth a closer look."
    else:
        verdict = "Weak match against this specific job description."

    parts = [verdict]
    if breakdown.missing_required_skills:
        parts.append(
            f"Key gap(s): {', '.join(breakdown.missing_required_skills[:3])}."
        )
    if breakdown.candidate_experience_years < breakdown.required_experience_years:
        parts.append(
            f"Has {breakdown.candidate_experience_years:.1f} of the "
            f"{breakdown.required_experience_years:.1f} years requested."
        )
    return " ".join(parts)
