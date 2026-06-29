from __future__ import annotations

from typing import Optional

_SBERT_MODEL = None
_SBERT_LOAD_ATTEMPTED = False
_BACKEND_IN_USE: Optional[str] = None


def _try_load_sentence_transformer():
    global _SBERT_MODEL, _SBERT_LOAD_ATTEMPTED, _BACKEND_IN_USE
    if _SBERT_LOAD_ATTEMPTED:
        return _SBERT_MODEL
    _SBERT_LOAD_ATTEMPTED = True
    try:
        from sentence_transformers import SentenceTransformer

        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _BACKEND_IN_USE = "sentence-transformers (all-MiniLM-L6-v2)"
    except Exception:
        _SBERT_MODEL = None
    return _SBERT_MODEL


def active_backend() -> str:
    global _BACKEND_IN_USE
    if _BACKEND_IN_USE is None:
        _try_load_sentence_transformer()
        if _BACKEND_IN_USE is None:
            _BACKEND_IN_USE = "TF-IDF cosine similarity (scikit-learn fallback)"
    return _BACKEND_IN_USE


def _tfidf_similarity(text_a: str, text_b: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not text_a.strip() or not text_b.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform([text_a, text_b])
    except ValueError:
        # Happens if, after stopword removal, there's no vocabulary left
        return 0.0
    score = cosine_similarity(matrix[0], matrix[1])[0][0]
    return round(float(score), 4)


def _sbert_similarity(text_a: str, text_b: str, model) -> float:
    from sentence_transformers import util

    embeddings = model.encode([text_a, text_b], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    # SBERT cosine similarity can dip slightly negative for unrelated text;
    # clamp to 0 so it stays a clean 0..1 "match strength" scale.
    return round(max(0.0, float(score)), 4)


def compute_similarity(text_a: str, text_b: str) -> float:
    """Return a 0..1 similarity score between two texts, using the best
    available backend.
    """
    model = _try_load_sentence_transformer()
    if model is not None:
        return _sbert_similarity(text_a, text_b, model)
    return _tfidf_similarity(text_a, text_b)
