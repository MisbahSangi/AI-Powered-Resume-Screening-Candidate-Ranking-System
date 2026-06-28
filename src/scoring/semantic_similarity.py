"""
semantic_similarity.py
-----------------------
Computes how semantically similar a resume is to a job description — this
is what catches matches that pure keyword/skill overlap misses (e.g. "Built
REST APIs with FastAPI" vs a JD asking for "Python web framework experience").

Two backends, auto-selected:

  1. TF-IDF + cosine similarity (scikit-learn). Zero downloads, runs anywhere,
     this is the default and what's actually tested in this build.
  2. Sentence-Transformers (all-MiniLM-L6-v2) embeddings + cosine similarity.
     Genuinely better at semantic equivalence (it understands that "FastAPI"
     relates to "Python web framework" even with zero shared words), but it
     requires downloading model weights from Hugging Face on first use —
     which this sandbox's network policy blocks, so it could not be tested
     here. The code auto-activates on any machine with normal internet
     access (`pip install sentence-transformers` is all that's needed).

Either way, the *scoring engine* that calls this module never needs to know
which backend is active — it just gets a 0..1 similarity float back.
"""

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
    """Report which similarity backend is actually active. Useful for the
    dashboard/README so the user always knows what produced a given score.
    """
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
