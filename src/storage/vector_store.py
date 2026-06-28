"""
vector_store.py
-----------------
Powers two bonus features — Vector Search and Multi-Job Candidate Matching —
using plain numpy instead of a dedicated vector database (ChromaDB/FAISS).

Why not ChromaDB here: at the scale this tool actually operates at (tens to
a few hundred resumes per run, on one machine), nearest-neighbor search
over a numpy matrix with scikit-learn's cosine_similarity is just as
correct and has zero extra moving parts — no server process, no separate
dependency that could itself fail to install or need a network call for its
own default embedding model. The interface below is intentionally the same
shape ChromaDB's API uses (add / query by vector), so swapping to ChromaDB
later for a larger deployment is a contained, single-file change.

Embedding space: every resume/JD passed in here is embedded with the SAME
backend (whatever semantic_similarity.active_backend() resolves to), fit
once across the whole corpus — unlike the pairwise comparisons in
semantic_similarity.py, this module needs one shared coordinate space so
"how similar is candidate A to candidate B" is meaningful.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class VectorRecord:
    record_id: str
    label: str  # e.g. candidate name, for human-readable results
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """In-memory vector store with optional disk persistence."""

    def __init__(self):
        self._vectorizer = None  # the fitted embedding model/vectorizer
        self._matrix: Optional[np.ndarray] = None
        self._records: List[VectorRecord] = []

    @property
    def is_empty(self) -> bool:
        return self._matrix is None or len(self._records) == 0

    def build(self, texts: List[str], records: List[VectorRecord]) -> None:
        """Fit a shared embedding space over `texts` and store the vectors.

        Tries sentence-transformers first (better semantics), falls back to
        TF-IDF (always available) — same backend-detection pattern as
        semantic_similarity.py, kept separate because this one needs a
        *corpus-level* fit rather than a pairwise comparison.
        """
        assert len(texts) == len(records), "texts and records must align 1:1"

        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            self._vectorizer = ("sbert", model)
            self._matrix = np.array(model.encode(texts))
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
            sparse_matrix = vectorizer.fit_transform(texts)
            self._vectorizer = ("tfidf", vectorizer)
            self._matrix = sparse_matrix.toarray()

        self._records = records

    def _embed_query(self, text: str) -> np.ndarray:
        kind, model = self._vectorizer
        if kind == "sbert":
            return np.array(model.encode([text]))[0]
        return model.transform([text]).toarray()[0]

    def most_similar(self, text: str, top_k: int = 5) -> List[Tuple[VectorRecord, float]]:
        """Find the `top_k` stored records most similar to `text`.

        Used for both Vector Search (find similar candidates to a resume)
        and Multi-Job Matching (find candidates similar to a NEW job
        description, without re-running the full extraction pipeline).
        """
        if self.is_empty:
            return []
        query_vec = self._embed_query(text).reshape(1, -1)
        sims = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(zip(self._records, sims), key=lambda pair: pair[1], reverse=True)
        return [(rec, round(float(score), 4)) for rec, score in ranked[:top_k]]

    def cluster(self, n_clusters: int = 3) -> dict:
        """Bonus feature: candidate clustering via KMeans over the same
        embedding space — free once the embeddings already exist.
        Returns {cluster_label: [record_id, ...]}.
        """
        from sklearn.cluster import KMeans

        if self.is_empty or len(self._records) < n_clusters:
            return {}
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self._matrix)
        clusters: dict = {}
        for record, label in zip(self._records, labels):
            clusters.setdefault(int(label), []).append(record.label)
        return clusters

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"vectorizer": self._vectorizer, "matrix": self._matrix, "records": self._records},
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        store = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        store._vectorizer = data["vectorizer"]
        store._matrix = data["matrix"]
        store._records = data["records"]
        return store
