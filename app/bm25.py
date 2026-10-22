"""Dependency-free BM25 (Okapi) keyword ranking, plus reciprocal rank fusion.

BM25 scores a chunk by how often the query's words appear in it, down-weighting
common words (IDF) and long chunks (length normalisation `b`).
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to was what when where which who why will with you your do does did can could should would".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    return [w for w in _TOKEN.findall(text.lower()) if w not in STOPWORDS]


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in corpus]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.n = len(self.docs)
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        # Lucene-style IDF: always non-negative.
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        """BM25 score of every document for `query` (0.0 when no query term matches)."""
        out = [0.0] * self.n
        terms = set(tokenize(query))
        for i, doc in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl) if self.avgdl else self.k1
            s = 0.0
            for t in terms:
                f = doc.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out[i] = s
        return out


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank), rank starting at 1.

    `rankings` are lists of item ids, best first. Returns (id, fused_score) best first.
    Ties break on the lower id so results are deterministic.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
