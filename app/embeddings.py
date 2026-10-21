"""Retrieval backends.

local : hashed TF-IDF vectors (numpy only, no model downloads, works offline)
dense : sentence-transformers embeddings (optional install)
Both expose fit(corpus) / embed_docs / embed_query and return L2-normalised vectors,
so cosine similarity is a dot product.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to was what when where which who why will with you your do does did can could should would".split()
)


class Embedder(Protocol):
    def fit(self, corpus: list[str]) -> None: ...
    def embed_docs(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _tokens(text: str) -> list[str]:
    words = [w for w in _TOKEN.findall(text.lower()) if w not in _STOP]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]  # unigrams + bigrams


def _hash(token: str, dim: int) -> tuple[int, float]:
    h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
    return h % dim, 1.0 if (h >> 63) & 1 else -1.0  # signed hashing reduces collision bias


class HashedTfidf:
    def __init__(self, dim: int = 4096):
        self.dim = dim
        self.idf = np.ones(dim, dtype=np.float32)

    def _counts(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokens(text):
            i, sign = _hash(tok, self.dim)
            v[i] += sign
        return v

    def fit(self, corpus: list[str]) -> None:
        n = max(len(corpus), 1)
        df = np.zeros(self.dim, dtype=np.float32)
        for text in corpus:
            seen = {_hash(t, self.dim)[0] for t in _tokens(text)}
            for i in seen:
                df[i] += 1
        self.idf = (np.log((n + 1) / (df + 1)) + 1).astype(np.float32)

    def _weight(self, counts: np.ndarray) -> np.ndarray:
        w = np.sign(counts) * np.log1p(np.abs(counts)) * self.idf  # sublinear tf
        norm = np.linalg.norm(w)
        return w / norm if norm else w

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._weight(self._counts(t)) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._weight(self._counts(text))


class Dense:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Install sentence-transformers to use EMBEDDER=dense") from e
        self.model = SentenceTransformer(model_name)

    def fit(self, corpus: list[str]) -> None:
        pass

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)


def get_embedder() -> Embedder:
    kind = os.getenv("EMBEDDER", "local").lower()
    if kind == "dense":
        return Dense()
    if kind == "local":
        return HashedTfidf()
    raise ValueError(f"Unknown EMBEDDER '{kind}' (use 'local' or 'dense')")
