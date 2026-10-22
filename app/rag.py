"""Index, retrieval and answer generation."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_pages
from .bm25 import BM25, rrf
from .embeddings import Embedder, get_embedder
from .faithfulness import check_faithfulness

MODES = ("local", "bm25", "hybrid")
RRF_K = 60


def default_mode() -> str:
    mode = os.getenv("RETRIEVAL", "hybrid").lower()
    if mode not in MODES:
        raise ValueError(f"Unknown RETRIEVAL '{mode}' (use one of {', '.join(MODES)})")
    return mode

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered sources provided. "
    "Cite every claim with its source number in square brackets, like [1] or [2][3]. "
    "If the sources do not contain the answer, say you could not find it in the documents. "
    "Do not use outside knowledge."
)


class Index:
    def __init__(self, path: str | Path | None = None, embedder: Embedder | None = None):
        self.path = Path(path) if path else None
        self.embedder = embedder or get_embedder()
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self.bm25 = BM25([])
        if self.path and self.path.exists():
            self._load()

    # ---- building -------------------------------------------------
    def add_document(self, name: str, pages: list[str]) -> int:
        self.remove_document(name, rebuild=False)
        new = chunk_pages(name, pages)
        self.chunks.extend(new)
        self._rebuild()
        self._save()
        return len(new)

    def remove_document(self, name: str, rebuild: bool = True) -> bool:
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.doc != name]
        removed = len(self.chunks) != before
        if removed and rebuild:
            self._rebuild()
            self._save()
        return removed

    def documents(self) -> list[dict]:
        counts: dict[str, int] = {}
        for c in self.chunks:
            counts[c.doc] = counts.get(c.doc, 0) + 1
        return [{"name": n, "chunks": k} for n, k in sorted(counts.items())]

    def _rebuild(self) -> None:
        texts = [c.text for c in self.chunks]
        self.embedder.fit(texts)
        self.matrix = self.embedder.embed_docs(texts)
        self.bm25 = BM25(texts)

    # ---- searching ------------------------------------------------
    def _vector_scores(self, query: str) -> np.ndarray:
        return self.matrix @ self.embedder.embed_query(query)

    def search(self, query: str, k: int = 4, mode: str | None = None) -> list[tuple[float, Chunk]]:
        """Top-k (score, chunk). mode: local (vectors), bm25 (keywords) or hybrid (RRF of both).

        Scores are comparable only within one mode: cosine for local, BM25 for bm25,
        fused RRF score for hybrid.
        """
        if not self.chunks:
            return []
        mode = (mode or default_mode()).lower()
        if mode not in MODES:
            raise ValueError(f"Unknown mode '{mode}' (use one of {', '.join(MODES)})")

        if mode == "local":
            scores = self._vector_scores(query)
        elif mode == "bm25":
            scores = np.asarray(self.bm25.scores(query))
        else:
            vec = self._vector_scores(query)
            bm = np.asarray(self.bm25.scores(query))
            vr = [int(i) for i in np.argsort(-vec, kind="stable") if vec[i] > 0]
            br = [int(i) for i in np.argsort(-bm, kind="stable") if bm[i] > 0]
            fused = rrf([vr, br], k=RRF_K)[:k]
            return [(s, self.chunks[i]) for i, s in fused]
        order = np.argsort(-scores, kind="stable")[:k]
        return [(float(scores[i]), self.chunks[i]) for i in order if scores[i] > 0]

    # ---- persistence ----------------------------------------------
    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(c) for c in self.chunks]), encoding="utf-8")
        tmp.replace(self.path)

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**d) for d in data]
        self._rebuild()


def build_context(hits: list[tuple[float, Chunk]]) -> str:
    return "\n\n".join(f"[{i}] ({c.doc}, p.{c.page})\n{c.text}" for i, (_, c) in enumerate(hits, start=1))


def answer(question: str, hits: list[tuple[float, Chunk]]) -> dict:
    """Return {answer, mode, sources, faithfulness}. Uses Claude if ANTHROPIC_API_KEY is set, else extractive."""
    sources = [
        {"n": i, "doc": c.doc, "page": c.page, "score": round(s, 3), "text": c.text}
        for i, (s, c) in enumerate(hits, start=1)
    ]
    if not hits:
        return {
            "answer": "I could not find anything relevant in the uploaded documents.",
            "mode": "none",
            "sources": [],
            "faithfulness": check_faithfulness("", []),
        }

    chunk_texts = [c.text for _, c in hits]

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
                max_tokens=700,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Sources:\n\n{build_context(hits)}\n\nQuestion: {question}"}],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            text = text.strip()
            return {"answer": text, "mode": "llm", "sources": sources, "faithfulness": check_faithfulness(text, chunk_texts)}
        except Exception as e:  # network/key/model problems: degrade gracefully
            note = f"(LLM unavailable: {type(e).__name__}. Showing top passages instead.)\n\n"
            return _extractive_result(note, hits, sources)

    return _extractive_result("", hits, sources)


def _extractive_result(note: str, hits: list[tuple[float, Chunk]], sources: list[dict]) -> dict:
    # Extractive answers quote the best chunk verbatim; check only the quoted passage, not the header line.
    faith = check_faithfulness(hits[0][1].text + " [1]", [c.text for _, c in hits])
    return {"answer": note + _extractive(hits), "mode": "extractive", "sources": sources, "faithfulness": faith}


def _extractive(hits: list[tuple[float, Chunk]]) -> str:
    best = hits[0][1]
    return f"Most relevant passage [1] ({best.doc}, p.{best.page}):\n\n{best.text}"
