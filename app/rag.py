"""Index, retrieval and answer generation."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_pages
from .embeddings import Embedder, get_embedder

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

    # ---- searching ------------------------------------------------
    def search(self, query: str, k: int = 4) -> list[tuple[float, Chunk]]:
        if not self.chunks:
            return []
        q = self.embedder.embed_query(query)
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
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
    """Return {answer, mode, sources}. Uses Claude if ANTHROPIC_API_KEY is set, else extractive."""
    sources = [
        {"n": i, "doc": c.doc, "page": c.page, "score": round(s, 3), "text": c.text}
        for i, (s, c) in enumerate(hits, start=1)
    ]
    if not hits:
        return {"answer": "I could not find anything relevant in the uploaded documents.", "mode": "none", "sources": []}

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
            return {"answer": text.strip(), "mode": "llm", "sources": sources}
        except Exception as e:  # network/key/model problems: degrade gracefully
            note = f"(LLM unavailable: {type(e).__name__}. Showing top passages instead.)\n\n"
            return {"answer": note + _extractive(hits), "mode": "extractive", "sources": sources}

    return {"answer": _extractive(hits), "mode": "extractive", "sources": sources}


def _extractive(hits: list[tuple[float, Chunk]]) -> str:
    best = hits[0][1]
    return f"Most relevant passage [1] ({best.doc}, p.{best.page}):\n\n{best.text}"
