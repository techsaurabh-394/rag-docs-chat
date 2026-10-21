"""Split text into overlapping chunks, keeping page numbers for citations."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    doc: str
    page: int
    text: str


def split_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split on paragraph/sentence boundaries into chunks of roughly `size` characters."""
    if size <= overlap:
        raise ValueError("size must be larger than overlap")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    # Sentence-ish units: keep paragraphs apart, split long ones on sentence ends.
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.replace("\n", " ").strip()
        if not para:
            continue
        units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip())

    chunks: list[str] = []
    current = ""
    for unit in units:
        # A single oversized unit gets hard-split.
        while len(unit) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(unit[:size])
            unit = unit[size - overlap:]
        if current and len(current) + 1 + len(unit) > size:
            chunks.append(current)
            current = _tail(current, overlap) + " " + unit
        else:
            current = f"{current} {unit}".strip()
    if current:
        chunks.append(current)
    return chunks


def _tail(text: str, overlap: int) -> str:
    """Last ~overlap characters, starting at a word boundary."""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def chunk_pages(doc: str, pages: list[str], size: int = 800, overlap: int = 150) -> list[Chunk]:
    out: list[Chunk] = []
    for i, page_text in enumerate(pages, start=1):
        for piece in split_text(page_text, size, overlap):
            out.append(Chunk(doc=doc, page=i, text=piece))
    return out
