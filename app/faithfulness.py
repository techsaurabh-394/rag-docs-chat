"""Cheap lexical answer-faithfulness check.

HONEST CAVEAT: this is a token-overlap heuristic, NOT an LLM judge and NOT
entailment. It catches sentences that talk about things absent from the cited
passages (new names, numbers, topics). It cannot catch a sentence that reuses
the right words but says something false (e.g. negation, swapped subjects), and
it may flag correct paraphrases that share few words with the source.

Method
1. Split the answer into sentences; strip citation markers like [1] (the
   numbers listed there decide which sources a sentence is checked against).
2. Content words = lowercase alphanumeric tokens, minus stopwords and tokens
   shorter than 3 chars, with a light suffix stemmer (plural/-ing/-ed).
3. support(sentence) = max over its cited chunks of
   |content words found in chunk| / |content words|.
   A sentence without citations is checked against all provided chunks.
4. A sentence is `supported` when support >= THRESHOLD (0.6).
5. overall `score` = mean support over sentences that have content words.
"""
from __future__ import annotations

import re

from .bm25 import STOPWORDS

THRESHOLD = 0.6

_CITE = re.compile(r"\[(\d+)\]")
_WORD = re.compile(r"[a-z0-9]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def content_words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) >= 3}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def check_faithfulness(answer: str, chunks: list[str], threshold: float = THRESHOLD) -> dict:
    """Score `answer` against `chunks` (list index i is source number i+1).

    Returns {score, threshold, unsupported, sentences:[{text, support, supported, cited}]}.
    """
    chunk_words = [content_words(c) for c in chunks]
    results = []
    for sent in split_sentences(answer):
        cited = sorted({int(n) for n in _CITE.findall(sent) if 1 <= int(n) <= len(chunks)})
        words = content_words(_CITE.sub(" ", sent))
        if not words:
            continue  # nothing checkable (e.g. just "[1]")
        pool = [chunk_words[n - 1] for n in cited] or chunk_words
        support = max((len(words & cw) / len(words) for cw in pool), default=0.0)
        results.append(
            {
                "text": _CITE.sub("", sent).strip(),
                "support": round(support, 3),
                "supported": support >= threshold,
                "cited": cited,
            }
        )
    faith = round(sum(r["support"] for r in results) / len(results), 3) if results else 0.0
    return {
        "score": faith,
        "threshold": threshold,
        "unsupported": sum(1 for r in results if not r["supported"]),
        "sentences": results,
    }
