"""Retrieval evaluation: hit@k and MRR over eval/questions.jsonl.

A retrieved chunk counts as correct when it comes from the expected document AND
contains the expected phrase. Run:  python -m eval.run_eval [--k 5] [--dense]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dense", action="store_true", help="use sentence-transformers embeddings")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.dense:
        os.environ["EMBEDDER"] = "dense"

    from app.rag import Index  # imported after env is set

    index = Index(path=None)
    for f in sorted((ROOT / "corpus").glob("*.txt")):
        index.add_document(f.name, [f.read_text(encoding="utf-8")])

    questions = [json.loads(line) for line in (ROOT / "questions.jsonl").read_text().splitlines() if line.strip()]
    ks = sorted({1, 3, args.k})
    hits_at = {k: 0 for k in ks}
    rr_total = 0.0

    for q in questions:
        results = index.search(q["question"], k=args.k)
        rank = next(
            (i for i, (_, c) in enumerate(results, start=1) if c.doc == q["doc"] and q["contains"].lower() in c.text.lower()),
            None,
        )
        for k in ks:
            if rank is not None and rank <= k:
                hits_at[k] += 1
        rr_total += 1 / rank if rank else 0.0
        if args.verbose or rank is None:
            print(f"{'OK  ' if rank else 'MISS'} rank={rank}  {q['question']}")

    n = len(questions)
    print(f"\nQuestions: {n}   Chunks: {len(index.chunks)}   Embedder: {'dense' if args.dense else 'local'}")
    for k in ks:
        print(f"hit@{k}: {hits_at[k]}/{n} = {hits_at[k] / n:.0%}")
    print(f"MRR: {rr_total / n:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
