"""Retrieval evaluation: hit@1, hit@3 and MRR over eval/questions.jsonl.

A retrieved chunk counts as correct when it comes from the expected document AND
contains the expected phrase. Run:

    python -m eval.run_eval --mode all          # compare local, bm25, hybrid
    python -m eval.run_eval --mode bm25 --verbose
    python -m eval.run_eval --mode all --dense  # vectors from sentence-transformers
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODES = ("local", "bm25", "hybrid")


def evaluate(index, questions: list[dict], mode: str, k: int, verbose: bool = False) -> dict:
    ks = sorted({1, 3, k})
    hits_at = {x: 0 for x in ks}
    rr_total = 0.0
    for q in questions:
        results = index.search(q["question"], k=k, mode=mode)
        rank = next(
            (i for i, (_, c) in enumerate(results, start=1) if c.doc == q["doc"] and q["contains"].lower() in c.text.lower()),
            None,
        )
        for x in ks:
            if rank is not None and rank <= x:
                hits_at[x] += 1
        rr_total += 1 / rank if rank else 0.0
        if verbose or rank is None:
            print(f"[{mode}] {'OK  ' if rank else 'MISS'} rank={rank}  {q['question']}")
    n = len(questions)
    return {"mode": mode, "n": n, "hit@1": hits_at[1] / n, "hit@3": hits_at[3] / n, "mrr": rr_total / n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", choices=[*MODES, "all"], default="all")
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
    modes = MODES if args.mode == "all" else (args.mode,)
    rows = [evaluate(index, questions, m, args.k, args.verbose) for m in modes]

    print(f"\nQuestions: {len(questions)}   Chunks: {len(index.chunks)}   Embedder: {'dense' if args.dense else 'local'}   k={args.k}")
    print("\n| mode   | hit@1 | hit@3 | MRR   |")
    print("|--------|-------|-------|-------|")
    for r in rows:
        print(f"| {r['mode']:<6} | {r['hit@1']:.2f}  | {r['hit@3']:.2f}  | {r['mrr']:.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
