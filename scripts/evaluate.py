"""Measure retrieval quality against questions with known answers.

Ground truth is the discourse a question is actually answered in, read off the
text rather than assumed: several of the sample study questions are filed under
the wrong discourse number in circulating question papers, so the labels in
those papers cannot be used directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vachanamrut_rag.retrieve import Retriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--questions", default="tests/eval_questions.jsonl")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    retriever = Retriever.load(args.corpus)
    rows = [json.loads(x) for x in Path(args.questions).read_text(
        encoding="utf-8").splitlines() if x.strip()]

    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    reciprocal = 0.0
    misses = []
    for row in rows:
        results = retriever.search(row["q"], limit=args.depth, context=0)
        found = [r.chunk.discourse for r in results]
        rank = next((i + 1 for i, slug in enumerate(found) if slug == row["expect"]), None)
        if rank:
            reciprocal += 1 / rank
            for k in hits:
                if rank <= k:
                    hits[k] += 1
        else:
            misses.append((row["q"], row["expect"], found[:3]))
        if args.verbose:
            mark = f"@{rank}" if rank else "MISS"
            print(f"  {mark:>5}  {row['q'][:66]}")

    n = len(rows)
    print(f"\n{n} questions")
    for k in sorted(hits):
        print(f"  recall@{k:<3} {hits[k]/n:6.1%}  ({hits[k]}/{n})")
    print(f"  MRR      {reciprocal/n:6.3f}")
    if misses:
        print(f"\n{len(misses)} miss(es):")
        for q, expect, got in misses:
            print(f"  - {q[:70]}\n      expected {expect}, got {got}")


if __name__ == "__main__":
    main()
