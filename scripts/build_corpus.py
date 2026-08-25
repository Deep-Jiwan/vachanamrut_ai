"""Build the retrieval corpus from the PDF extraction, with validation.

The build fails loudly rather than shipping a corpus with holes in it: a
missing discourse or a gap in paragraph numbering would show up later as a
citation that points at the wrong text, which is the one failure mode this
system cannot tolerate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vachanamrut_rag.corpus import build_chunks, write_jsonl  # noqa: E402
from vachanamrut_rag.ids import SECTION_SIZES, all_slugs  # noqa: E402
from vachanamrut_rag.parse import load_lines, parse_lines, to_dicts  # noqa: E402

EXPECTED_DISCOURSES = 274      # 273 in the canon, plus Bhugol-Khagol
EXPECTED_MAIN = 262            # discourses accepted by every diocese
EXPECTED_ADDITIONAL = 11       # accepted only by the Amdavad diocese


def validate(discourses: list[dict], glossary: list[dict],
             endnotes: list[dict]) -> list[str]:
    problems: list[str] = []
    slugs = {d["slug"] for d in discourses}

    missing = [s for s in all_slugs() if s not in slugs]
    if missing:
        problems.append(f"missing discourses: {missing}")
    if len(discourses) != EXPECTED_DISCOURSES:
        problems.append(f"expected {EXPECTED_DISCOURSES} discourses, got {len(discourses)}")

    additional = sum(1 for d in discourses if d["is_additional"])
    if additional != EXPECTED_ADDITIONAL + 1:   # +1 for Bhugol-Khagol
        problems.append(f"expected {EXPECTED_ADDITIONAL + 1} additional, got {additional}")

    counts: dict[str, int] = {}
    for d in discourses:
        counts[d["section"]] = counts.get(d["section"], 0) + 1
    for section, expected in SECTION_SIZES.items():
        if counts.get(section, 0) != expected:
            problems.append(f"{section}: expected {expected}, got {counts.get(section, 0)}")

    for d in discourses:
        numbers = [p["n"] for p in d["paragraphs"]]
        if numbers != list(range(1, len(numbers) + 1)):
            problems.append(f"{d['citation']}: paragraph numbers not contiguous: {numbers}")
        if not d["paragraphs"]:
            problems.append(f"{d['citation']}: no paragraphs")
        for p in d["paragraphs"]:
            if not p["text"].strip():
                problems.append(f"{d['citation']}.{p['n']}: empty paragraph")

    if len(glossary) < 400:
        problems.append(f"glossary looks short: {len(glossary)} entries")
    note_numbers = [n["n"] for n in endnotes]
    if note_numbers != list(range(1, len(note_numbers) + 1)):
        problems.append(f"endnotes not contiguous: {note_numbers}")

    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default="data/cache/lines.jsonl")
    ap.add_argument("--outdir", default="data/corpus")
    ap.add_argument("--allow-problems", action="store_true",
                    help="write the corpus even if validation fails")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    discourses = to_dicts(parse_lines(load_lines(args.lines)))
    glossary = [json.loads(x) for x in (outdir / "glossary.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    endnotes = [json.loads(x) for x in (outdir / "endnotes.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]

    problems = validate(discourses, glossary, endnotes)
    if problems:
        print("VALIDATION PROBLEMS:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        if not args.allow_problems:
            sys.exit(1)

    chunks = build_chunks(discourses, glossary, endnotes)   # counted, not stored
    write_jsonl(outdir / "discourses.jsonl", discourses)

    paragraphs = sum(len(d["paragraphs"]) for d in discourses)
    words = sum(len(p["text"].split()) for d in discourses for p in d["paragraphs"])
    stats = {
        "discourses": len(discourses),
        "main": sum(1 for d in discourses if not d["is_additional"]),
        "additional": sum(1 for d in discourses if d["is_additional"]),
        "paragraphs": paragraphs,
        "words": words,
        "glossary_terms": len(glossary),
        "endnotes": len(endnotes),
        "chunks": len(chunks),
        "footnotes": sum(len(d["footnotes"]) for d in discourses),
    }
    (outdir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print("Corpus built. " + "  ".join(f"{k}={v}" for k, v in stats.items()))
    print("Validation: OK" if not problems else f"Validation: {len(problems)} problem(s)")


if __name__ == "__main__":
    main()
