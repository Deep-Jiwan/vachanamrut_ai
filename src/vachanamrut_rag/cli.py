"""Command line access to the corpus, for checking answers without an MCP client."""
from __future__ import annotations

import argparse
import json
import sys

from .corpus import DEFAULT_CORPUS_DIR
from .retrieve import Retriever


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vachanamrut", description=__doc__)
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS_DIR))
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    sub = ap.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="search for paragraphs")
    search.add_argument("query", nargs="+")
    search.add_argument("-n", "--limit", type=int, default=5)
    search.add_argument("--section")

    show = sub.add_parser("show", help="print a discourse or paragraph")
    show.add_argument("reference", nargs="+")

    verify = sub.add_parser("verify", help="check a quotation is verbatim")
    verify.add_argument("quote", nargs="+")

    term = sub.add_parser("term", help="look up a glossary term")
    term.add_argument("word", nargs="+")

    args = ap.parse_args(argv)
    retriever = Retriever.load(args.corpus)

    if args.command == "search":
        query = " ".join(args.query)
        results = retriever.search(query, limit=args.limit, section=args.section)
        if args.json:
            print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
            return 0
        for r in results:
            print(f"\n\033[1m{r.chunk.citation}\033[0m  — {r.chunk.title}"
                  f"  [{r.reason}, {r.score:.3f}]")
            print(f"  {r.chunk.text}")
        return 0

    if args.command == "show":
        reference = " ".join(args.reference)
        paragraph = retriever.paragraph(reference)
        if paragraph:
            print(json.dumps(paragraph, ensure_ascii=False, indent=2) if args.json
                  else f"{paragraph['citation']}\n\n{paragraph['text']}")
            return 0
        discourse = retriever.lookup(reference)
        if discourse is None:
            print(f"Could not resolve {reference!r}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(discourse, ensure_ascii=False, indent=2))
            return 0
        print(f"\033[1m{discourse['citation']}  {discourse['title']}\033[0m")
        print(f"{discourse['date_text'] or ''}  ({discourse['place']})\n")
        for p in discourse["paragraphs"]:
            print(f"{p['n']:>3}  {p['text']}\n")
        return 0

    if args.command == "verify":
        print(json.dumps(retriever.verify_quote(" ".join(args.quote)),
                         ensure_ascii=False, indent=2))
        return 0

    if args.command == "term":
        results = retriever.search(" ".join(args.word), limit=5,
                                   kinds=("glossary", "endnote"), context=0)
        for r in results:
            print(f"\n\033[1m{r.chunk.citation}\033[0m\n  {r.chunk.text}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
