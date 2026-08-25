"""Extract font-aware, structure-tagged line records from the Vachanamrut PDF.

Verbatim quoting is the whole point of this system, so body text must come
out byte-exact. Three hazards in this PDF, all solved by working at the
character level rather than taking a page's text blob:

  * Justified text makes naive extractors insert intra-word spaces
    ("enti rely"). pdfplumber's char clustering avoids this.
  * Footnotes (verse citations) and superscript markers interleave with body
    text in reading order. Splicing them into a paragraph would corrupt any
    quote taken from it.
  * Paragraph numbers sit in the left margin at superscript size. They are
    the citation anchors, so they must be captured as structure, not text.

Font size and x-position separate everything cleanly across all 768 pages:

    size <= 7.5, x0 < 52    paragraph number (left margin)
    size ~ 6.0, inline      endnote digit / footnote roman-numeral marker
    size ~ 8.2              footnote text
    size 9.0-9.8            body text (9.0 in the glossary, 9.8 in discourses)
    size >= 11.0            headings and titles

Devanagari has no usable ToUnicode map in this PDF and extracts as
"(cid:NNN)"; those chars are dropped. Every verse's romanised transliteration
and English translation survive in the footnote text that follows it.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pdfplumber

RUNNING_HEAD = re.compile(r"^\s*The Vachan[ãa]mrut\s+\d+\s*$")
CID_GLYPH = re.compile(r"\(cid:\d+\)")

SMALL_MAX = 7.5      # paragraph numbers and inline markers
FOOTNOTE_MAX = 8.6
BODY_MAX = 11.0
MARGIN_X = 52.0      # body text starts at x0 >= 54
LINE_TOL = 3.0       # line spacing is ~11.7pt, so this only merges sub/superscripts


def render(chars: list[dict]) -> str:
    """Join chars into text, inserting a space where the inter-char gap implies one."""
    out: list[str] = []
    prev = None
    for c in chars:
        if prev is not None and c["x0"] - prev["x1"] > max(0.28 * c["size"], 0.9):
            out.append(" ")
        out.append(c["text"])
        prev = c
    text = unicodedata.normalize("NFC", "".join(out))
    # The book sets two spaces after a sentence; collapse so quotes compare cleanly.
    return re.sub(r"\s+", " ", text).strip()


def cluster_lines(chars: list[dict]) -> list[list[dict]]:
    """Group chars into visual lines, keeping margin numbers with their line."""
    lines: list[list[dict]] = []
    for c in sorted(chars, key=lambda c: c["top"]):
        if lines and c["top"] - lines[-1][0]["top"] <= LINE_TOL:
            lines[-1].append(c)
        else:
            lines.append([c])
    return [sorted(ln, key=lambda c: c["x0"]) for ln in lines]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="vachanamrut_-_english.pdf")
    ap.add_argument("--out", default="data/cache/lines.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    stats = {"lines": 0, "cid": 0, "para_marks": 0, "markers": 0}
    with pdfplumber.open(args.pdf) as pdf, out.open("w", encoding="utf-8") as fh:
        for pno, page in enumerate(pdf.pages, start=1):
            usable = []
            for c in page.chars:
                if CID_GLYPH.search(c["text"]):
                    stats["cid"] += 1
                    continue
                usable.append(c)

            for line in cluster_lines(usable):
                para_mark: list[dict] = []
                markers: list[dict] = []
                bands: dict[str, list[dict]] = {}
                for c in line:
                    size = round(c["size"], 1)
                    if size <= SMALL_MAX:
                        (para_mark if c["x0"] < MARGIN_X else markers).append(c)
                    elif size <= FOOTNOTE_MAX:
                        bands.setdefault("footnote", []).append(c)
                    elif size < BODY_MAX:
                        bands.setdefault("body", []).append(c)
                    else:
                        bands.setdefault("heading", []).append(c)

                num = render(para_mark)
                marks = render(markers)
                if num:
                    stats["para_marks"] += 1
                if marks:
                    stats["markers"] += 1

                if num and not any(k in bands for k in ("heading", "body", "footnote")):
                    # The rest of this line was an unmappable Devanagari verse, so
                    # the paragraph number would otherwise be dropped. Emit it on
                    # its own; the parser attaches it to the next body line.
                    fh.write(json.dumps(
                        {"page": pno, "kind": "marker", "text": "", "para_num": num},
                        ensure_ascii=False) + "\n")
                    stats["lines"] += 1
                    num = ""

                for kind in ("heading", "body", "footnote"):
                    if kind not in bands:
                        continue
                    text = render(bands[kind])
                    if not text or RUNNING_HEAD.match(text):
                        continue
                    rec: dict = {"page": pno, "kind": kind, "text": text}
                    if num:
                        rec["para_num"] = num
                        num = ""
                    if marks:
                        rec["markers"] = marks
                        marks = ""
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["lines"] += 1

            if pno % 150 == 0:
                print(f"  ...{pno} pages", flush=True)

    print(f"Wrote {out}: {stats}")


if __name__ == "__main__":
    main()
