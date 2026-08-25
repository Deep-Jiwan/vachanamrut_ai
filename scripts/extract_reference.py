"""Extract the Glossary and Appendix A (Endnotes) from the Vachanamrut PDF.

These are laid out as two columns — term on the left at x0 ~= 54, definition
on the right at x0 ~= 118 — so the split is taken from x-position rather than
guessed from the text. Definitions carry cross-references to discourses in
braces ("{Loyã-15.2}"), which are resolved into canonical citations so the
glossary can be used as an entry point into the text.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from vachanamrut_rag.ids import find_references  # noqa: E402

GLOSSARY_PAGES = range(701, 761)   # 700 is the section's own preface
ENDNOTE_PAGES = range(762, 767)

# Glossary terms are set in Helvetica-Bold, definitions in CenturySchoolbook.
# Font marks where a term ends (exact, unlike a column x-threshold, which cuts
# long terms mid-word); x-position marks whether a bold run *starts* an entry,
# since definitions also cite other terms in bold ("See: ekãntik dharma.").
BOLD = "Bold"
TERM_COLUMN_MAX = 110.0
LINE_TOL = 3.0
CID_GLYPH = re.compile(r"\(cid:\d+\)")
RUNNING_HEAD = re.compile(r"^\s*The Vachan[ãa]mrut\s+\d+\s*$")
# "1 The eight factors of influence" — endnotes are numbered in the left column.
ENDNOTE_START = re.compile(r"^(?P<n>\d{1,3})\s+(?P<rest>\S.*)$")


def render(chars: list[dict]) -> str:
    out: list[str] = []
    prev = None
    for c in chars:
        if prev is not None and c["x0"] - prev["x1"] > max(0.28 * c["size"], 0.9):
            out.append(" ")
        out.append(c["text"])
        prev = c
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", "".join(out))).strip()


def page_lines(page, min_size: float = 8.6) -> list[list[dict]]:
    usable = [c for c in page.chars
              if not CID_GLYPH.search(c["text"]) and min_size < round(c["size"], 1) < 11.0]
    lines: list[list[dict]] = []
    for c in sorted(usable, key=lambda c: c["top"]):
        if lines and c["top"] - lines[-1][0]["top"] <= LINE_TOL:
            lines[-1].append(c)
        else:
            lines.append([c])
    return [sorted(ln, key=lambda c: c["x0"]) for ln in lines]


def parse_glossary(pdf) -> list[dict]:
    entries: list[dict] = []
    for pno in GLOSSARY_PAGES:
        for line in page_lines(pdf.pages[pno - 1]):
            # A term is the leading run of bold chars; everything after it is
            # definition. Long terms ("Yãgnavalkya Smruti") overflow the term
            # column, so font is the only reliable boundary.
            opens_entry = (line[0]["x0"] < TERM_COLUMN_MAX
                           and BOLD in line[0]["fontname"])
            if opens_entry:
                split = next((i for i, c in enumerate(line)
                              if BOLD not in c["fontname"]), len(line))
                term, body = render(line[:split]), render(line[split:])
            else:
                term, body = "", render(line)
            # The running head splits across both columns ("The Vacha" / "nãmrut 669").
            if re.fullmatch(r"TheVachan[ãa]mrut\d+", re.sub(r"\s+", "", term + body)):
                continue
            if term and len(term) <= 3 and not body:
                continue        # the A / B / C letter dividers
            if term:
                entries.append({"term": term, "definition": body, "page": pno})
            elif body and entries:
                entries[-1]["definition"] += " " + body
    for entry in entries:
        entry["definition"] = entry["definition"].strip()
        entry["see_also"] = sorted({str(r) for r in find_references(entry["definition"])})
    return [e for e in entries if e["definition"]]


def parse_endnotes(pdf) -> list[dict]:
    """Appendix A. Each note opens with a superscript number and a bold title."""
    notes: list[dict] = []
    for pno in ENDNOTE_PAGES:
        # min_size 0 so the superscript numbers that open each note are kept.
        for line in page_lines(pdf.pages[pno - 1], min_size=0.0):
            number = render([c for c in line if round(c["size"], 1) <= 7.5])
            text = render([c for c in line if round(c["size"], 1) > 7.5])
            if not text or RUNNING_HEAD.match(text) or text in {"Appendix A", "Endnotes"}:
                continue
            if number.isdigit():
                notes.append({"n": int(number), "title": text, "text": "", "page": pno})
            elif notes:
                notes[-1]["text"] += " " + text
    for note in notes:
        note["text"] = note["text"].strip()
        note["see_also"] = sorted({str(r) for r in find_references(note["text"])})
    return notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="vachanamrut_-_english.pdf")
    ap.add_argument("--outdir", default="data/corpus")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(args.pdf) as pdf:
        glossary = parse_glossary(pdf)
        endnotes = parse_endnotes(pdf)

    for name, rows in (("glossary", glossary), ("endnotes", endnotes)):
        path = outdir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {path}: {len(rows)} entries")


if __name__ == "__main__":
    main()
