"""Turn extracted line records into structured, citable discourses.

The Vachanamrut's own structure is the citation scheme: every discourse is
identified by section and number ("Gadhadã I-35") and every paragraph within
it is numbered. That makes a paragraph the natural atomic unit for retrieval —
small enough to quote exactly, and addressable in a form the tradition already
uses.

Discourse boundaries come from the headings, which are complete and
consistently placed. The closing markers the book prints after each discourse
("|| Vachanãmrut Loyã-12 || 120 ||") are used only to recover the running
discourse count, because four of them are typeset wrongly in the source:

    Gadhadã I-13   "| Vachanãmrut ..."      single opening pipe
    Gadhadã I-33   "|| ... Gadhadã I-33 ||" running number omitted
    Gadhadã II-54  "|| Vachanãmrut II-54 ||" section name omitted
    Loyã-12        heading reads "Loya-12"  diacritic omitted

Reference resolution folds diacritics, so the Loyã-12 heading still resolves.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ids import (
    ADDITIONAL,
    SECTION_INDEX,
    SECTION_SIZES,
    Reference,
    find_references,
    parse_reference,
)

# Any closing marker line. Pipe counts vary in the source, and the running
# number is sometimes absent, so both are matched leniently.
CLOSING_MARKER = re.compile(r"^\|{1,2}\s*(?P<body>.*?)\s*\|{1,2}$")
RUNNING_NUMBER = re.compile(r"\|{1,2}\s*(?P<n>\d+)\s*\|{1,2}\s*$")
SECTION_MARKER = re.compile(r"^\|{1,2}\s*End of .*\|{1,2}$", re.IGNORECASE)

SPEECH_VERBS = (
    "asked", "inquired", "enquired", "questioned", "replied", "answered",
    "said", "explained", "continued", "added", "began", "stated", "requested",
    "responded", "commented", "remarked", "narrated", "told", "raised",
)
_SPEECH = re.compile(r"\b(" + "|".join(SPEECH_VERBS) + r")\b")

# Leading connectives to strip when reading a speaker's name off a paragraph.
_LEAD = re.compile(
    r"^(thereupon|thereafter|then|next|afterwards|after that|at this|at that time|"
    r"having said this|so|now|again|further|moreover|finally|once more|"
    r"the devotee|the sãdhu|the muni|shri)\b[,:]?\s*",
    re.IGNORECASE,
)

QUESTION_VERBS = {"asked", "inquired", "enquired", "questioned", "requested", "raised"}

# Devanagari that survived as replacement glyphs: almost no ASCII letters.
_ASCII_LETTERS = re.compile(r"[A-Za-z]")


def _same(a: str, b: str) -> bool:
    """Compare ignoring diacritics, case and punctuation."""
    fold = lambda t: re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKD", t).lower())
    return fold(a) == fold(b)


def _is_garbled(text: str) -> bool:
    """True for lines that are unmapped Devanagari rather than English."""
    letters = len(_ASCII_LETTERS.findall(text))
    return letters < max(3, 0.4 * len(text.replace(" ", "")))


@dataclass
class Paragraph:
    n: int
    text: str
    speaker: str | None = None
    kind: str = "narration"      # setting | question | answer | narration
    pages: list[int] = field(default_factory=list)
    endnote_refs: list[str] = field(default_factory=list)


@dataclass
class Footnote:
    marker: str
    text: str
    source: str | None = None


@dataclass
class Discourse:
    slug: str
    citation: str
    section: str
    section_index: int
    number: int | None
    running_number: int | None
    title: str
    is_additional: bool
    pages: list[int]
    paragraphs: list[Paragraph]
    footnotes: list[Footnote]
    date_text: str | None = None
    date_iso: str | None = None
    place: str | None = None
    speakers: list[str] = field(default_factory=list)
    cross_references: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(f"{p.n} {p.text}" for p in self.paragraphs)


# "On the night of Mãgshar sudi 4, Samvat 1876 [21 November 1819]"
_DATE = re.compile(
    r"(?P<tithi>[A-Za-zãÃ]+\s+(?:sudi|vadi)\s+\d+(?:\s*[-–]\s*\d+)?,\s*"
    r"Samvat\s+\d{4})\s*\[(?P<greg>[^\]]+)\]"
)
# Each section is named for the village the discourses were delivered in, so
# the place is a property of the section rather than something to guess at.
SECTION_PLACE = {
    "Gadhadã I": "Gadhadã", "Gadhadã II": "Gadhadã", "Gadhadã III": "Gadhadã",
    "Sãrangpur": "Sãrangpur", "Kãriyãni": "Kãriyãni", "Loyã": "Loyã",
    "Panchãlã": "Panchãlã", "Vartãl": "Vartãl", "Amdãvãd": "Amdãvãd",
    "Ashlãli": "Ashlãli", "Jetalpur": "Jetalpur", "Bhugol-Khagol": "Amdãvãd",
}

_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


def _iso_date(gregorian: str) -> str | None:
    """'21 November 1819' -> '1819-11-21'. Ranges take the first date."""
    m = re.search(r"(\d{1,2})\s+([A-Z][a-z]+)\s+(\d{4})", gregorian)
    if not m:
        return None
    day, month, year = m.group(1), m.group(2), m.group(3)
    if month not in _MONTHS:
        return None
    return f"{year}-{_MONTHS[month]:02d}-{int(day):02d}"


def _speaker_of(text: str) -> tuple[str | None, str]:
    """Read the speaker and the kind of utterance off a paragraph opening."""
    head = text[:170]
    match = _SPEECH.search(head)
    if not match:
        return None, "narration"
    verb = match.group(1)
    before = _LEAD.sub("", head[: match.start()]).strip(" ,:")
    # Keep only the trailing capitalised name phrase.
    words = before.split()
    name: list[str] = []
    for word in reversed(words):
        if word[:1].isupper() or word.lower() in {"swãmi", "muni", "bhatt", "shethg"}:
            name.insert(0, word)
        else:
            break
    speaker = " ".join(name).strip(" ,") or None
    if speaker and verb in QUESTION_VERBS:
        # "X asked Shriji Mahãrãj" — the asker is X, not the addressee.
        return speaker, "question"
    kind = "question" if verb in QUESTION_VERBS else "answer"
    return speaker, kind


def parse_lines(lines: list[dict]) -> list[Discourse]:
    """Assemble discourses from the extracted line stream."""
    start = next(i for i, r in enumerate(lines)
                 if r["kind"] == "heading" and r["text"].startswith("SECTION I"))
    end = next(i for i, r in enumerate(lines) if "End of Additional" in r["text"])
    stream = lines[start:end + 1]

    additional = {Reference(s, n).slug for s, n in ADDITIONAL}
    discourses: list[Discourse] = []

    # State for the discourse currently being assembled.
    ref: Reference | None = None
    title: list[str] = []
    paragraphs: list[Paragraph] = []
    footnotes: list[Footnote] = []
    pages: set[int] = set()
    running: int | None = None
    current: Paragraph | None = None
    # A paragraph number can be typeset alongside an inline Devanagari verse
    # rather than the body line it belongs to, so it is carried forward until
    # the next body line claims it.
    pending_number: int | None = None

    def flush() -> None:
        nonlocal ref, title, paragraphs, footnotes, pages, running, current
        nonlocal pending_number
        if ref is None:
            return
        setting = paragraphs[0].text if paragraphs else ""
        date_match = _DATE.search(setting)
        for para in paragraphs:
            para.endnote_refs = sorted(set(para.endnote_refs))
        xrefs = sorted({str(r) for p in paragraphs for r in find_references(p.text)}
                       - {ref.citation})
        discourses.append(Discourse(
            slug=ref.slug,
            citation=ref.citation,
            section=ref.section,
            section_index=SECTION_INDEX[ref.section],
            number=ref.number,
            running_number=running,
            title=" ".join(title).strip(),
            is_additional=ref.slug in additional or ref.section == "Bhugol-Khagol",
            pages=sorted(pages),
            paragraphs=paragraphs,
            footnotes=footnotes,
            date_text=date_match.group(0) if date_match else None,
            date_iso=_iso_date(date_match.group("greg")) if date_match else None,
            place=SECTION_PLACE.get(ref.section),
            speakers=sorted({p.speaker for p in paragraphs if p.speaker}),
            cross_references=xrefs,
        ))
        ref, title, paragraphs, footnotes = None, [], [], []
        pages, running, current = set(), None, None
        pending_number = None

    def starts_discourse(text: str) -> Reference | None:
        """A heading that is *only* a reference opens the next discourse."""
        candidate = parse_reference(text)
        if candidate is None:
            return None
        # Compare diacritic-insensitively: the book prints "Loya-12".
        return candidate if _same(str(candidate), text) else None

    for rec in stream:
        text, kind, page = rec["text"], rec["kind"], rec["page"]

        if rec.get("para_num", "").isdigit():
            pending_number = int(rec["para_num"])

        if kind == "heading":
            if _is_garbled(text) or text.startswith("SECTION"):
                continue
            opened = starts_discourse(text)
            if opened is not None:
                flush()
                ref = opened
            elif ref is not None and not paragraphs:
                # Titles wrap onto a second line; they precede the first paragraph.
                title.append(text)
            continue

        if kind == "footnote":
            marker = rec.get("markers", "")
            if marker or not footnotes:
                footnotes.append(Footnote(marker=marker or "*", text=text))
            else:
                footnotes[-1].text += " " + text
            continue

        # Body text.
        if SECTION_MARKER.match(text):
            continue
        marker_match = CLOSING_MARKER.match(text)
        if marker_match and ref is not None:
            number_match = RUNNING_NUMBER.search(text)
            if number_match:
                running = int(number_match.group("n"))
            continue

        pages.add(page)
        if pending_number is not None:
            speaker, utterance = _speaker_of(text)
            if speaker is None and current is not None and text.startswith(("\u201c", '"')):
                speaker, utterance = current.speaker, current.kind
            current = Paragraph(
                n=pending_number, text=text, speaker=speaker,
                kind="setting" if not paragraphs else utterance, pages=[page],
            )
            paragraphs.append(current)
            pending_number = None
        elif current is not None:
            current.text += " " + text
            if page not in current.pages:
                current.pages.append(page)
        if current is not None and rec.get("markers"):
            current.endnote_refs.append(rec["markers"])

    flush()
    return discourses


def load_lines(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def to_dicts(discourses: list[Discourse]) -> list[dict]:
    return [asdict(d) | {"text": d.text} for d in discourses]
