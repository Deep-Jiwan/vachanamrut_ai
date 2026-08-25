"""Corpus model: discourses, retrieval chunks, and on-disk format.

Retrieval works on paragraphs rather than fixed-size windows. A paragraph is
the unit the Vachanamrut itself numbers, so it is both the smallest piece that
can be quoted exactly and the piece a citation points at. Every chunk carries
the citation for the text it holds, which is what lets an answer quote the
scripture rather than paraphrase it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"


@dataclass
class Chunk:
    """One retrievable, citable unit of text."""

    id: str                       # "gadhada-i-35#4"
    kind: str                     # paragraph | glossary | endnote
    citation: str                 # "Gadhadã I-35.4"
    text: str                     # verbatim
    discourse: str = ""           # "gadhada-i-35"
    discourse_citation: str = ""  # "Gadhadã I-35"
    title: str = ""
    section: str = ""
    number: int | None = None
    paragraph: int | None = None
    speaker: str | None = None
    para_kind: str = ""
    pages: list[int] = field(default_factory=list)
    is_additional: bool = False
    term: str = ""                # glossary entries only
    see_also: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_chunks(discourses: list[dict], glossary: list[dict],
                 endnotes: list[dict]) -> list[Chunk]:
    """Flatten the corpus into retrievable chunks."""
    chunks: list[Chunk] = []

    for d in discourses:
        for para in d["paragraphs"]:
            chunks.append(Chunk(
                id=f"{d['slug']}#{para['n']}",
                kind="paragraph",
                citation=f"{d['citation']}.{para['n']}",
                text=para["text"],
                discourse=d["slug"],
                discourse_citation=d["citation"],
                title=d["title"],
                section=d["section"],
                number=d["number"],
                paragraph=para["n"],
                speaker=para.get("speaker"),
                para_kind=para.get("kind", ""),
                pages=para.get("pages", []),
                is_additional=d["is_additional"],
            ))

    for i, entry in enumerate(glossary):
        chunks.append(Chunk(
            id=f"glossary#{i}",
            kind="glossary",
            citation=f"Glossary: {entry['term']}",
            text=f"{entry['term']} — {entry['definition']}",
            term=entry["term"],
            pages=[entry["page"]],
            see_also=entry.get("see_also", []),
        ))

    for note in endnotes:
        chunks.append(Chunk(
            id=f"endnote#{note['n']}",
            kind="endnote",
            citation=f"Appendix A, Endnote {note['n']}: {note['title']}",
            text=f"{note['title']} — {note['text']}",
            term=note["title"],
            pages=[note["page"]],
            see_also=note.get("see_also", []),
        ))

    return chunks


class Corpus:
    """Loaded corpus with lookup by slug, citation and chunk id."""

    def __init__(self, discourses: list[dict], chunks: list[Chunk],
                 glossary: list[dict], endnotes: list[dict]):
        self.discourses = discourses
        self.chunks = chunks
        self.glossary = glossary
        self.endnotes = endnotes
        self.by_slug = {d["slug"]: d for d in discourses}
        self.by_id = {c.id: c for c in chunks}
        self.order = [d["slug"] for d in discourses]

    @classmethod
    def load(cls, directory: str | Path = DEFAULT_CORPUS_DIR) -> "Corpus":
        directory = Path(directory)
        discourses = list(_read_jsonl(directory / "discourses.jsonl"))
        glossary = list(_read_jsonl(directory / "glossary.jsonl"))
        endnotes = list(_read_jsonl(directory / "endnotes.jsonl"))
        # Chunks are derived, not stored: keeping a second copy of the text on
        # disk only creates a way for the two to drift apart.
        chunks = build_chunks(discourses, glossary, endnotes)
        return cls(discourses, chunks, glossary, endnotes)

    def discourse(self, slug: str) -> dict | None:
        return self.by_slug.get(slug)

    def paragraph(self, slug: str, n: int) -> dict | None:
        d = self.by_slug.get(slug)
        if not d:
            return None
        return next((p for p in d["paragraphs"] if p["n"] == n), None)

    def neighbours(self, chunk: Chunk, before: int = 1, after: int = 1) -> list[Chunk]:
        """Surrounding paragraphs, so a quoted answer keeps its question."""
        if chunk.kind != "paragraph" or chunk.paragraph is None:
            return []
        out = []
        for n in range(chunk.paragraph - before, chunk.paragraph + after + 1):
            if n == chunk.paragraph or n < 1:
                continue
            neighbour = self.by_id.get(f"{chunk.discourse}#{n}")
            if neighbour:
                out.append(neighbour)
        return out
