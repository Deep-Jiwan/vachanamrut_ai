"""MCP server exposing the Vachanamrut for grounded, quoted answers.

Design goal: make quoting the path of least resistance. Every tool returns
verbatim scripture text already paired with the citation it belongs to, in the
form the tradition uses ("Gadhadã I-37.5"), so an assistant answering a
question has the exact words and the exact reference in hand and never has to
reconstruct either from memory. `verify_quote` closes the loop by checking a
quote against the text before it is presented as scripture.
"""
from __future__ import annotations

import os
from typing import Annotated, Any, Literal

from pydantic import Field

from mcp.server.mcpserver import MCPServer

from .corpus import DEFAULT_CORPUS_DIR
from .ids import SECTIONS, find_references
from .retrieve import Retriever

INSTRUCTIONS = """\
The Vachanãmrut: 273 discourses of Bhagwãn Swãminãrãyan (1819-1829), in the
official English translation, plus the book's glossary and endnotes.

How to answer questions with these tools:

1. If the question names a discourse ("Gadhadã I-37", "Gadhada Pratham 37",
   "Loya 12"), call `get_discourse` to read it in full before answering.
   Study questions are often filed under a slightly wrong number, so if the
   named discourse does not contain the answer, fall back to `search`.
2. Otherwise call `search` and read the returned paragraphs.
3. Ground every claim in a quotation. Quote the exact words returned by the
   tools -- never paraphrase into quotation marks, and never reconstruct a
   passage from memory. Cite as: Gadhadã I-37.5, where 37 is the discourse
   and 5 is the paragraph.
4. If you are unsure a quotation is exact, call `verify_quote` first.
5. If the tools return nothing relevant, say the Vachanãmrut does not appear
   to address the question rather than answering from general knowledge.

Citation format: `Gadhadã I-37.5` (section, discourse number, paragraph).
Sections: Gadhadã I, Sãrangpur, Kãriyãni, Loyã, Panchãlã, Gadhadã II, Vartãl,
Amdãvãd, Gadhadã III, and the Additional Vachanãmruts (Amdãvãd 4-8, Ashlãli,
Jetalpur 1-5, Bhugol-Khagol) accepted only by the Amdãvãd diocese.
"""

server = MCPServer(
    name="vachanamrut",
    title="Vachanãmrut",
    version="1.0.0",
    instructions=INSTRUCTIONS,
)

_retriever: Retriever | None = None


def retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        directory = os.environ.get("VACHANAMRUT_CORPUS", str(DEFAULT_CORPUS_DIR))
        _retriever = Retriever.load(directory)
    return _retriever


@server.tool(
    title="Search the Vachanamrut",
    description=(
        "Search the Vachanãmrut and return matching paragraphs with their exact "
        "text and citation. Use for questions about what Bhagwãn Swãminãrãyan "
        "said on a topic. Each result is verbatim scripture -- quote it directly."
    ),
)
def search(
    query: Annotated[str, Field(description="A question or topic, in English.")],
    limit: Annotated[int, Field(ge=1, le=25, description="Number of paragraphs.")] = 8,
    section: Annotated[str | None, Field(description="Restrict to one section, e.g. 'Gadhadã I'.")] = None,
    include_context: Annotated[bool, Field(
        description="Include the neighbouring paragraphs. Usually needed, since an "
                    "answer's question sits in the paragraph before it.")] = True,
    include_glossary: Annotated[bool, Field(
        description="Also search the glossary and endnotes for term definitions.")] = False,
) -> dict[str, Any]:
    kinds: tuple[str, ...] = ("paragraph", "glossary", "endnote") if include_glossary \
        else ("paragraph",)
    results = retriever().search(
        query, limit=limit, kinds=kinds, section=section,
        context=1 if include_context else 0,
    )
    return {
        "query": query,
        "results": [r.to_dict(include_context) for r in results],
        "note": "Text is verbatim from the Vachanãmrut. Quote it exactly and cite it.",
    }


@server.tool(
    title="Get a discourse",
    description=(
        "Retrieve one complete Vachanãmrut discourse by reference. Accepts any "
        "convention: 'Gadhadã I-37', 'Gadhada Pratham 37', 'Gadh-I.37', 'Loya 12', "
        "'Jetalpur-3'. Returns every numbered paragraph verbatim."
    ),
)
def get_discourse(
    reference: Annotated[str, Field(description="A discourse reference.")],
) -> dict[str, Any]:
    discourse = retriever().lookup(reference)
    if discourse is None:
        return {"error": f"Could not resolve {reference!r} to a discourse.",
                "hint": "Try a form like 'Gadhadã I-37' or 'Gadhada Pratham 37'."}
    return {
        "citation": discourse["citation"],
        "title": discourse["title"],
        "section": discourse["section"],
        "date": discourse["date_text"],
        "place": discourse["place"],
        "pages": discourse["pages"],
        "is_additional": discourse["is_additional"],
        "speakers": discourse["speakers"],
        "paragraphs": [
            {"citation": f"{discourse['citation']}.{p['n']}", "n": p["n"],
             "text": p["text"], "speaker": p.get("speaker"), "kind": p.get("kind")}
            for p in discourse["paragraphs"]
        ],
        "footnotes": discourse["footnotes"],
        "cross_references": discourse["cross_references"],
    }


@server.tool(
    title="Get a paragraph",
    description=(
        "Retrieve one numbered paragraph verbatim, for an exact quotation. "
        "Reference must include the paragraph, e.g. 'Gadhadã I-37.5'."
    ),
)
def get_paragraph(
    reference: Annotated[str, Field(description="e.g. 'Gadhadã I-37.5'")],
) -> dict[str, Any]:
    paragraph = retriever().paragraph(reference)
    if paragraph is None:
        return {"error": f"Could not resolve {reference!r} to a paragraph.",
                "hint": "Include the paragraph number, e.g. 'Gadhadã I-37.5'."}
    return paragraph


@server.tool(
    title="Verify a quotation",
    description=(
        "Check whether a quotation appears verbatim in the Vachanãmrut. Use this "
        "before presenting wording as scripture when you are not certain it is "
        "exact. Returns the true citation on a match."
    ),
)
def verify_quote(
    quote: Annotated[str, Field(description="The wording to check.")],
    reference: Annotated[str | None, Field(
        description="Optional discourse to search within, e.g. 'Gadhadã I-37'.")] = None,
) -> dict[str, Any]:
    return retriever().verify_quote(quote, reference)


@server.tool(
    title="Look up a term",
    description=(
        "Look up a Gujarati or Sanskrit term in the Vachanãmrut's own glossary "
        "(e.g. vairãgya, ekãntik dharma, swabhãv, mãyã). Diacritics optional."
    ),
)
def lookup_term(
    term: Annotated[str, Field(description="The term to define.")],
    limit: Annotated[int, Field(ge=1, le=10)] = 5,
) -> dict[str, Any]:
    results = retriever().search(term, limit=limit, kinds=("glossary", "endnote"),
                                 context=0)
    return {"term": term,
            "definitions": [r.to_dict(include_context=False) for r in results]}


@server.tool(
    title="List discourses",
    description=(
        "List discourses with their titles, to browse the book or find one by "
        "subject. Optionally filter by section or by words in the title."
    ),
)
def list_discourses(
    section: Annotated[str | None, Field(description=f"One of: {', '.join(SECTIONS)}")] = None,
    title_contains: Annotated[str | None, Field(description="Filter titles by substring.")] = None,
) -> dict[str, Any]:
    corpus = retriever().corpus
    rows = []
    for d in corpus.discourses:
        if section and d["section"] != section:
            continue
        if title_contains and title_contains.lower() not in d["title"].lower():
            continue
        rows.append({"citation": d["citation"], "title": d["title"],
                     "paragraphs": len(d["paragraphs"]), "date": d["date_text"]})
    return {"count": len(rows), "discourses": rows}


@server.tool(
    title="Resolve a reference",
    description=(
        "Normalise a Vachanãmrut reference written in any convention into its "
        "canonical form. 'GADHADA PRATHAM 35' becomes 'Gadhadã I-35'."
    ),
)
def resolve_reference(
    text: Annotated[str, Field(description="Text containing one or more references.")],
) -> dict[str, Any]:
    references = find_references(text)
    return {"references": [
        {"canonical": str(r), "discourse": r.citation, "section": r.section,
         "number": r.number, "paragraph": r.paragraph}
        for r in references
    ]}


def main() -> None:
    server.run(transport=os.environ.get("VACHANAMRUT_TRANSPORT", "stdio"))


if __name__ == "__main__":
    main()
