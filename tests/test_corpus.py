"""The corpus must be complete and its citations must point at the right text."""
from vachanamrut_rag.ids import SECTION_SIZES, all_slugs


def test_every_discourse_is_present(corpus):
    assert {d["slug"] for d in corpus.discourses} == set(all_slugs())


def test_section_counts_match_the_canon(corpus):
    counts: dict[str, int] = {}
    for d in corpus.discourses:
        counts[d["section"]] = counts.get(d["section"], 0) + 1
    assert counts == SECTION_SIZES


def test_the_canon_splits_into_262_main_and_12_additional(corpus):
    additional = [d for d in corpus.discourses if d["is_additional"]]
    # 11 accepted only by the Amdavad diocese, plus Bhugol-Khagol.
    assert len(additional) == 12
    assert len(corpus.discourses) - len(additional) == 262


def test_paragraph_numbering_is_contiguous(corpus):
    for d in corpus.discourses:
        numbers = [p["n"] for p in d["paragraphs"]]
        assert numbers == list(range(1, len(numbers) + 1)), d["citation"]


def test_no_empty_paragraphs(corpus):
    for d in corpus.discourses:
        for p in d["paragraphs"]:
            assert p["text"].strip(), f"{d['citation']}.{p['n']}"


def test_extraction_did_not_split_words(corpus):
    """Justified text can produce 'enti rely'; the extractor must not."""
    joined = " ".join(p["text"] for d in corpus.discourses for p in d["paragraphs"])
    for broken in (" enti rely", " house holders", " Mah ãrãj", " spirit ual"):
        assert broken not in joined


def test_chunk_citations_resolve_to_their_own_text(corpus):
    for chunk in corpus.chunks:
        if chunk.kind != "paragraph":
            continue
        paragraph = corpus.paragraph(chunk.discourse, chunk.paragraph)
        assert paragraph is not None and paragraph["text"] == chunk.text


def test_the_loya_12_heading_typo_still_resolves(corpus):
    """The book prints 'Loya-12' without the diacritic."""
    discourse = corpus.discourse("loya-12")
    assert discourse is not None
    assert discourse["citation"] == "Loyã-12"
    assert "savikalp" in discourse["title"].lower()


def test_glossary_and_endnotes_loaded(corpus):
    assert len(corpus.glossary) > 400
    assert [n["n"] for n in corpus.endnotes] == list(range(1, len(corpus.endnotes) + 1))
