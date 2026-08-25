"""Retrieval must find the right passage and hand back quotable text."""
import json
from pathlib import Path

import pytest

QUESTIONS = [json.loads(line) for line in
             (Path(__file__).parent / "eval_questions.jsonl").read_text(
                 encoding="utf-8").splitlines() if line.strip()]


def test_finds_the_answer_for_every_evaluation_question(retriever):
    """Every question with a known answer must surface it in the top 10."""
    misses = []
    for row in QUESTIONS:
        found = [r.chunk.discourse for r in retriever.search(row["q"], limit=10, context=0)]
        if row["expect"] not in found:
            misses.append((row["q"], row["expect"], found[:3]))
    assert not misses, misses


def test_ranks_the_answer_first_most_of_the_time(retriever):
    first = sum(
        1 for row in QUESTIONS
        if retriever.search(row["q"], limit=1, context=0)[0].chunk.discourse == row["expect"]
    )
    assert first / len(QUESTIONS) >= 0.75


def test_a_named_discourse_anchors_the_search(retriever):
    results = retriever.search("GADHADA PRATHAM 37 what about relatives", limit=5, context=0)
    assert results[0].chunk.discourse == "gadhada-i-37"
    assert results[0].reason == "named reference"


def test_results_carry_a_citation_and_verbatim_text(retriever):
    for result in retriever.search("conquering the mind", limit=5):
        payload = result.to_dict()
        assert payload["citation"] and payload["text"]
        assert payload["citation"].startswith(payload["discourse"])


def test_context_includes_the_neighbouring_paragraphs(retriever):
    result = retriever.search("What is the nature of God's maya?", limit=1)[0]
    numbers = [c.paragraph for c in result.context]
    assert numbers and all(abs(n - result.chunk.paragraph) == 1 for n in numbers)


def test_diacritics_are_optional_in_queries(retriever):
    with_marks = [r.chunk.id for r in retriever.search("vairãgya", limit=5, context=0)]
    without = [r.chunk.id for r in retriever.search("vairagya", limit=5, context=0)]
    assert with_marks == without


def test_section_filter(retriever):
    for result in retriever.search("God", limit=8, section="Loyã", context=0):
        assert result.chunk.section == "Loyã"


class TestQuoteVerification:
    """The guard against presenting invented wording as scripture."""

    def test_accepts_an_exact_quotation(self, retriever):
        result = retriever.verify_quote(
            "I have no affection towards any of My relatives")
        assert result["verbatim"] is True
        assert result["citation"] == "Gadhadã I-37.5"

    def test_accepts_a_quotation_regardless_of_diacritics(self, retriever):
        assert retriever.verify_quote("Maya is anything that obstructs")["verbatim"]

    def test_rejects_invented_wording(self, retriever):
        assert not retriever.verify_quote(
            "Love thy neighbour as thyself, verily I say unto you")["verbatim"]

    def test_rejects_a_plausible_paraphrase(self, retriever):
        assert not retriever.verify_quote(
            "I do not feel affection for any of my family members")["verbatim"]

    def test_can_be_scoped_to_one_discourse(self, retriever):
        assert not retriever.verify_quote(
            "I have no affection towards any of My relatives",
            reference="Loyã-12")["verbatim"]


class TestLookup:
    def test_by_gujarati_ordinal(self, retriever):
        assert retriever.lookup("Gadhada Pratham 37")["citation"] == "Gadhadã I-37"

    def test_paragraph_lookup_returns_exact_text(self, retriever):
        paragraph = retriever.paragraph("Gadhadã I-37.5")
        assert paragraph["citation"] == "Gadhadã I-37.5"
        assert paragraph["text"].startswith("“As for Me, I have no affection")

    def test_unknown_reference_returns_none(self, retriever):
        assert retriever.lookup("Gadhadã I-999") is None
