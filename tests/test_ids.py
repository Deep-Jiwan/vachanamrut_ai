"""Reference resolution must accept every convention a question might use."""
import pytest

from vachanamrut_rag.ids import (
    SECTION_SIZES, all_slugs, find_references, parse_reference,
)


@pytest.mark.parametrize("text,expected", [
    # The book's own roman-numeral form.
    ("Gadhadã I-35", "Gadhadã I-35"),
    ("Gadhada I-35", "Gadhadã I-35"),
    # Gujarati ordinals, as used by Satsang exam papers.
    ("GADHADA PRATHAM 35", "Gadhadã I-35"),
    ("gadhada madhya 28", "Gadhadã II-28"),
    ("Gadhada Antya 39", "Gadhadã III-39"),
    # The book's cross-reference abbreviations.
    ("Gadh-I.35", "Gadhadã I-35"),
    ("G.II.66", "Gadhadã II-66"),
    ("Sar-9", "Sãrangpur-9"),
    # Other sections, with and without diacritics.
    ("Loya 12", "Loyã-12"),
    ("Kãriyãni-11", "Kãriyãni-11"),
    ("vartal 20", "Vartãl-20"),
    ("amdavad 4", "Amdãvãd-4"),
    ("Jetalpur-3", "Jetalpur-3"),
    ("Ashlali", "Ashlãli"),
    ("Bhugol-Khagol", "Bhugol-Khagol"),
])
def test_resolves_every_convention(text, expected):
    assert str(parse_reference(text)) == expected


def test_paragraph_references():
    ref = parse_reference("Gadh-I.77.6")
    assert (ref.section, ref.number, ref.paragraph) == ("Gadhadã I", 77, 6)


def test_finds_multiple_references_in_a_cross_reference():
    found = [str(r) for r in find_references("see {Gadh-I.77.6; Sar-9}")]
    assert found == ["Gadhadã I-77.6", "Sãrangpur-9"]


@pytest.mark.parametrize("text", ["Gadhadã I-99", "Loyã-40", "Panchãlã-8", "nonsense"])
def test_rejects_out_of_range_and_unknown(text):
    assert parse_reference(text) is None


def test_a_bare_section_name_is_not_a_discourse():
    assert parse_reference("Gadhadã") is None


def test_slug_inventory_matches_the_canon():
    slugs = all_slugs()
    assert len(slugs) == sum(SECTION_SIZES.values()) == 274
    assert len(set(slugs)) == len(slugs)
