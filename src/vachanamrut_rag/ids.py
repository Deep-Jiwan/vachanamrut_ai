"""Canonical discourse identifiers and reference resolution.

The Vachanamrut is cited in many conventions. The same discourse appears as
"Gadhada Pratham 34", "Gadhadã I-34", "Gadh-I.34", "G.I.34" and "Gadhada 1-34"
depending on whether the writer is using Gujarati ordinals, roman numerals or
the abbreviations used in the book's own cross-references. Satsang exam papers
overwhelmingly use the Gujarati form ("GADHADA PRATHAM 35"), while the English
translation prints roman numerals.

Everything here funnels those variants into one canonical form so that a
question phrased in any convention resolves to the right text.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Canonical section names, in the order they appear in the book.
SECTIONS: list[str] = [
    "Gadhadã I",
    "Sãrangpur",
    "Kãriyãni",
    "Loyã",
    "Panchãlã",
    "Gadhadã II",
    "Vartãl",
    "Amdãvãd",
    "Gadhadã III",
    "Ashlãli",
    "Jetalpur",
    "Bhugol-Khagol",
]

SECTION_INDEX = {name: i for i, name in enumerate(SECTIONS, start=1)}

# Discourses per section, used to validate the parse and to reject out-of-range
# references such as "Gadhada I-99".
SECTION_SIZES: dict[str, int] = {
    "Gadhadã I": 78,
    "Sãrangpur": 18,
    "Kãriyãni": 12,
    "Loyã": 18,
    "Panchãlã": 7,
    "Gadhadã II": 67,
    "Vartãl": 20,
    "Amdãvãd": 8,      # 1-3 in the main text, 4-8 among the Additional Vachanamruts
    "Gadhadã III": 39,
    "Ashlãli": 1,
    "Jetalpur": 5,
    "Bhugol-Khagol": 1,
}

# Sections whose discourses are numbered; the rest are standalone.
UNNUMBERED = {"Ashlãli", "Bhugol-Khagol"}

# The 11 discourses accepted only by the Amdavad diocese.
ADDITIONAL = (
    [("Amdãvãd", n) for n in range(4, 9)]
    + [("Ashlãli", None)]
    + [("Jetalpur", n) for n in range(1, 6)]
)

# Spellings that map onto a canonical section. Keys are already normalised by
# `_norm` (diacritics stripped, lowercased, punctuation removed).
_SECTION_ALIASES: dict[str, str] = {}


def _fold(text: str) -> str:
    """Strip diacritics and case but keep punctuation, so 'Gadhadã I-1' -> 'gadhada i-1'.

    Punctuation is preserved because it is what distinguishes a paragraph
    reference ("Gadh-I.35.4") from a discourse reference followed by a number.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def _norm(text: str) -> str:
    """Fold diacritics, case and punctuation so 'Gadhadã' == 'gadhada'."""
    return re.sub(r"[^a-z0-9]+", " ", _fold(text)).strip()


def _register(canonical: str, *spellings: str) -> None:
    for spelling in spellings:
        _SECTION_ALIASES[_norm(spelling)] = canonical


# Gadhada, in roman numerals, Gujarati ordinals and English ordinals.
_register("Gadhadã I", "gadhada i", "gadhada 1", "gadhada pratham", "gadh i", "gadh 1",
          "g i", "g 1", "gadhda i", "gadhada first", "pratham", "gadhadaa i")
_register("Gadhadã II", "gadhada ii", "gadhada 2", "gadhada madhya", "gadh ii", "gadh 2",
          "g ii", "g 2", "gadhda ii", "gadhada second", "madhya", "gadhadaa ii")
_register("Gadhadã III", "gadhada iii", "gadhada 3", "gadhada antya", "gadh iii",
          "gadh 3", "g iii", "g 3", "gadhda iii", "gadhada last", "gadhada third",
          "antya", "chhellu", "chhelu", "gadhadaa iii")
_register("Sãrangpur", "sarangpur", "sar", "sarang", "sarangapur")
_register("Kãriyãni", "kariyani", "kari", "karyani", "kariani")
_register("Loyã", "loya", "loy")
_register("Panchãlã", "panchala", "pancha", "panchalla", "panchal")
_register("Vartãl", "vartal", "var", "vadtal", "vartaal")
_register("Amdãvãd", "amdavad", "amd", "ahmedabad", "amadavad", "amdaavad")
_register("Ashlãli", "ashlali", "ashlaali")
_register("Jetalpur", "jetalpur", "jet", "jetalpour")
_register("Bhugol-Khagol", "bhugol khagol", "bhugol", "khagol",
          "geography and astronomy")

# Aliases are stored space-separated but may be written with any of these
# separators ("Gadh-I", "Gadh.I", "Gadh I"). Longest first, so
# "gadhada pratham" wins over a bare "gadhada".
_SEP = r"[\s._\-\u2013\u2014]*"
_ALIAS_PATTERN = "|".join(
    _SEP.join(re.escape(word) for word in alias.split())
    for alias in sorted(_SECTION_ALIASES, key=len, reverse=True)
)

# "Gadhada Pratham 35", "Gadh-I.35", "Loya 7", "Gadhada I-35.4".
# A paragraph number must be joined by "." or ":" so that prose like
# "Gadhada I 35 4 times" does not read the 4 as a paragraph.
_REFERENCE = re.compile(
    rf"\b(?P<section>{_ALIAS_PATTERN})"
    r"(?:[\s\-\u2013\u2014.:#]*(?P<number>\d{1,3})"
    r"(?:[.:](?P<para>\d{1,3}))?)?",
)


@dataclass(frozen=True)
class Reference:
    """A resolved pointer to a discourse, optionally to one paragraph."""

    section: str
    number: int | None
    paragraph: int | None = None

    @property
    def slug(self) -> str:
        """Stable machine identifier, e.g. 'gadhada-i-35'."""
        base = _norm(self.section).replace(" ", "-")
        return f"{base}-{self.number}" if self.number else base

    @property
    def citation(self) -> str:
        """Human-facing citation as printed in the book, e.g. 'Gadhadã I-35'."""
        if self.number is None:
            return self.section
        return f"{self.section}-{self.number}"

    def __str__(self) -> str:
        if self.paragraph is None:
            return self.citation
        return f"{self.citation}.{self.paragraph}"


def canonical_section(text: str) -> str | None:
    """Resolve any spelling of a section name to its canonical form."""
    return _SECTION_ALIASES.get(_norm(text))


def is_valid(section: str, number: int | None) -> bool:
    if section not in SECTION_SIZES:
        return False
    if section in UNNUMBERED:
        return number in (None, 1)
    return number is not None and 1 <= number <= SECTION_SIZES[section]


def parse_reference(text: str) -> Reference | None:
    """Resolve the first discourse reference in `text`, or None."""
    refs = find_references(text)
    return refs[0] if refs else None


def find_references(text: str) -> list[Reference]:
    """Find every discourse reference in `text`, in order of appearance."""
    found: list[Reference] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    for match in _REFERENCE.finditer(_fold(text)):
        section = _SECTION_ALIASES.get(_norm(match.group("section")))
        if section is None:
            continue
        raw_number = match.group("number")
        number = int(raw_number) if raw_number else None
        if section in UNNUMBERED:
            number = None
        elif number is None:
            # A bare section name is not a discourse reference.
            continue
        if not is_valid(section, number):
            continue
        para = match.group("para")
        key = (section, number, int(para) if para else None)
        if key in seen:
            continue
        seen.add(key)
        found.append(Reference(section, number, int(para) if para else None))
    return found


def slug_for(section: str, number: int | None) -> str:
    return Reference(section, number).slug


def all_slugs() -> list[str]:
    """Every discourse slug in book order."""
    slugs: list[str] = []
    for section in SECTIONS:
        if section in UNNUMBERED:
            slugs.append(slug_for(section, None))
        else:
            size = SECTION_SIZES[section]
            slugs.extend(slug_for(section, n) for n in range(1, size + 1))
    return slugs
