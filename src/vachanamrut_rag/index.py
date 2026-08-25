"""Lexical index: BM25 over the corpus with Vachanamrut-aware tokenization.

Two things make a generic tokenizer perform badly on this text:

  * Transliterated Sanskrit and Gujarati carry diacritics ("vairãgya",
    "Mahãrãj", "ãtmã"). Almost nobody types them, so tokens are folded to
    plain ASCII and a query for "vairagya" reaches "vairãgya".
  * The vocabulary is domain-specific, so an English stemmer would mangle it
    ("mãyã" -> "mãy"). Only conservative, reversible suffixes are stripped.

BM25 is implemented directly rather than pulled from a dependency: it is
thirty lines, it keeps the package installable with no network access, and it
lets the scorer expose per-field weights that the hybrid retriever needs.
"""
from __future__ import annotations

import math
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

K1 = 1.4
B = 0.72

# A question usually turns on one discriminative word — a name ("Bharatji"), a
# technical term ("anvay-vyatirek"), an image ("Vishalyakarani"). Plain BM25
# lets three merely-uncommon words outvote it, because in a corpus this small
# their IDFs are close. Documents are therefore scored partly on how many of
# the query's *rare* terms they cover, which pulls the entity back to the top.
RARE_DF = 15                 # a term in <= this many chunks counts as rare
RARE_COVERAGE_WEIGHT = 0.75

_WORD = re.compile(r"[a-z0-9]+")

# The discourse title is a strong signal ("A Merchant's Balance Sheet") but is
# much shorter than the paragraph, so it is weighted below it. Glossary terms
# are weighted above, since a term lookup should match the headword.
FIELD_WEIGHTS = {"text": 1.0, "title": 0.35, "term": 1.5}

# Words that carry no signal here. "God", "Mahãrãj" and "Shriji" are on almost
# every page, so they are near-useless for ranking but must stay searchable;
# BM25's IDF already discounts them, so only true function words are dropped.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for with
by from as is are was were be been being do does did have has had having it its
he she they them his her their you your i we our us not no nor so such very
which who whom whose what when where why how all any both each few more most
other some only own same too can will just should now upon
""".split())

# Only inflectional endings are stripped. Derivational ones are not: turning
# "liberation" into "liber" would stop it matching itself, and this corpus uses
# such nouns as fixed terms ("liberation", "realisation", "attainment").
_SUFFIXES = ("ings", "ing", "edly", "ies", "ied", "es", "ed", "s")
# "es" is a plural ending only after a sibilant ("wishes" -> "wish"). Elsewhere
# the plural is a bare "s", so "devotees" must stem to "devotee", not "devote",
# or it stops matching the singular.
_SIBILANTS = ("s", "x", "z", "ch", "sh")


def fold(text: str) -> str:
    """Lowercase and strip diacritics: 'Vairãgya' -> 'vairagya'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if not word.endswith(suffix) or len(word) - len(suffix) < 3:
            continue
        root = word[: -len(suffix)]
        if suffix == "es" and not root.endswith(_SIBILANTS):
            continue          # fall through to the bare "s" rule
        return root + "y" if suffix in ("ies", "ied") else root
    return word


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    tokens = [t for t in _WORD.findall(fold(text)) if len(t) > 1]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [stem(t) for t in tokens]


@dataclass
class Posting:
    doc: int
    tf: int


class BM25Index:
    """Okapi BM25 over a list of documents, each a list of weighted fields."""

    def __init__(self, field_weights: dict[str, float] | None = None):
        self.field_weights = field_weights or {"text": 1.0}
        self.postings: dict[str, list[Posting]] = defaultdict(list)
        self.doc_len: list[float] = []
        self.doc_ids: list[str] = []
        self.avg_len = 0.0

    def add_all(self, documents: list[tuple[str, dict[str, str]]]) -> None:
        """documents: (doc_id, {field: text}) in index order."""
        for doc_id, fields in documents:
            counts: Counter[str] = Counter()
            length = 0.0
            for field, text in fields.items():
                weight = self.field_weights.get(field, 0.0)
                if not weight or not text:
                    continue
                tokens = tokenize(text)
                length += weight * len(tokens)
                for token in tokens:
                    counts[token] += weight
            index = len(self.doc_ids)
            self.doc_ids.append(doc_id)
            self.doc_len.append(length or 1.0)
            for token, tf in counts.items():
                self.postings[token].append(Posting(index, tf))
        self.avg_len = sum(self.doc_len) / max(len(self.doc_len), 1)

    def search(self, query: str, limit: int = 50,
               allowed: set[int] | None = None) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        n_docs = len(self.doc_ids)
        scores: dict[int, float] = defaultdict(float)
        covered: dict[int, set[str]] = defaultdict(set)
        rare = {t for t in set(tokens)
                if 0 < len(self.postings.get(t, [])) <= RARE_DF}
        for token in set(tokens):
            postings = self.postings.get(token)
            if not postings:
                continue
            idf = math.log(1 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for posting in postings:
                if allowed is not None and posting.doc not in allowed:
                    continue
                norm = 1 - B + B * self.doc_len[posting.doc] / self.avg_len
                scores[posting.doc] += idf * posting.tf * (K1 + 1) / (posting.tf + K1 * norm)
                if token in rare:
                    covered[posting.doc].add(token)
        if rare:
            for doc in scores:
                coverage = len(covered[doc]) / len(rare)
                scores[doc] *= 1 + RARE_COVERAGE_WEIGHT * coverage
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked[:limit]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({
                "field_weights": self.field_weights,
                "postings": dict(self.postings),
                "doc_len": self.doc_len,
                "doc_ids": self.doc_ids,
                "avg_len": self.avg_len,
            }, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        index = cls(state["field_weights"])
        index.postings = defaultdict(list, state["postings"])
        index.doc_len = state["doc_len"]
        index.doc_ids = state["doc_ids"]
        index.avg_len = state["avg_len"]
        return index
