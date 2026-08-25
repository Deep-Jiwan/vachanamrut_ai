"""Hybrid retrieval over the Vachanamrut.

Three signals are combined, because questions about this text arrive in three
different shapes:

  1. Reference routing. Satsang study questions almost always name their
     discourse ("GADHADA PRATHAM 35: ... 3. Towards whom does Shriji Maharaj
     not develop a liking?"). When the query names a discourse, that discourse
     is the answer's home and its paragraphs are ranked first — no amount of
     semantic similarity should send the reader somewhere else.
  2. Lexical BM25 over paragraph text, with diacritics folded so "vairagya"
     finds "vairãgya".
  3. Dense similarity, when an embedding backend is configured.

Lexical and dense rankings are fused with Reciprocal Rank Fusion, which needs
no score calibration between the two and degrades gracefully when only one is
present.

Results are returned as paragraphs with their citation and, optionally, their
neighbouring paragraphs — a reply is usually unusable without the question it
answers, which sits in the paragraph before it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .corpus import Chunk, Corpus, DEFAULT_CORPUS_DIR
from .embed import Embedder, load_embedder
from .ids import Reference, find_references
from .index import BM25Index, FIELD_WEIGHTS, fold

RRF_K = 60          # standard Reciprocal Rank Fusion damping
REFERENCE_BOOST = 3.0


@dataclass
class Result:
    chunk: Chunk
    score: float
    context: list[Chunk] = field(default_factory=list)
    reason: str = ""

    def to_dict(self, include_context: bool = True) -> dict:
        out = {
            "citation": self.chunk.citation,
            "text": self.chunk.text,
            "kind": self.chunk.kind,
            "score": round(self.score, 4),
        }
        if self.chunk.kind == "paragraph":
            out |= {
                "discourse": self.chunk.discourse_citation,
                "title": self.chunk.title,
                "section": self.chunk.section,
                "paragraph": self.chunk.paragraph,
                "speaker": self.chunk.speaker,
                "pages": self.chunk.pages,
            }
        if self.chunk.see_also:
            out["see_also"] = self.chunk.see_also
        if self.reason:
            out["matched_by"] = self.reason
        if include_context and self.context:
            out["context"] = [
                {"citation": c.citation, "text": c.text, "speaker": c.speaker}
                for c in self.context
            ]
        return out


def _rrf(rankings: list[list[int]], weights: list[float]) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc in enumerate(ranking):
            fused[doc] = fused.get(doc, 0.0) + weight / (RRF_K + rank + 1)
    return fused


class Retriever:
    """Loads the corpus and its indexes, and answers queries."""

    def __init__(self, corpus: Corpus, bm25: BM25Index,
                 embedder: Embedder | None = None,
                 vectors: list[list[float]] | None = None):
        self.corpus = corpus
        self.bm25 = bm25
        self.embedder = embedder
        self.vectors = vectors
        self.positions = {cid: i for i, cid in enumerate(bm25.doc_ids)}

    @classmethod
    def load(cls, directory: str | Path = DEFAULT_CORPUS_DIR,
             embedder: Embedder | None = None) -> "Retriever":
        directory = Path(directory)
        corpus = Corpus.load(directory)
        # Building the index takes under a second, which is quicker than reading
        # a persisted one, so it is never written to disk.
        bm25 = BM25Index(FIELD_WEIGHTS)
        bm25.add_all([(c.id, {"text": c.text, "title": c.title, "term": c.term})
                      for c in corpus.chunks])
        embedder = embedder if embedder is not None else load_embedder()
        vectors = None
        vector_path = directory / "vectors.json"
        if embedder is not None and vector_path.exists():
            payload = json.loads(vector_path.read_text(encoding="utf-8"))
            if payload.get("dimensions") == embedder.dimensions:
                vectors = payload["vectors"]
        return cls(corpus, bm25, embedder, vectors)

    # ---------------------------------------------------------------- lookup

    def lookup(self, query: str) -> dict | None:
        """Return a whole discourse named by `query` ('Gadhada Pratham 35')."""
        references = find_references(query)
        if not references:
            return None
        return self.corpus.discourse(references[0].slug)

    def paragraph(self, query: str) -> dict | None:
        """Return one paragraph named by `query` ('Gadhada I-35.4')."""
        references = find_references(query)
        if not references or references[0].paragraph is None:
            return None
        ref = references[0]
        para = self.corpus.paragraph(ref.slug, ref.paragraph)
        if para is None:
            return None
        discourse = self.corpus.discourse(ref.slug)
        return {
            "citation": f"{discourse['citation']}.{ref.paragraph}",
            "discourse": discourse["citation"],
            "title": discourse["title"],
            "text": para["text"],
            "speaker": para.get("speaker"),
            "pages": para.get("pages", []),
        }

    def verify_quote(self, quote: str, reference: str | None = None) -> dict:
        """Check whether `quote` appears verbatim in the corpus."""
        needle = " ".join(fold(quote).split())
        if not needle:
            return {"verbatim": False, "reason": "empty quote"}
        candidates = self.corpus.chunks
        if reference:
            refs = find_references(reference)
            if refs:
                candidates = [c for c in candidates if c.discourse == refs[0].slug]
        for chunk in candidates:
            if needle in " ".join(fold(chunk.text).split()):
                return {"verbatim": True, "citation": chunk.citation,
                        "source_text": chunk.text}
        return {"verbatim": False,
                "reason": "not found in the Vachanamrut as written",
                "searched": reference or "whole corpus"}

    # ---------------------------------------------------------------- search

    def search(self, query: str, limit: int = 8, *, kinds: tuple[str, ...] = ("paragraph",),
               section: str | None = None, context: int = 1,
               restrict_to_reference: bool = True) -> list[Result]:
        references = find_references(query) if restrict_to_reference else []
        allowed = self._allowed(kinds, section)

        # A named discourse anchors the search inside it.
        anchored: set[int] = set()
        for ref in references:
            for position, chunk_id in enumerate(self.bm25.doc_ids):
                chunk = self.corpus.by_id[chunk_id]
                if chunk.discourse == ref.slug and position in allowed:
                    anchored.add(position)

        lexical = [doc for doc, _ in self.bm25.search(query, limit=200, allowed=allowed)]
        rankings, weights = [lexical], [1.0]

        dense = self._dense(query, allowed, limit=200)
        if dense:
            rankings.append(dense)
            weights.append(1.0)

        fused = _rrf(rankings, weights)
        for doc in anchored:
            fused[doc] = fused.get(doc, 0.0) + REFERENCE_BOOST / (RRF_K + 1)

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:limit]
        results: list[Result] = []
        for doc, score in ordered:
            chunk = self.corpus.by_id[self.bm25.doc_ids[doc]]
            reason = "named reference" if doc in anchored else (
                "lexical+dense" if dense else "lexical")
            results.append(Result(
                chunk=chunk,
                score=score,
                context=self.corpus.neighbours(chunk, context, context) if context else [],
                reason=reason,
            ))
        return results

    # --------------------------------------------------------------- helpers

    def _allowed(self, kinds: tuple[str, ...], section: str | None) -> set[int]:
        allowed: set[int] = set()
        for position, chunk_id in enumerate(self.bm25.doc_ids):
            chunk = self.corpus.by_id[chunk_id]
            if kinds and chunk.kind not in kinds:
                continue
            if section and chunk.section != section:
                continue
            allowed.add(position)
        return allowed

    def _dense(self, query: str, allowed: set[int], limit: int) -> list[int]:
        if not (self.embedder and self.vectors):
            return []
        vector = self.embedder.encode([query], is_query=True)[0]
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        scored: list[tuple[int, float]] = []
        for doc in allowed:
            candidate = self.vectors[doc]
            dot = sum(a * b for a, b in zip(vector, candidate))
            scored.append((doc, dot / norm))
        scored.sort(key=lambda kv: -kv[1])
        return [doc for doc, _ in scored[:limit]]
