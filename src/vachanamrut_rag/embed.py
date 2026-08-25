"""Optional dense embeddings.

The system is built to work with no model download at all: BM25 over
paragraph text, plus reference routing, answers the great majority of
questions about this corpus, and it runs anywhere with no API key and no
network. Dense retrieval is a strict upgrade layered on top when available.

Enable a backend with the VACHANAMRUT_EMBEDDINGS environment variable:

    local:BAAI/bge-base-en-v1.5     sentence-transformers, runs offline
    voyage:voyage-3                 Voyage AI  (VOYAGE_API_KEY)
    openai:text-embedding-3-large   OpenAI     (OPENAI_API_KEY)
    gemini:text-embedding-004       Gemini     (GEMINI_API_KEY)

Backends are imported lazily so none of them is a hard dependency.
"""
from __future__ import annotations

import os
from typing import Protocol, Sequence


class Embedder(Protocol):
    dimensions: int

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        ...


class LocalEmbedder:
    """sentence-transformers. Downloads the model once, then runs offline."""

    def __init__(self, model: str = "BAAI/bge-base-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        self.dimensions = self.model.get_sentence_embedding_dimension()
        # bge models expect an instruction prefix on queries only.
        self.query_prefix = "Represent this sentence for searching relevant passages: " \
            if "bge" in model.lower() else ""

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        prepared = [self.query_prefix + t for t in texts] if is_query else list(texts)
        return self.model.encode(prepared, normalize_embeddings=True,
                                 show_progress_bar=False).tolist()


class VoyageEmbedder:
    def __init__(self, model: str = "voyage-3"):
        import voyageai

        self.client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        self.model = model
        self.dimensions = 1024

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        kind = "query" if is_query else "document"
        return self.client.embed(list(texts), model=self.model, input_type=kind).embeddings


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-large"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.dimensions = 3072

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in response.data]


class GeminiEmbedder:
    def __init__(self, model: str = "text-embedding-004"):
        from google import genai

        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model
        self.dimensions = 768

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        kind = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        result = self.client.models.embed_content(
            model=self.model, contents=list(texts),
            config={"task_type": kind},
        )
        return [e.values for e in result.embeddings]


_BACKENDS = {
    "local": LocalEmbedder,
    "voyage": VoyageEmbedder,
    "openai": OpenAIEmbedder,
    "gemini": GeminiEmbedder,
}


def load_embedder(spec: str | None = None) -> Embedder | None:
    """Build the embedder named by `spec` or VACHANAMRUT_EMBEDDINGS, else None."""
    spec = spec or os.environ.get("VACHANAMRUT_EMBEDDINGS", "").strip()
    if not spec or spec.lower() in {"none", "off", "0", "false"}:
        return None
    backend, _, model = spec.partition(":")
    factory = _BACKENDS.get(backend.lower())
    if factory is None:
        raise ValueError(
            f"unknown embedding backend {backend!r}; expected one of {sorted(_BACKENDS)}"
        )
    return factory(model) if model else factory()
