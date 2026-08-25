"""Build dense vectors, when an embedding backend is configured.

The BM25 index is not built here: it takes under a second to construct from
the corpus, so the retriever builds it in memory at startup rather than
carrying a derived file in the repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vachanamrut_rag.corpus import Corpus  # noqa: E402
from vachanamrut_rag.embed import load_embedder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    args = ap.parse_args()
    directory = Path(args.corpus)

    corpus = Corpus.load(directory)
    embedder = load_embedder()
    if embedder is None:
        print("Embeddings: disabled (set VACHANAMRUT_EMBEDDINGS to enable)")
        return

    texts = [f"{c.title}\n{c.text}" if c.title else c.text for c in corpus.chunks]
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 64):
        vectors.extend(embedder.encode(texts[start:start + 64]))
        print(f"  embedded {min(start + 64, len(texts))}/{len(texts)}", flush=True)
    (directory / "vectors.json").write_text(
        json.dumps({"dimensions": embedder.dimensions, "vectors": vectors}),
        encoding="utf-8")
    print(f"Embeddings: {len(vectors)} vectors of {embedder.dimensions} dims")


if __name__ == "__main__":
    main()
