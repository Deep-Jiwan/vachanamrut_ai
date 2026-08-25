import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from vachanamrut_rag.corpus import Corpus  # noqa: E402
from vachanamrut_rag.retrieve import Retriever  # noqa: E402

CORPUS_DIR = ROOT / "data" / "corpus"


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return Corpus.load(CORPUS_DIR)


@pytest.fixture(scope="session")
def retriever() -> Retriever:
    return Retriever.load(CORPUS_DIR)
