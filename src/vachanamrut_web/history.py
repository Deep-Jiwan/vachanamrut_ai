"""The shared question history.

Every answered question is kept so anyone can read what has been asked before,
which for a scripture app is most of the value: the same questions recur, and a
good answer with its citations is worth more the second time it is found than
the first.

Two records are written per question. A small summary goes into a capped list,
so the index page is one Redis call and search can run over it without pulling
megabytes of reasoning. The full record -- answer, reasoning, passages,
quotation checks -- is a separate key, fetched only when a reader opens that
entry.

Nothing identifying is stored. The voucher a question was asked with is not
written to the record, because the history is public to every visitor and
linking questions to a voucher would make it a record of who asked what.
"""
from __future__ import annotations

import os
import secrets

from .store import dumps, get_store, loads, now

INDEX_KEY = "h:index"
MAX_ENTRIES = int(os.environ.get("HISTORY_MAX", "1000"))

# Reasoning traces run to tens of thousands of characters. Enough is kept to
# follow how an answer was reached, without storing an essay per question.
REASONING_CAP = 20000


def new_id() -> str:
    return secrets.token_hex(8)


def record(*, question: str, answer: str, reasoning: str, sources: list[dict],
           verified: list[dict], model: str, elapsed: float) -> str | None:
    """Save one answered question. Returns its id, or None if it was empty."""
    if not answer.strip():
        return None

    entry_id = new_id()
    store = get_store()
    citations = [s.get("citation", "") for s in sources][:6]

    full = {
        "id": entry_id,
        "ts": now(),
        "question": question,
        "answer": answer,
        "reasoning": reasoning[:REASONING_CAP],
        "reasoning_truncated": len(reasoning) > REASONING_CAP,
        "sources": sources,
        "verified": verified,
        "model": model,
        "elapsed": round(elapsed, 1),
    }
    summary = {
        "id": entry_id,
        "ts": full["ts"],
        "question": question,
        "citations": citations,
        "verified_count": sum(1 for v in verified if v.get("verbatim")),
        "quote_count": len(verified),
        "model": model,
    }

    store.set(f"h:{entry_id}", dumps(full))
    store.lpush(INDEX_KEY, dumps(summary))
    store.ltrim(INDEX_KEY, 0, MAX_ENTRIES - 1)
    return entry_id


def listing(limit: int = 300) -> list[dict]:
    """Recent questions, newest first, as lightweight summaries."""
    rows = get_store().lrange(INDEX_KEY, 0, max(0, limit - 1))
    out = []
    for raw in rows:
        parsed = loads(raw)
        if parsed:
            out.append(parsed)
    return out


def entry(entry_id: str) -> dict | None:
    if not entry_id or not entry_id.isalnum():
        return None
    return loads(get_store().get(f"h:{entry_id}"))


def count() -> int:
    return get_store().llen(INDEX_KEY)
