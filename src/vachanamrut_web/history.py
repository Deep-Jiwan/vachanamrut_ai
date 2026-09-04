"""The shared question history, and the feedback left on it.

Every answered question is kept so anyone can read what has been asked before,
which for a scripture app is most of the value: the same questions recur, and a
good answer with its citations is worth more the second time it is found.

Feedback is anonymous by construction. The table has no device, voucher or
address column, only a rating, a comment and a time -- the history is public,
so anything identifying would turn it into a record of who asked and who said
what. Anyone may leave as many comments as they like.

The listing reads only the columns the index needs, so browsing does not pull
every answer and reasoning trace out of the database.
"""
from __future__ import annotations

import os
import secrets

from .db import dumps, execute, loads, now, one, query

MAX_ENTRIES = int(os.environ.get("HISTORY_MAX", "1000"))

# Reasoning traces run to tens of thousands of characters. Enough is kept to
# follow how an answer was reached, without storing an essay per question.
REASONING_CAP = 20000

RATING_MIN, RATING_MAX = 0, 5
COMMENT_MAX = 2000


def new_id() -> str:
    return secrets.token_hex(8)


def record(*, question: str, answer: str, reasoning: str, sources: list[dict],
           verified: list[dict], model: str, elapsed: float) -> str | None:
    """Save one answered question. Returns its id, or None if it was empty."""
    if not answer.strip():
        return None

    entry_id = new_id()
    citations = [s.get("citation", "") for s in sources][:6]
    execute(
        "INSERT INTO questions (id, ts, question, answer, reasoning, "
        "reasoning_truncated, sources, verified, citations, model, elapsed, "
        "quote_count, verified_count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (entry_id, now(), question, answer, reasoning[:REASONING_CAP],
         1 if len(reasoning) > REASONING_CAP else 0,
         dumps(sources), dumps(verified), dumps(citations), model,
         round(elapsed, 1), len(verified),
         sum(1 for v in verified if v.get("verbatim"))))
    _trim()
    return entry_id


def _trim() -> None:
    """Keep the table bounded, oldest first."""
    execute("DELETE FROM feedback WHERE question_id IN ("
            "SELECT id FROM questions ORDER BY ts DESC LIMIT -1 OFFSET ?)",
            (MAX_ENTRIES,))
    execute("DELETE FROM questions WHERE id IN ("
            "SELECT id FROM questions ORDER BY ts DESC LIMIT -1 OFFSET ?)",
            (MAX_ENTRIES,))


def listing(limit: int = 300) -> list[dict]:
    """Recent questions, newest first, as lightweight summaries."""
    rows = query(
        "SELECT q.id, q.ts, q.question, q.citations, q.model, q.quote_count, "
        "q.verified_count, "
        "(SELECT COUNT(*) FROM feedback f WHERE f.question_id = q.id) "
        "  AS comment_count, "
        "(SELECT ROUND(AVG(f.rating), 1) FROM feedback f "
        "  WHERE f.question_id = q.id AND f.rating IS NOT NULL) AS avg_rating "
        "FROM questions q ORDER BY q.ts DESC LIMIT ?", (max(1, limit),))
    for row in rows:
        row["citations"] = loads(row.get("citations"), [])
    return rows


def entry(entry_id: str) -> dict | None:
    if not entry_id or not entry_id.isalnum():
        return None
    row = one("SELECT * FROM questions WHERE id = ?", (entry_id,))
    if not row:
        return None
    row["sources"] = loads(row.get("sources"), [])
    row["verified"] = loads(row.get("verified"), [])
    row["citations"] = loads(row.get("citations"), [])
    row["reasoning_truncated"] = bool(row.get("reasoning_truncated"))
    row["feedback"] = feedback_for(entry_id)
    return row


def count() -> int:
    return int((one("SELECT COUNT(*) AS n FROM questions") or {}).get("n") or 0)


# ----------------------------------------------------------------- feedback

def feedback_for(entry_id: str) -> list[dict]:
    """Comments on one question, oldest first, so a thread reads in order."""
    return query("SELECT id, ts, rating, comment FROM feedback "
                 "WHERE question_id = ? ORDER BY ts ASC, id ASC", (entry_id,))


def add_feedback(entry_id: str, rating: int | None, comment: str) -> dict:
    """Attach a rating and/or a comment to a question.

    Either alone is accepted: a star rating with nothing to add is useful, and
    so is a remark from someone who does not want to score it.
    """
    if not entry_id or not entry_id.isalnum():
        return {"ok": False, "error": "Unknown question."}
    if not one("SELECT id FROM questions WHERE id = ?", (entry_id,)):
        return {"ok": False, "error": "Unknown question."}

    comment = (comment or "").strip()[:COMMENT_MAX]
    if rating is not None:
        try:
            rating = int(rating)
        except (TypeError, ValueError):
            return {"ok": False, "error": "Rating must be a whole number."}
        if not RATING_MIN <= rating <= RATING_MAX:
            return {"ok": False,
                    "error": f"Rating must be between {RATING_MIN} and {RATING_MAX}."}
    if rating is None and not comment:
        return {"ok": False, "error": "Add a rating or a comment."}

    execute("INSERT INTO feedback (question_id, ts, rating, comment) "
            "VALUES (?,?,?,?)", (entry_id, now(), rating, comment or None))
    return {"ok": True, "feedback": feedback_for(entry_id)}


# -------------------------------------------------------------------- stats

def stats() -> dict:
    """Aggregates for usage analysis, computed in SQL."""
    totals = one(
        "SELECT COUNT(*) AS questions, "
        "ROUND(AVG(elapsed), 1) AS avg_seconds, "
        "SUM(quote_count) AS quotes, SUM(verified_count) AS verbatim_quotes, "
        "MIN(ts) AS first_ts, MAX(ts) AS last_ts FROM questions") or {}
    feedback = one(
        "SELECT COUNT(*) AS entries, COUNT(rating) AS ratings, "
        "ROUND(AVG(rating), 2) AS avg_rating, "
        "COUNT(comment) AS comments FROM feedback") or {}
    return {
        "questions": totals,
        "feedback": feedback,
        "by_model": query("SELECT model, COUNT(*) AS n, ROUND(AVG(elapsed),1) "
                          "AS avg_seconds FROM questions GROUP BY model"),
        "rating_spread": query("SELECT rating, COUNT(*) AS n FROM feedback "
                               "WHERE rating IS NOT NULL GROUP BY rating "
                               "ORDER BY rating"),
        "vouchers": one("SELECT COUNT(*) AS redeemed, SUM(used) AS questions_used, "
                        "SUM(CASE WHEN used >= 10 THEN 1 ELSE 0 END) AS exhausted "
                        "FROM vouchers") or {},
        # json_each needs the JSON1 extension. It is present in modern SQLite
        # and in libSQL, but one optional breakdown is not worth failing the
        # whole endpoint over if a build lacks it.
        "top_citations": _top_citations(),
    }


def _top_citations() -> list[dict]:
    try:
        return query("SELECT value AS citation, COUNT(*) AS n FROM questions, "
                     "json_each(questions.citations) GROUP BY value "
                     "ORDER BY n DESC LIMIT 15")
    except Exception:
        return []
