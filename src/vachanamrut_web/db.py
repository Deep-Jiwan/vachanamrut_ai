"""SQLite storage, local as a file and on a serverless host over libSQL.

Everything durable lives here: voucher balances, answered questions, and the
feedback left on them. One relational store rather than several, so usage can
be analysed with ordinary SQL instead of being reassembled from key-value
fragments.

Plain SQLite cannot be used directly on a serverless host: the filesystem is
ephemeral, so a database written by one invocation is gone when that instance
recycles. libSQL (Turso) is the same engine reached over HTTP, which keeps the
schema, the SQL and the exported file identical to the local one while actually
persisting. Configure LIBSQL_URL and LIBSQL_AUTH_TOKEN (TURSO_* names are also
accepted) and it is used; otherwise a local file is, which is right for
development and wrong in production. `describe()` says which is active, because
the failure mode of getting this wrong is silent: everything works, and the
data quietly disappears.

Counting is done in SQL rather than read-modify-write. Spending a question is
`UPDATE ... WHERE used < limit`, and the row count says whether it succeeded,
so two requests racing for a voucher's last question cannot both win.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import httpx

SCHEMA = """
CREATE TABLE IF NOT EXISTS vouchers (
    code         TEXT PRIMARY KEY,
    device       TEXT,
    used         INTEGER NOT NULL DEFAULT 0,
    claimed_at   INTEGER,
    last_used_at INTEGER
);

CREATE TABLE IF NOT EXISTS questions (
    id                  TEXT PRIMARY KEY,
    ts                  INTEGER NOT NULL,
    question            TEXT NOT NULL,
    answer              TEXT NOT NULL,
    reasoning           TEXT,
    reasoning_truncated INTEGER NOT NULL DEFAULT 0,
    sources             TEXT,
    verified            TEXT,
    citations           TEXT,
    model               TEXT,
    elapsed             REAL,
    quote_count         INTEGER NOT NULL DEFAULT 0,
    verified_count      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_questions_ts ON questions (ts DESC);

-- No device, voucher or address column, deliberately. The history is public,
-- so anything identifying here would turn it into a record of who asked what.
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id TEXT NOT NULL,
    ts          INTEGER NOT NULL,
    rating      INTEGER,
    comment     TEXT,
    FOREIGN KEY (question_id) REFERENCES questions (id)
);

CREATE INDEX IF NOT EXISTS idx_feedback_question ON feedback (question_id, ts);
"""


def now() -> int:
    return int(time.time())


class SqliteBackend:
    """A real SQLite file. Durable wherever the filesystem is."""

    kind = "sqlite-file"

    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._ready = False

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def ensure_schema(self) -> None:
        if self._ready:
            return
        with self._init_lock:
            self._conn().executescript(SCHEMA)
            self._conn().commit()
            self._ready = True

    def execute(self, sql: str, args: Sequence[Any] = ()) -> tuple[list[dict], int]:
        self.ensure_schema()
        conn = self._conn()
        cur = conn.execute(sql, tuple(args))
        rows = [dict(zip([c[0] for c in cur.description], r))
                for r in cur.fetchall()] if cur.description else []
        conn.commit()
        return rows, cur.rowcount


class LibsqlBackend:
    """libSQL over HTTP: the same engine, reachable from a stateless function."""

    kind = "libsql"

    def __init__(self, url: str, token: str) -> None:
        # Turso hands out libsql:// URLs; the HTTP endpoint is the same host.
        if url.startswith("libsql://"):
            url = "https://" + url[len("libsql://"):]
        self.url = url.rstrip("/")
        self.token = token
        self._client = httpx.Client(timeout=httpx.Timeout(15.0, connect=8.0))
        self._ready = False
        self._init_lock = threading.Lock()

    @staticmethod
    def _encode(value: Any) -> dict:
        if value is None:
            return {"type": "null", "value": None}
        if isinstance(value, bool):
            return {"type": "integer", "value": str(int(value))}
        if isinstance(value, int):
            return {"type": "integer", "value": str(value)}
        if isinstance(value, float):
            return {"type": "float", "value": value}
        return {"type": "text", "value": str(value)}

    @staticmethod
    def _decode(cell: dict) -> Any:
        kind, value = cell.get("type"), cell.get("value")
        if kind == "null":
            return None
        if kind == "integer":
            return int(value)
        if kind == "float":
            return float(value)
        return value

    def _pipeline(self, statements: list[tuple[str, Sequence[Any]]]) -> list[dict]:
        requests = [{"type": "execute",
                     "stmt": {"sql": sql, "args": [self._encode(a) for a in args]}}
                    for sql, args in statements]
        requests.append({"type": "close"})
        response = self._client.post(
            f"{self.url}/v2/pipeline",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"requests": requests},
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        for entry in results:
            if entry.get("type") == "error":
                raise RuntimeError(
                    f"libSQL error: {entry.get('error', {}).get('message')}")
        return results

    def ensure_schema(self) -> None:
        if self._ready:
            return
        with self._init_lock:
            statements = [(s.strip(), ()) for s in SCHEMA.split(";") if s.strip()]
            self._pipeline(statements)
            self._ready = True

    def execute(self, sql: str, args: Sequence[Any] = ()) -> tuple[list[dict], int]:
        self.ensure_schema()
        results = self._pipeline([(sql, args)])
        payload = results[0].get("response", {}).get("result", {})
        cols = [c.get("name") for c in payload.get("cols", [])]
        rows = [dict(zip(cols, [self._decode(cell) for cell in row]))
                for row in payload.get("rows", [])]
        return rows, int(payload.get("affected_row_count") or 0)


Backend = SqliteBackend | LibsqlBackend
_backend: Backend | None = None


def get_backend() -> Backend:
    global _backend
    if _backend is not None:
        return _backend
    url = (os.environ.get("LIBSQL_URL")
           or os.environ.get("TURSO_DATABASE_URL") or "").strip()
    token = (os.environ.get("LIBSQL_AUTH_TOKEN")
             or os.environ.get("TURSO_AUTH_TOKEN") or "").strip()
    if url and token:
        _backend = LibsqlBackend(url, token)
    else:
        # On a serverless host only /tmp is writable, and it does not survive
        # the instance. Local development gets a real file next to the project.
        default = "/tmp/vachanamrut.db" if os.environ.get("VERCEL") \
            else str(Path(__file__).resolve().parents[2] / "data" / "vachanamrut.db")
        _backend = SqliteBackend(os.environ.get("SQLITE_PATH", default))
    return _backend


def execute(sql: str, args: Sequence[Any] = ()) -> tuple[list[dict], int]:
    return get_backend().execute(sql, args)


def query(sql: str, args: Sequence[Any] = ()) -> list[dict]:
    return execute(sql, args)[0]


def one(sql: str, args: Sequence[Any] = ()) -> dict | None:
    rows = query(sql, args)
    return rows[0] if rows else None


def describe() -> dict:
    backend = get_backend()
    ephemeral = (isinstance(backend, SqliteBackend)
                 and bool(os.environ.get("VERCEL")))
    return {
        "kind": backend.kind,
        "durable": not ephemeral,
        "location": getattr(backend, "path", getattr(backend, "url", "")),
        "note": ("Writing to the serverless host's temporary disk: this is lost "
                 "when the instance recycles. Set LIBSQL_URL and "
                 "LIBSQL_AUTH_TOKEN for durable storage."
                 if ephemeral else "Durable."),
    }


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def loads(raw: Any, default: Any = None) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default
