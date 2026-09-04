"""Durable key-value storage, with an in-memory stand-in for local use.

A serverless function keeps nothing between requests, so voucher balances,
device bindings and the shared history all need somewhere to live. Redis is
that place in production, reached over Upstash's REST API so no driver is
needed and a cold start costs one HTTPS round trip rather than a connection
pool.

The counting operations are the reason this is Redis rather than a JSON blob:
two people spending the last question on a voucher at the same moment must not
both succeed, so the balance moves with INCR and the device binding is claimed
with SET NX. Both are atomic on the server; read-modify-write from a stateless
function is not.

Locally, with no credentials configured, MemoryStore keeps the same interface
so the whole app runs and can be tested without provisioning anything. It
forgets everything when the process exits, which is fine for development and
would be wrong in production -- `Store.describe()` says which one is active so
that is visible rather than silent.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import httpx


class MemoryStore:
    """Process-local stand-in. Correct for one process, lost on restart."""

    kind = "memory"

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lists: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._data[key] = value

    def setnx(self, key: str, value: str) -> bool:
        with self._lock:
            if key in self._data:
                return False
            self._data[key] = value
            return True

    def incr(self, key: str) -> int:
        with self._lock:
            value = int(self._data.get(key, 0)) + 1
            self._data[key] = str(value)
            return value

    def decr(self, key: str) -> int:
        with self._lock:
            value = int(self._data.get(key, 0)) - 1
            self._data[key] = str(value)
            return value

    def lpush(self, key: str, value: str) -> None:
        with self._lock:
            self._lists.setdefault(key, []).insert(0, value)

    def lrange(self, key: str, start: int, stop: int) -> list[str]:
        with self._lock:
            items = self._lists.get(key, [])
            return items[start:] if stop < 0 else items[start:stop + 1]

    def ltrim(self, key: str, start: int, stop: int) -> None:
        with self._lock:
            items = self._lists.get(key, [])
            self._lists[key] = items[start:] if stop < 0 else items[start:stop + 1]

    def llen(self, key: str) -> int:
        with self._lock:
            return len(self._lists.get(key, []))

    def mget(self, keys: list[str]) -> list[str | None]:
        with self._lock:
            return [self._data.get(k) for k in keys]


class RedisStore:
    """Upstash Redis over its REST API.

    Commands are sent as JSON arrays, which keeps values with slashes, spaces
    or newlines intact -- the URL-path form of the API does not.
    """

    kind = "redis"

    def __init__(self, url: str, token: str) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self._client = httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0))

    def _command(self, *args: Any) -> Any:
        response = self._client.post(
            self.url,
            headers={"Authorization": f"Bearer {self.token}"},
            json=[str(a) for a in args],
        )
        response.raise_for_status()
        return response.json().get("result")

    def _pipeline(self, commands: list[list[Any]]) -> list[Any]:
        response = self._client.post(
            f"{self.url}/pipeline",
            headers={"Authorization": f"Bearer {self.token}"},
            json=[[str(a) for a in cmd] for cmd in commands],
        )
        response.raise_for_status()
        return [row.get("result") for row in response.json()]

    def get(self, key: str) -> str | None:
        return self._command("GET", key)

    def set(self, key: str, value: str) -> None:
        self._command("SET", key, value)

    def setnx(self, key: str, value: str) -> bool:
        return bool(self._command("SETNX", key, value))

    def incr(self, key: str) -> int:
        return int(self._command("INCR", key))

    def decr(self, key: str) -> int:
        return int(self._command("DECR", key))

    def lpush(self, key: str, value: str) -> None:
        self._command("LPUSH", key, value)

    def lrange(self, key: str, start: int, stop: int) -> list[str]:
        return self._command("LRANGE", key, start, stop) or []

    def ltrim(self, key: str, start: int, stop: int) -> None:
        self._command("LTRIM", key, start, stop)

    def llen(self, key: str) -> int:
        return int(self._command("LLEN", key) or 0)

    def mget(self, keys: list[str]) -> list[str | None]:
        if not keys:
            return []
        return self._command("MGET", *keys) or []


Store = MemoryStore | RedisStore
_store: Store | None = None


def get_store() -> Store:
    """Redis when credentials are configured, otherwise the memory stand-in.

    Both Vercel KV's variable names and Upstash's own are accepted, since which
    pair appears depends on how the database was attached to the project.
    """
    global _store
    if _store is not None:
        return _store
    url = (os.environ.get("KV_REST_API_URL")
           or os.environ.get("UPSTASH_REDIS_REST_URL") or "").strip()
    token = (os.environ.get("KV_REST_API_TOKEN")
             or os.environ.get("UPSTASH_REDIS_REST_TOKEN") or "").strip()
    _store = RedisStore(url, token) if url and token else MemoryStore()
    return _store


def describe() -> dict:
    store = get_store()
    return {
        "kind": store.kind,
        "durable": store.kind != "memory",
        "note": ("In-memory: balances and history reset whenever the server "
                 "restarts. Attach a KV database for durable storage."
                 if store.kind == "memory" else "Durable."),
    }


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def loads(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def now() -> int:
    return int(time.time())
