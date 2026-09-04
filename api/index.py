"""Web API: retrieval-grounded answers over the Vachanamrut, streamed from DeepSeek.

The web app is the same idea as the MCP server, aimed at a browser instead of a
model client: retrieve verbatim paragraphs first, hand them to the model as its
only permitted source, and stream the reasoning so a reader watches the answer
being built out of the text rather than trusting it blindly.

Every quotation the model produces is checked back against the corpus afterwards
by `verify_quote`, so a paraphrase presented as scripture surfaces in the UI as
unverified rather than passing as a citation.

The corpus import is deliberately not allowed to fail at module scope. On a
serverless host a raised ImportError becomes an opaque FUNCTION_INVOCATION_FAILED
with no way to tell a bundling mistake from a code bug, so the error is captured
and reported by /api/health instead.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

Retriever = None
IMPORT_ERROR: str | None = None
try:
    from vachanamrut_rag.retrieve import Retriever  # type: ignore[no-redef]
except Exception:                                   # bundling or dependency fault
    IMPORT_ERROR = traceback.format_exc()

CORPUS_DIR = os.environ.get("VACHANAMRUT_CORPUS", str(ROOT / "data" / "corpus"))
DEEPSEEK_URL = os.environ.get(
    "DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-reasoner")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

# Retrieval is cheap (3ms) but building the index costs ~0.5s, so it is built
# once per warm container and reused.
_retriever = None


def retriever():
    global _retriever
    if _retriever is None:
        if Retriever is None:
            raise RuntimeError("corpus package failed to import; see /api/health")
        _retriever = Retriever.load(CORPUS_DIR)
    return _retriever


def password_ok(supplied: str | None) -> bool:
    """Constant-time check. An unset APP_PASSWORD leaves the app open."""
    if not APP_PASSWORD:
        return True
    return bool(supplied) and hmac.compare_digest(supplied, APP_PASSWORD)


SYSTEM_PROMPT = """You answer questions about the Vachanamrut: 273 discourses of
Bhagwan Swaminarayan (1819-1829), in the official English translation published
by Swaminarayan Aksharpith.

You are given passages retrieved from the scripture. They are verbatim. They are
your only permitted source.

Rules:
1. Ground every claim in the passages provided. Never answer from memory, and
   never reconstruct a passage you were not given.
2. Quote exactly. Copy the wording character-for-character from a passage, with
   its diacritics. Never paraphrase inside quotation marks.
3. Cite in the tradition's form, immediately after the quotation: Gadhada I-37.5
   -- section, discourse number, paragraph number. The citation for each passage
   is printed in brackets above it; use that, unchanged.
4. If the passages do not answer the question, say that the retrieved passages do
   not appear to address it. Do not fill the gap with general knowledge of
   Hinduism or of Swaminarayan theology.
5. Answer in English, plainly. Explain the passage in your own words around the
   quotation, but leave the quotation itself untouched.

The reader sees your citations rendered beside the source text, so a wrong
citation is immediately visible. Take care to attach the right one.
"""


def build_context(results: list) -> str:
    """Render retrieved paragraphs as the model's source block."""
    blocks: list[str] = []
    for r in results:
        c = r.chunk
        header = f"[{c.citation}]"
        if c.title:
            header += f" -- {c.discourse_citation}: {c.title}"
        if c.speaker:
            header += f" (speaker: {c.speaker})"
        body = [header, c.text]
        for neighbour in r.context:
            body.append(f"    [context {neighbour.citation}] {neighbour.text}")
        blocks.append("\n".join(body))
    return "\n\n---\n\n".join(blocks)


# Quotations long enough to be worth verifying. Short fragments ("God", "the
# mind") appear everywhere and would make the check meaningless.
_QUOTED = re.compile("[“\"]([^“”\"]{25,})[”\"]")

# A quotation may legitimately elide its middle ("... and so on"). Each side of
# an elision is checked separately, and the quote counts as verbatim only if
# every segment is found in the same paragraph.
_ELLIPSIS = re.compile(r"\s*(?:\.\s*\.\s*\.|…)\s*")


def verify_answer(answer: str) -> list[dict]:
    """Check each quotation in the answer against the corpus."""
    checked: list[dict] = []
    seen: set[str] = set()
    for match in _QUOTED.finditer(answer):
        quote = " ".join(match.group(1).split())
        if quote in seen:
            continue
        seen.add(quote)

        segments = [s for s in (p.strip() for p in _ELLIPSIS.split(quote)) if len(s) > 12]
        results = [retriever().verify_quote(s) for s in segments] or \
                  [retriever().verify_quote(quote)]
        citations = {r.get("citation") for r in results if r["verbatim"]}
        verbatim = all(r["verbatim"] for r in results) and len(citations) == 1

        checked.append({
            "quote": quote[:160] + ("…" if len(quote) > 160 else ""),
            "verbatim": verbatim,
            "elided": len(segments) > 1,
            "citation": next(iter(citations)) if len(citations) == 1 else None,
        })
    return checked


app = FastAPI(title="Vachanamrut", docs_url=None, redoc_url=None)


class Ask(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    limit: int = Field(default=8, ge=1, le=20)
    section: str | None = None


class Login(BaseModel):
    password: str = Field(default="", max_length=200)


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def stream_answer(ask: Ask) -> AsyncIterator[str]:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        yield sse({"type": "error",
                   "message": "DEEPSEEK_API_KEY is not set on the server."})
        return

    try:
        results = retriever().search(
            ask.question, limit=ask.limit, section=ask.section, context=1)
    except Exception as exc:                     # corpus missing or unreadable
        yield sse({"type": "error", "message": f"Retrieval failed: {exc}"})
        return

    yield sse({
        "type": "sources",
        "sources": [{
            "citation": r.chunk.citation,
            "discourse": r.chunk.discourse_citation,
            "title": r.chunk.title,
            "speaker": r.chunk.speaker,
            "text": r.chunk.text,
            "matched_by": r.reason,
        } for r in results],
    })

    if not results:
        yield sse({"type": "answer",
                   "text": "Nothing in the Vachanamrut matched that question."})
        yield sse({"type": "done"})
        return

    payload = {
        "model": MODEL,
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                "Passages retrieved from the Vachanamrut:\n\n"
                f"{build_context(results)}\n\n---\n\nQuestion: {ask.question}"},
        ],
    }

    answer_parts: list[str] = []
    usage: dict | None = None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=15.0)) as client:
            async with client.stream(
                "POST", DEEPSEEK_URL,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json=payload,
            ) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode("utf-8", "replace")[:400]
                    yield sse({"type": "error",
                               "message": f"DeepSeek returned {response.status_code}: {detail}"})
                    return
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if parsed.get("usage"):
                        usage = parsed["usage"]
                    for choice in parsed.get("choices", []):
                        delta = choice.get("delta") or {}
                        if delta.get("reasoning_content"):
                            yield sse({"type": "reasoning",
                                       "text": delta["reasoning_content"]})
                        if delta.get("content"):
                            answer_parts.append(delta["content"])
                            yield sse({"type": "answer", "text": delta["content"]})
    except httpx.HTTPError as exc:
        yield sse({"type": "error", "message": f"Could not reach DeepSeek: {exc}"})
        return

    answer = "".join(answer_parts)
    if answer:
        yield sse({"type": "verified", "quotes": verify_answer(answer)})
    yield sse({"type": "done", "usage": usage})


@app.post("/api/login")
@app.post("/login")
async def login(body: Login):
    if not password_ok(body.password):
        return JSONResponse({"ok": False, "error": "Wrong password."}, status_code=401)
    return {"ok": True, "required": bool(APP_PASSWORD)}


@app.post("/api/ask")
@app.post("/ask")
async def ask(body: Ask, x_app_password: str | None = Header(default=None)):
    if not password_ok(x_app_password):
        return JSONResponse({"ok": False, "error": "Password required."}, status_code=401)
    return StreamingResponse(
        stream_answer(body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform",
                 "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
@app.get("/health")
async def health():
    """Reports enough to tell a bundling fault from a configuration one."""
    info: dict = {
        "import_ok": Retriever is not None,
        "password_required": bool(APP_PASSWORD),
        "key_configured": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "model": MODEL,
        "root": str(ROOT),
        "corpus_dir": CORPUS_DIR,
        "corpus_dir_exists": Path(CORPUS_DIR).is_dir(),
        "root_entries": sorted(p.name for p in ROOT.iterdir())[:40]
        if ROOT.is_dir() else [],
    }
    if IMPORT_ERROR:
        info["ok"] = False
        info["import_error"] = IMPORT_ERROR.strip().splitlines()[-6:]
        return JSONResponse(info, status_code=500)
    try:
        r = retriever()
        info |= {"ok": True, "chunks": len(r.corpus.chunks),
                 "discourses": len(r.corpus.discourses)}
        return info
    except Exception:
        info["ok"] = False
        info["load_error"] = traceback.format_exc().strip().splitlines()[-6:]
        return JSONResponse(info, status_code=500)


# Local development only. On Vercel the static files are served by the CDN and
# never reach this function.
_PUBLIC = ROOT / "public"
if _PUBLIC.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_PUBLIC), html=True), name="static")
