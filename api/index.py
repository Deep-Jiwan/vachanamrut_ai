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
import time
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

# The reasoning model's latency is variable and not a clean function of prompt
# size: measured on one question, first answer token arrived at 10.9s with four
# passages, 45.3s with five and 22.6s with six. Since the host kills the
# function at a fixed ceiling, a slow run streams its whole reasoning and then
# dies before saying anything -- the reader watches it think and gets no answer.
# So the reasoning model gets a budget, and if it has not begun answering by
# then the same passages go to the fast model, which answers in a few seconds.
FALLBACK_MODEL = os.environ.get("DEEPSEEK_FALLBACK_MODEL", "deepseek-chat")
ANSWER_DEADLINE = float(os.environ.get("ANSWER_DEADLINE_S", "38"))

# Eight passages with their neighbours made a ~10.7k-character prompt that the
# model deliberated over exhaustively. Five keeps the neighbouring paragraphs,
# which an answer usually needs to make sense, at a third of the reasoning cost.
DEFAULT_LIMIT = 5

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
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=20)
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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            "Passages retrieved from the Vachanamrut:\n\n"
            f"{build_context(results)}\n\n---\n\nQuestion: {ask.question}"},
    ]

    answer_parts: list[str] = []
    started = time.monotonic()
    state: dict = {"usage": None, "error": None, "timed_out": False}

    async def run(client, model: str, deadline: float | None):
        """Stream one completion, yielding SSE events as they arrive.

        `deadline` abandons a run that is still reasoning with no answer begun,
        so the caller can fall back to a faster model while time remains.
        """
        async with client.stream(
            "POST", DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json={"model": model, "stream": True, "messages": messages},
        ) as response:
            if response.status_code != 200:
                detail = (await response.aread()).decode("utf-8", "replace")[:400]
                state["error"] = f"DeepSeek returned {response.status_code}: {detail}"
                return
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError:
                        parsed = None
                    if parsed:
                        if parsed.get("usage"):
                            state["usage"] = parsed["usage"]
                        for choice in parsed.get("choices", []):
                            delta = choice.get("delta") or {}
                            if delta.get("reasoning_content"):
                                yield sse({"type": "reasoning",
                                           "text": delta["reasoning_content"]})
                            if delta.get("content"):
                                answer_parts.append(delta["content"])
                                yield sse({"type": "answer",
                                           "text": delta["content"]})
                if (deadline is not None and not answer_parts
                        and time.monotonic() - started > deadline):
                    state["timed_out"] = True
                    return

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=15.0)) as client:
            async for event in run(client, MODEL, ANSWER_DEADLINE):
                yield event
            if state["error"]:
                yield sse({"type": "error", "message": state["error"]})
                return
            if state["timed_out"] and not answer_parts:
                yield sse({"type": "notice", "message":
                           f"The reasoning model was still thinking after "
                           f"{int(ANSWER_DEADLINE)}s, so the answer comes from "
                           f"{FALLBACK_MODEL}, using the same passages."})
                async for event in run(client, FALLBACK_MODEL, None):
                    yield event
                if state["error"]:
                    yield sse({"type": "error", "message": state["error"]})
                    return
    except httpx.HTTPError as exc:
        yield sse({"type": "error", "message": f"Could not reach DeepSeek: {exc}"})
        return

    usage = state["usage"]
    answer = "".join(answer_parts)
    if answer:
        yield sse({"type": "verified", "quotes": verify_answer(answer)})
    else:
        yield sse({"type": "error",
                   "message": "The model produced reasoning but no answer."})
    yield sse({"type": "done", "usage": usage})


async def login(body: Login):
    if not password_ok(body.password):
        return JSONResponse({"ok": False, "error": "Wrong password."}, status_code=401)
    return {"ok": True, "required": bool(APP_PASSWORD)}


async def ask(body: Ask, supplied_password: str | None):
    if not password_ok(supplied_password):
        return JSONResponse({"ok": False, "error": "Password required."}, status_code=401)
    return StreamingResponse(
        stream_answer(body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform",
                 "X-Accel-Buffering": "no"},
    )


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


def resolve_action(request: Request, rest: str) -> str:
    """Work out which endpoint was asked for.

    Vercel's rewrite replaces the path the function sees: every /api/* request
    arrives as /api/index, so the path alone cannot say whether this is an ask,
    a login or a health check. The rewrite therefore carries the original
    segment in ?action=, and that wins when present. Falling back to the last
    path segment keeps the same code serving plain /api/ask locally, where
    there is no rewrite.
    """
    return (request.query_params.get("action") or rest.strip("/").rsplit("/", 1)[-1]).lower()


@app.api_route("/api/{rest:path}", methods=["GET", "POST"])
async def dispatch(request: Request, rest: str):
    action = resolve_action(request, rest)

    if action == "health":
        return await health()

    if action == "login":
        try:
            body = Login(**(await request.json()))
        except Exception:
            body = Login(password="")
        return await login(body)

    if action == "ask":
        try:
            body = Ask(**(await request.json()))
        except Exception as exc:
            return JSONResponse({"ok": False, "error": f"Bad request: {exc}"},
                                status_code=422)
        return await ask(body, request.headers.get("x-app-password"))

    return JSONResponse({
        "error": "No such endpoint.",
        "action": action,
        "received_path": request.url.path,
        "available": ["ask", "login", "health"],
    }, status_code=404)


# Local development only. On Vercel the CDN serves the static files, and
# mounting them here would shadow the API routes.
_PUBLIC = ROOT / "public"
if _PUBLIC.is_dir() and not os.environ.get("VERCEL"):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_PUBLIC), html=True), name="static")
