"""Web API: retrieval-grounded answers over the Vachanamrut, streamed from DeepSeek.

The web app is the same idea as the MCP server, aimed at a browser instead of a
model client: retrieve verbatim paragraphs first, hand them to the model as the
only permitted source, and stream the reasoning so a reader can watch the answer
being built out of the text rather than trust it blindly.

Every quotation the model produces is checked back against the corpus afterwards
by `verify_quote`, so a paraphrase presented as scripture surfaces in the UI as
unverified rather than passing as a citation.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vachanamrut_rag.retrieve import Retriever  # noqa: E402

CORPUS_DIR = os.environ.get("VACHANAMRUT_CORPUS", str(ROOT / "data" / "corpus"))
DEEPSEEK_URL = os.environ.get(
    "DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions"
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-reasoner")

# Retrieval is cheap (3ms) but building the index costs ~0.5s, so it is built
# once per warm container and reused.
_retriever: Retriever | None = None


def retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever.load(CORPUS_DIR)
    return _retriever


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


@app.post("/api/ask")
@app.post("/ask")
async def ask(body: Ask):
    return StreamingResponse(
        stream_answer(body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform",
                 "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
@app.get("/health")
async def health():
    try:
        r = retriever()
        return {"ok": True,
                "chunks": len(r.corpus.chunks),
                "discourses": len(r.corpus.discourses),
                "model": MODEL,
                "key_configured": bool(os.environ.get("DEEPSEEK_API_KEY"))}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


# Local development only. On Vercel the static files are served by the CDN and
# never reach this function, so `public/` is absent from the bundle and this
# mount is skipped.
_PUBLIC = ROOT / "public"
if _PUBLIC.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_PUBLIC), html=True), name="static")
