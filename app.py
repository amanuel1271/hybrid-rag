"""HTTP API: ingest a docs folder at startup, then /search and /ask over it."""
import asyncio
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import asyncpg
import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from pydantic import BaseModel, Field

import rag

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:55433/rag")
DOCS_DIR = Path(os.environ.get("DOCS_DIR", Path(__file__).parent / "docs"))
# API key -> extra groups, e.g. API_KEYS='{"key-for-eng-team": ["engineering"]}'. Groups come from the
# server's key map, never from the request, so a caller can't claim a group it wasn't given.
API_KEYS: dict[str, list[str]] = json.loads(os.environ.get("API_KEYS", "{}"))

log = logging.getLogger("hybrid-rag")
logging.basicConfig(level=logging.INFO, format="%(message)s")
ANSWERS = Counter("rag_answers_total", "Answers returned by /ask", ["engine", "verified"])
QUARANTINED = Counter("rag_quarantined_passages_total", "Retrieved passages dropped as prompt-injection attempts")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # iterative_scan: when the group filter drops HNSW candidates, keep scanning instead of returning
    # fewer than LIMIT rows (pgvector >= 0.8); strict_order keeps the distance ordering exact.
    app.state.pool = await asyncpg.create_pool(DATABASE_URL, server_settings={"hnsw.iterative_scan": "strict_order"})
    async with app.state.pool.acquire() as conn:
        await conn.execute((Path(__file__).parent / "schema.sql").read_text())
        app.state.ingested = await rag.ingest(conn, DOCS_DIR)
    yield
    await app.state.pool.close()


app = FastAPI(title="hybrid-rag", lifespan=lifespan)


def caller_groups(x_api_key: str | None = Header(None)) -> list[str]:
    """No key: public docs only. Unknown key: 401, not a silent downgrade, so a typo'd key is noticed."""
    if x_api_key is None:
        return ["everyone"]
    for key, groups in API_KEYS.items():
        if secrets.compare_digest(key, x_api_key):
            return ["everyone", *groups]
    raise HTTPException(401, "unknown API key")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/health")
async def health():
    async with app.state.pool.acquire() as conn:
        chunks = await conn.fetchval("SELECT count(*) FROM chunks")
    return {"status": "ok", "chunks": chunks}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/search")
async def search(q: str = Query(min_length=2, max_length=500),
                 mode: Literal["vector", "keyword", "hybrid", "rerank"] = "rerank",
                 k: int = Query(5, ge=1, le=20), groups: list[str] = Depends(caller_groups)):
    async with app.state.pool.acquire() as conn:
        return await rag.search(conn, q, mode, k, groups=groups)


@app.post("/ask")
async def ask(body: AskRequest, groups: list[str] = Depends(caller_groups)):
    start = time.perf_counter()
    with rag.timed("ask_total"):
        passages, dropped = await retrieve(body.question, groups)
        answer = await rag.generate(body.question, passages)
        return await finish(answer, body.question, passages, start, dropped)


@app.post("/ask/stream")
async def ask_stream(body: AskRequest, groups: list[str] = Depends(caller_groups)):
    """Server-Sent Events: `sources` after retrieval, `token` while the model writes, then `final`
    (same body as /ask). Tokens are a draft: if the checked answer fails, `final` replaces it."""
    async def events():
        start = time.perf_counter()
        with rag.timed("ask_total"):
            passages, dropped = await retrieve(body.question, groups)
            yield sse("sources", sources(passages))
            parts = []
            try:
                with rag.timed("llm"):
                    async for token in rag.generate_stream(body.question, passages):
                        if not parts:
                            rag.record("first_token", time.perf_counter() - start)
                        parts.append(token)
                        yield sse("token", {"text": token})
                answer = "".join(parts).strip() or None
            except (httpx.HTTPError, ValueError):
                answer = None  # model failed, possibly mid-answer: discard the draft
            result = await finish(answer, body.question, passages, start, dropped)
        yield sse("final", result)
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def retrieve(question: str, groups: list[str]) -> tuple[list[dict], list[dict]]:
    """Search, then quarantine passages that try to instruct the model (docs can be untrusted)."""
    async with app.state.pool.acquire() as conn:
        found = await rag.search(conn, question, groups=groups)
    passages, dropped = rag.quarantine(found)
    if dropped:
        QUARANTINED.inc(len(dropped))
        log.warning(json.dumps({"event": "quarantine", "docs": [d["doc"] for d in dropped]}))
    return passages, dropped


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def sources(passages: list[dict]) -> list[dict]:
    return [{"n": i, "doc": p["doc"], "content": p["content"]} for i, p in enumerate(passages, start=1)]


async def finish(answer: str | None, question: str, passages: list[dict], start: float,
                 quarantined: list[dict] = ()) -> dict:
    """Check the model's answer; fall back to an extractive one if it is missing or unsupported."""
    with rag.timed("verify"):
        claims = await asyncio.to_thread(rag.verify, answer, passages) if answer else []
    engine, rejected = "qwen", None
    # Guardrail: never return an LLM answer that is empty or fails verification; fall back to an
    # extractive answer quoted from the top passages, and show what was rejected and why.
    if answer is None or not rag.strip_citations(answer) or not all(c["supported"] for c in claims):
        if answer is not None:
            rejected = {"answer": answer, "claims": claims}
        with rag.timed("extractive"):
            answer, engine = await asyncio.to_thread(rag.extractive_answer, question, passages), "extractive"
            claims = await asyncio.to_thread(rag.verify, answer, passages)
    verified = all(c["supported"] for c in claims)
    ANSWERS.labels(engine, str(verified).lower()).inc()
    # One structured line per request: grep/jq-able, no question text (may hold user data).
    log.info(json.dumps({"event": "ask", "engine": engine, "verified": verified,
                         "llm_rejected": rejected is not None, "passages": len(passages),
                         "quarantined": len(quarantined),
                         "ms": round((time.perf_counter() - start) * 1000, 1)}))
    return {"answer": answer, "engine": engine, "verified": verified, "claims": claims,
            "rejected_llm_answer": rejected, "quarantined": list(quarantined), "sources": sources(passages)}
