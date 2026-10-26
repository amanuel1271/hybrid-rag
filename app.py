"""HTTP API: ingest a docs folder at startup, then /search and /ask over it."""
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import asyncpg
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

import rag

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:55433/rag")
DOCS_DIR = Path(os.environ.get("DOCS_DIR", Path(__file__).parent / "docs"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await asyncpg.create_pool(DATABASE_URL)
    async with app.state.pool.acquire() as conn:
        await conn.execute((Path(__file__).parent / "schema.sql").read_text())
        app.state.ingested = await rag.ingest(conn, DOCS_DIR)
    yield
    await app.state.pool.close()


app = FastAPI(title="hybrid-rag", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/health")
async def health():
    async with app.state.pool.acquire() as conn:
        chunks = await conn.fetchval("SELECT count(*) FROM chunks")
    return {"status": "ok", "chunks": chunks}


@app.get("/search")
async def search(q: str = Query(min_length=2, max_length=500),
                 mode: Literal["vector", "keyword", "hybrid", "rerank"] = "rerank",
                 k: int = Query(5, ge=1, le=20)):
    async with app.state.pool.acquire() as conn:
        return await rag.search(conn, q, mode, k)


@app.post("/ask")
async def ask(body: AskRequest):
    async with app.state.pool.acquire() as conn:
        passages = await rag.search(conn, body.question)
    answer = await rag.generate(body.question, passages)
    claims = await asyncio.to_thread(rag.verify, answer, passages) if answer else []
    engine, rejected = "qwen", None
    # Guardrail: never return an LLM answer that is empty or fails verification; fall back to an
    # extractive answer quoted from the top passages, and show what was rejected and why.
    if answer is None or not rag.strip_citations(answer) or not all(c["supported"] for c in claims):
        if answer is not None:
            rejected = {"answer": answer, "claims": claims}
        answer, engine = await asyncio.to_thread(rag.extractive_answer, body.question, passages), "extractive"
        claims = await asyncio.to_thread(rag.verify, answer, passages)
    return {
        "answer": answer,
        "engine": engine,
        "verified": all(c["supported"] for c in claims),
        "claims": claims,
        "rejected_llm_answer": rejected,
        "sources": [{"n": i, "doc": p["doc"], "content": p["content"]} for i, p in enumerate(passages, start=1)],
    }
