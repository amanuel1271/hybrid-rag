"""Hybrid retrieval-augmented QA over a folder of markdown docs.

question -> vector search (pgvector) + keyword search (Postgres full-text)
         -> reciprocal rank fusion -> cross-encoder rerank -> top passages
         -> answer with [n] citations (Qwen via Ollama, extractive fallback)
         -> every sentence checked against the passage it cites.
"""
import asyncio
import hashlib
import math
import os
import re
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import httpx
import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from prometheus_client import Histogram

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dims, runs locally on CPU
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 20  # per retriever, before fusion
RRF_K = 60  # standard reciprocal-rank-fusion constant
SUPPORT_MIN = 0.70  # calibrated: real cited claims 0.77-1.00 (median 0.88), unrelated sentences <= 0.67
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
MODEL_CACHE = os.environ.get("FASTEMBED_CACHE_PATH")
IDK = "I don't know."
CITE = r"\[\d+(?:\s*[,\-]\s*\d+)*\]"  # [1], [1][2], [1, 2], [1-3]
SENTENCE_END = r"(?<=[.!?])(?<!e\.g\.)(?<!i\.e\.)\s+|\n+"


# Per-stage latency: a Prometheus histogram for /metrics, plus recent raw samples for exact
# p50/p95 in eval.py. ponytail: in-process samples, per-worker; Prometheus is the cross-worker view.
STAGE_SECONDS = Histogram("rag_stage_seconds", "Time spent in each pipeline stage", ["stage"],
                          buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60))
SAMPLES: dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))


@contextmanager
def timed(stage: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        seconds = time.perf_counter() - start
        STAGE_SECONDS.labels(stage).observe(seconds)
        SAMPLES[stage].append(seconds)


def percentile(values, q: float) -> float:
    """Nearest-rank percentile (q in 0..100) of a non-empty sequence."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


@cache
def embedder() -> TextEmbedding:
    return TextEmbedding(EMBED_MODEL, cache_dir=MODEL_CACHE)


@cache
def reranker() -> TextCrossEncoder:
    return TextCrossEncoder(RERANK_MODEL, cache_dir=MODEL_CACHE)


def embed(texts: list[str]) -> list[np.ndarray]:
    return [v / np.linalg.norm(v) for v in embedder().embed(texts)]


def to_pgvector(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def chunk(text: str) -> list[str]:
    """One chunk per '## ' section, prefixed with the doc and section title so it stands alone."""
    title, _, rest = text.strip().partition("\n")
    title = title.lstrip("# ").strip()
    sections = re.split(r"\n(?=## )", rest.strip())
    chunks = []
    for section in sections:
        heading, _, body = section.partition("\n")
        if body.strip():
            chunks.append(f"{title} - {heading.lstrip('# ').strip()}\n{body.strip()}")
    return chunks


async def ingest(conn, docs_dir: Path) -> int:
    """Idempotent: unchanged sections are skipped by content hash; removed sections and files are deleted."""
    added = 0
    paths = sorted(docs_dir.glob("*.md"))
    await conn.execute("DELETE FROM chunks WHERE doc <> ALL($1::text[])", [p.name for p in paths])
    for path in paths:
        chunks = chunk(path.read_text())
        hashes = [hashlib.sha256(c.encode()).hexdigest() for c in chunks]
        known = {r["content_hash"] for r in await conn.fetch(
            "SELECT content_hash FROM chunks WHERE doc = $1", path.name)}
        new = [(c, h) for c, h in zip(chunks, hashes) if h not in known]
        for (content, h), vec in zip(new, embed([c for c, _ in new])):
            await conn.execute(
                "INSERT INTO chunks (doc, content, content_hash, embedding) VALUES ($1, $2, $3, $4::vector)",
                path.name, content, h, to_pgvector(vec))
        added += len(new)
        await conn.execute("DELETE FROM chunks WHERE doc = $1 AND content_hash <> ALL($2::text[])", path.name, hashes)
    return added


async def vector_search(conn, question: str, limit: int = CANDIDATES) -> list[int]:
    with timed("embed"):
        [qv] = await asyncio.to_thread(embed, [question])
    with timed("vector_db"):
        rows = await conn.fetch(
            "SELECT id FROM chunks ORDER BY embedding <=> $1::vector LIMIT $2", to_pgvector(qv), limit)
    return [r["id"] for r in rows]


async def keyword_search(conn, question: str, limit: int = CANDIDATES) -> list[int]:
    # OR the question's terms (plainto_tsquery ANDs them, which misses on long questions);
    # ts_rank_cd then rewards chunks matching more terms, closer together.
    with timed("keyword_db"):
        rows = await conn.fetch(
            """SELECT id FROM chunks,
                      (SELECT replace(plainto_tsquery('english', $1)::text, '&', '|')::tsquery AS q) t
               WHERE tsv @@ t.q ORDER BY ts_rank_cd(tsv, t.q) DESC LIMIT $2""",
            question, limit)
    return [r["id"] for r in rows]


def rrf(*rankings: list[int], k: int = RRF_K) -> list[int]:
    """Reciprocal rank fusion: an id ranked high by either retriever rises to the top."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, id_ in enumerate(ranking, start=1):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


async def fetch_chunks(conn, ids: list[int]) -> list[dict]:
    rows = await conn.fetch("SELECT id, doc, content FROM chunks WHERE id = ANY($1::bigint[])", ids)
    by_id = {r["id"]: dict(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


async def search(conn, question: str, mode: str = "rerank", k: int = 5) -> list[dict]:
    """mode: vector | keyword | hybrid (RRF) | rerank (RRF + cross-encoder). Returns top-k passages."""
    if mode == "vector":
        ids = await vector_search(conn, question)
    elif mode == "keyword":
        ids = await keyword_search(conn, question)
    elif mode in ("hybrid", "rerank"):
        ids = rrf(await vector_search(conn, question), await keyword_search(conn, question))
    else:
        raise ValueError(f"unknown search mode: {mode}")
    passages = await fetch_chunks(conn, ids if mode == "rerank" else ids[:k])
    if mode == "rerank" and passages:
        with timed("rerank"):
            scores = await asyncio.to_thread(lambda: list(reranker().rerank(question, [p["content"] for p in passages])))
        for p, s in zip(passages, scores):
            p["score"] = float(s)
        passages.sort(key=lambda p: p["score"], reverse=True)
    return passages[:k]


PROMPT = """Answer the question in 1 to 3 short sentences using ONLY the numbered passages below.
Put the passage number in square brackets at the end of each sentence, for example [3].
If the passages do not contain the answer, reply exactly: I don't know.

{passages}

Question: {question}
Answer:"""


async def generate(question: str, passages: list[dict]) -> str | None:
    """Qwen via Ollama. Returns None when the model is unavailable (caller falls back)."""
    numbered = "\n\n".join(f"[{i}] {p['content']}" for i, p in enumerate(passages, start=1))
    try:
        with timed("llm"):
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(f"{OLLAMA_URL}/api/generate", json={
                    "model": OLLAMA_MODEL, "stream": False, "options": {"temperature": 0},
                    "prompt": PROMPT.format(passages=numbered, question=question)})
        r.raise_for_status()
        return r.json()["response"].strip() or None
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def extractive_answer(question: str, passages: list[dict]) -> str:
    """Fallback without a usable LLM answer: the sentence from the top 3 passages that the
    cross-encoder rates most relevant to the question, quoted verbatim with its citation."""
    candidates = [(n, s) for n, p in enumerate(passages[:3], start=1) for s in sentences_of(p["content"])]
    if not candidates:
        return IDK
    scores = list(reranker().rerank(question, [s for _, s in candidates]))
    n, sentence = candidates[max(range(len(candidates)), key=scores.__getitem__)]
    return f"{sentence} [{n}]"


def sentences_of(content: str) -> list[str]:
    body = content.split("\n", 1)[-1]
    return [s.strip().lstrip("-*• ") for s in re.split(SENTENCE_END, body) if s.strip()]


def split_claims(answer: str) -> list[str]:
    """Sentences with their citations attached. Models cite either after a sentence ("x [1]." / "x. [1]")
    or before it ("[1] x."); an answer that opens with a citation uses the leading style throughout."""
    answer = re.sub(rf"^\s*((?:{CITE}\s*)*)answer\s*:\s*", r"\1", answer.strip(), flags=re.I)
    if not re.match(CITE, answer):  # trailing style: pull "x. [1]" back into its sentence
        answer = re.sub(rf"([.!?])[ \t]*((?:{CITE}[ \t]*)+)(?=\s|$)", lambda m: " " + m.group(2).strip() + m.group(1), answer)
    claims: list[str] = []
    for s in (s.strip().lstrip("-*• ") for s in re.split(SENTENCE_END, answer)):
        if s and not strip_citations(s) and claims:
            claims[-1] += " " + s  # a dangling "[1]" belongs to the sentence before it
        elif s:
            claims.append(s)
    return claims


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def strip_citations(s: str) -> str:
    return re.sub(rf"\s*{CITE}", "", s).strip()


def citations(s: str) -> list[int]:
    found = []
    for group in re.findall(CITE, s):
        for part in group.strip("[]").split(","):
            lo, _, hi = part.partition("-")
            found += range(int(lo), int(hi or lo) + 1)
    return list(dict.fromkeys(found))


def verify(answer: str, passages: list[dict]) -> list[dict]:
    """Every sentence must cite a real passage, and be semantically close to what it cites."""
    sentences = [s for s in split_claims(answer) if strip_citations(s).lower() != IDK.lower()]  # abstaining is not a claim
    claims = []
    for s in sentences:
        cited = [n for n in citations(s) if 1 <= n <= len(passages)]
        text = strip_citations(s)
        if not cited:
            claims.append({"sentence": text, "citations": [], "supported": False, "reason": "no valid citation"})
            continue
        # Compare with each sentence of the cited passages: a whole multi-fact passage dilutes the match.
        evidence = [e for n in cited for e in sentences_of(passages[n - 1]["content"])]
        sv, *evs = embed([text, *evidence])
        best = max(float(sv @ ev) for ev in evs)
        # Embeddings barely notice numbers ("45 days" ~ "10 days"), so every number must appear verbatim.
        missing = numbers(text) - numbers(" ".join(passages[n - 1]["content"] for n in cited))
        reason = ("cited passage does not support it" if best < SUPPORT_MIN
                  else f"numbers not in the cited passage: {', '.join(sorted(missing))}" if missing else None)
        claims.append({"sentence": text, "citations": cited, "supported": reason is None,
                       "similarity": round(best, 3), "reason": reason})
    return claims
