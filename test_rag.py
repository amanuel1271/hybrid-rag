import asyncio
import json

import asyncpg
import httpx
import pytest
from fastapi.testclient import TestClient

import rag
from app import DATABASE_URL, DOCS_DIR, app
from eval import evaluate, is_correct, regressions

PASSAGES = [
    {"doc": "refunds.md", "content": "Customer Refunds - Processing\nRefunds are issued to the original payment "
                                    "method within 5 business days. Support agents can approve refunds up to 200,000 KRW."},
    {"doc": "pto-policy.md", "content": "Paid Time Off - Sick leave\nEmployees get up to 10 paid sick days per year."},
]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_chunk_one_per_section_with_title():
    chunks = rag.chunk("# Deploy Process\n\n## Canary\nFive percent.\n\n## Rollback\nThree minutes.\n")
    assert chunks == ["Deploy Process - Canary\nFive percent.", "Deploy Process - Rollback\nThree minutes."]


def test_rrf_rewards_agreement_between_retrievers():
    # 7 is only 2nd and 3rd in each list, but both retrievers found it; 1 and 9 were found once.
    assert rag.rrf([1, 7, 3], [9, 2, 7])[0] == 7


@pytest.mark.parametrize("answer, want", [
    ("Refunds take 5 days [1]. Sick leave is 10 days [2].", ["Refunds take 5 days [1].", "Sick leave is 10 days [2]."]),
    ("Refunds take 5 days. [1] Sick leave is 10 days. [2]", ["Refunds take 5 days [1].", "Sick leave is 10 days [2]."]),
    ("[1] Answer: Refunds take 5 days. [2] Sick leave is 10 days.", ["[1] Refunds take 5 days.", "[2] Sick leave is 10 days."]),
    ("Refunds take 5 days. [1]", ["Refunds take 5 days [1]."]),
    ("See e.g. the runbook [1]. Sick leave is 10 days [1, 2].", ["See e.g. the runbook [1].", "Sick leave is 10 days [1, 2]."]),
    ("- Refunds take 5 days [1]\n- Sick leave is 10 days [2]", ["Refunds take 5 days [1]", "Sick leave is 10 days [2]"]),
])
def test_split_claims_handles_citation_styles(answer, want):
    assert rag.split_claims(answer) == want


@pytest.mark.parametrize("answer, supported, reason", [
    ("Refunds go back to the original payment method within 5 business days [1].", [True], None),
    ("Refunds go back to the original payment method within 5 business days [2].", [False], "cited passage does not support it"),
    ("Refunds go back to the original payment method within 5 business days.", [False], "no valid citation"),
    ("Refunds go back to the original payment method within 5 business days [9].", [False], "no valid citation"),
    ("Support agents can approve refunds up to 900,000 KRW [1].", [False], "numbers not in the cited passage: 900000"),
    ("I don't know.", [], None),
])
def test_verify_checks_each_claim_against_its_citation(answer, supported, reason):
    claims = rag.verify(answer, PASSAGES)
    assert [c["supported"] for c in claims] == supported
    if claims:
        assert claims[0]["reason"] == reason


@pytest.mark.parametrize("text, want", [("x [1]", [1]), ("x [1][2]", [1, 2]), ("x [1, 2]", [1, 2]), ("x [1-3]", [1, 2, 3])])
def test_citation_formats(text, want):
    assert rag.citations(text) == want


def test_ingest_is_idempotent(client):
    async def reingest():
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            return await rag.ingest(conn, DOCS_DIR), await conn.fetchval("SELECT count(*) FROM chunks")
        finally:
            await conn.close()
    before = client.get("/health").json()["chunks"]
    added, after = asyncio.run(reingest())
    assert (added, after) == (0, before)


def test_exact_error_code_ranks_first(client):
    top = client.get("/search", params={"q": "SHP-4031"}).json()[0]
    assert top["doc"] == "error-codes.md" and "SHP-4031" in top["content"]


def test_each_stage_earns_its_place(client):
    s = {m: evaluate(client, m) for m in ("vector", "keyword", "hybrid", "rerank")}
    # fusion: finds what either retriever alone misses
    assert s["hybrid"]["recall@5"] > max(s["vector"]["recall@5"], s["keyword"]["recall@5"])
    # rerank: puts the right passage first more often than fusion alone
    assert s["rerank"]["recall@1"] > s["hybrid"]["recall@1"]
    assert s["rerank"]["recall@5"] >= 0.95


def test_ask_falls_back_to_extractive_answer_without_llm(client, monkeypatch):
    monkeypatch.setattr(rag, "OLLAMA_URL", "http://127.0.0.1:9")  # nothing listens here
    r = client.post("/ask", json={"question": "How many paid sick days do employees get?"}).json()
    assert r["engine"] == "extractive"
    assert r["verified"] and r["claims"][0]["citations"] == [1]
    assert "10 paid sick days" in r["answer"]


def test_ask_replaces_an_unsupported_llm_answer(client, monkeypatch):
    async def hallucinating_llm(question, passages):
        return "Employees get 45 days of paid sick leave and a free car [1]."
    monkeypatch.setattr(rag, "generate", hallucinating_llm)
    r = client.post("/ask", json={"question": "How many paid sick days do employees get?"}).json()
    assert r["engine"] == "extractive" and r["verified"]
    assert "45" not in r["answer"]
    assert r["rejected_llm_answer"]["claims"][0]["supported"] is False


def test_ask_returns_a_verified_llm_answer(client, monkeypatch):
    async def grounded_llm(question, passages):
        n = next(i for i, p in enumerate(passages, start=1) if "sick days" in p["content"])
        return f"Employees get up to 10 paid sick days per year [{n}]."
    monkeypatch.setattr(rag, "generate", grounded_llm)
    r = client.post("/ask", json={"question": "How many paid sick days do employees get?"}).json()
    assert r["engine"] == "qwen" and r["verified"] and r["rejected_llm_answer"] is None


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


@pytest.mark.parametrize("answer, facts, want", [
    ("Escalates after 10 minutes [1].", [["10 minutes"]], True),
    ("Kept for 30 days hot, 1 year cold.", [["30 days"], ["1 year", "one year"]], True),
    ("Kept for 30 days.", [["30 days"], ["1 year", "one year"]], False),
    ("Budget is 1,500,000 KRW.", [["1500000"]], True),  # commas ignored
])
def test_answer_correctness_needs_every_fact_group(answer, facts, want):
    assert is_correct(answer, facts) is want


def test_regression_gate_flags_only_real_drops():
    baseline = {"rerank": {"mrr": 0.95}, "qwen+guardrail": {"correct": 0.88}}
    assert regressions({"rerank": {"mrr": 0.951}}, baseline) == []          # missing group (no LLM in CI) is skipped
    assert regressions({"rerank": {"mrr": 0.92}}, baseline) == ["rerank.mrr: 0.920 < baseline 0.950"]


def test_percentile_nearest_rank():
    assert rag.percentile([5, 1, 3, 2, 4], 50) == 3
    assert rag.percentile(range(1, 101), 95) == 95
    assert rag.percentile([7], 95) == 7


def test_timed_records_stage_even_when_it_raises():
    before = len(rag.SAMPLES["unit_test_stage"])
    with pytest.raises(ValueError):
        with rag.timed("unit_test_stage"):
            raise ValueError
    assert len(rag.SAMPLES["unit_test_stage"]) == before + 1


def test_ask_exports_stage_latency_and_answer_metrics(client, monkeypatch):
    monkeypatch.setattr(rag, "OLLAMA_URL", "http://127.0.0.1:9")
    client.post("/ask", json={"question": "How long do refunds take?"})
    metrics = client.get("/metrics").text
    for stage in ("embed", "vector_db", "keyword_db", "rerank", "llm", "extractive", "ask_total"):
        assert f'rag_stage_seconds_count{{stage="{stage}"}}' in metrics
    assert 'rag_answers_total{engine="extractive",verified="true"}' in metrics


def sse_events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((fields["event"], json.loads(fields["data"])))
    return out


def test_ask_stream_sends_sources_then_tokens_then_checked_answer(client, monkeypatch):
    async def grounded_stream(question, passages):
        n = next(i for i, p in enumerate(passages, start=1) if "sick days" in p["content"])
        for token in ["Employees get up to ", "10 paid sick days ", f"per year [{n}]."]:
            yield token
    monkeypatch.setattr(rag, "generate_stream", grounded_stream)
    events = sse_events(client.post("/ask/stream", json={"question": "How many paid sick days do employees get?"}).text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "final" and set(kinds[1:-1]) == {"token"}
    final = events[-1][1]
    assert final["engine"] == "qwen" and final["verified"]
    assert "".join(d["text"] for e, d in events if e == "token") == final["answer"]


def test_ask_stream_discards_a_draft_cut_off_mid_answer(client, monkeypatch):
    async def dying_stream(question, passages):
        yield "Employees get 45 "
        raise httpx.ReadTimeout("model stalled")
    monkeypatch.setattr(rag, "generate_stream", dying_stream)
    final = sse_events(client.post("/ask/stream", json={"question": "How many paid sick days do employees get?"}).text)[-1][1]
    assert final["engine"] == "extractive" and final["verified"] and "45" not in final["answer"]


def test_generate_rejects_a_model_stream_that_ends_before_done(monkeypatch):
    lines = b'{"response": "Refunds take 5 days [1]."}\n'  # connection closed, no {"done": true}
    real = httpx.AsyncClient
    monkeypatch.setattr(rag.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=lines))))
    assert asyncio.run(rag.generate("How long do refunds take?", PASSAGES)) is None
    done = lines + b'{"response": "", "done": true}\n'
    monkeypatch.setattr(rag.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=done))))
    assert asyncio.run(rag.generate("How long do refunds take?", PASSAGES)) == "Refunds take 5 days [1]."
