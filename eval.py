"""Eval on golden.json: retrieval (recall@1, recall@5, MRR per search mode) and answers
(correct = contains every required fact; verified = passes the claim checker).

  python eval.py           retrieval + extractive answers (no LLM needed)
  python eval.py --llm     also Qwen answers, raw and behind the guardrail (needs Ollama)
  python eval.py --check   fail (exit 1) if any metric drops below eval_baseline.json
"""
import json
import re
import sys
from pathlib import Path

from fastapi.testclient import TestClient

import rag
from app import app

HERE = Path(__file__).parent
MODES = ["vector", "keyword", "hybrid", "rerank"]
GOLDEN = json.loads((HERE / "golden.json").read_text())
BASELINE = HERE / "eval_baseline.json"
TOLERANCE = 0.005  # float noise only: any real drop on 35 questions is >= 1/35


def evaluate(client: TestClient, mode: str, k: int = 5) -> dict:
    hits1 = hitsk = rr = 0.0
    for case in GOLDEN:
        docs = [p["doc"] for p in client.get("/search", params={"q": case["question"], "mode": mode, "k": k}).json()]
        if case["doc"] in docs:
            rank = docs.index(case["doc"]) + 1
            hits1 += rank == 1
            hitsk += 1
            rr += 1 / rank
    n = len(GOLDEN)
    return {"recall@1": hits1 / n, f"recall@{k}": hitsk / n, "mrr": rr / n}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace(",", ""))


def is_correct(answer: str, facts: list[list[str]]) -> bool:
    """Every fact group must appear; a group is satisfied by any one of its alternatives."""
    text = normalize(answer)
    return all(any(normalize(alt) in text for alt in group) for group in facts)


def evaluate_answers(client: TestClient) -> dict:
    """Asks every golden question through /ask; scores what the API returns and, when the LLM
    answered, what the LLM said before the guardrail ("raw")."""
    final_ok = raw_ok = raw_verified = fallbacks = llm_answers = 0
    misses = []
    for case in GOLDEN:
        r = client.post("/ask", json={"question": case["question"]}).json()
        ok = is_correct(r["answer"], case["facts"])
        final_ok += ok
        if not ok:
            misses.append(f"{case['question']} -> {r['answer'][:90]}")
        rejected = r["rejected_llm_answer"]
        if r["engine"] == "qwen" or rejected:
            llm_answers += 1
            raw = rejected["answer"] if rejected else r["answer"]
            raw_ok += is_correct(raw, case["facts"])
            raw_verified += rejected is None
        fallbacks += r["engine"] == "extractive"
    n = len(GOLDEN)
    result = {"correct": final_ok / n, "fallback_rate": fallbacks / n, "misses": misses}
    if llm_answers:
        result |= {"llm_raw_correct": raw_ok / llm_answers, "llm_raw_verified": raw_verified / llm_answers}
    return result


def run(use_llm: bool) -> dict:
    if not use_llm:
        rag.OLLAMA_URL = "http://127.0.0.1:9"  # nothing listens: every answer is extractive
    with TestClient(app) as client:
        results = {m: evaluate(client, m) for m in MODES}
        results["qwen+guardrail" if use_llm else "extractive"] = evaluate_answers(client)
    return results


def regressions(results: dict, baseline: dict) -> list[str]:
    return [f"{group}.{metric}: {results[group][metric]:.3f} < baseline {floor:.3f}"
            for group, metrics in baseline.items() if group in results
            for metric, floor in metrics.items() if results[group][metric] < floor - TOLERANCE]


def report(results: dict) -> None:
    print(f"{len(GOLDEN)} questions\n\n| search mode | recall@1 | recall@5 | MRR |\n|---|---|---|---|")
    for m in MODES:
        r = results[m]
        print(f"| {m} | {r['recall@1']:.2f} | {r['recall@5']:.2f} | {r['mrr']:.2f} |")
    print("\n| answers | correct | fallback rate | LLM raw correct | LLM raw verified |\n|---|---|---|---|---|")
    for engine in ("extractive", "qwen+guardrail"):
        if engine in results:
            r = results[engine]
            raw = lambda key: f"{r[key]:.2f}" if key in r else "-"
            print(f"| {engine} | {r['correct']:.2f} | {r['fallback_rate']:.2f} | {raw('llm_raw_correct')} | {raw('llm_raw_verified')} |")
    for engine in ("extractive", "qwen+guardrail"):
        for miss in results.get(engine, {}).get("misses", []):
            print(f"  miss ({engine}): {miss}")


if __name__ == "__main__":
    results = run(use_llm="--llm" in sys.argv)
    report(results)
    if "--check" in sys.argv:
        failed = regressions(results, json.loads(BASELINE.read_text()))
        print("\nREGRESSION:\n  " + "\n  ".join(failed) if failed else "\nno regressions vs eval_baseline.json")
        sys.exit(1 if failed else 0)
