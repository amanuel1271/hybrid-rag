"""Retrieval eval: recall@1, recall@5 and MRR of each search mode on golden.json.

Run: docker compose run --rm api python eval.py
"""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app import app

MODES = ["vector", "keyword", "hybrid", "rerank"]
GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text())


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
    return {"mode": mode, "recall@1": hits1 / n, f"recall@{k}": hitsk / n, "mrr": rr / n}


if __name__ == "__main__":
    with TestClient(app) as client:
        print(f"{len(GOLDEN)} questions\n| mode | recall@1 | recall@5 | MRR |\n|---|---|---|---|")
        for mode in MODES:
            r = evaluate(client, mode)
            print(f"| {mode} | {r['recall@1']:.2f} | {r['recall@5']:.2f} | {r['mrr']:.2f} |")
