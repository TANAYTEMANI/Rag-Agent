"""
Ragas evaluation harness.
Run: python -m evaluation.ragas_eval --dataset path/to/golden.json
Golden dataset format: list of {question, ground_truth, relevant_doc_ids}
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx


async def run_query(client: httpx.AsyncClient, query: str, base_url: str) -> tuple[str, list]:
    resp = await client.post(
        f"{base_url}/query",
        json={"query": query, "stream": False},
        timeout=60.0,
    )
    data = resp.json()
    return data["answer"], data.get("citations", [])


async def build_ragas_dataset(golden: list[dict], base_url: str) -> dict:
    """Query the running API for each golden question and collect answers."""
    questions, answers, contexts, ground_truths = [], [], [], []

    async with httpx.AsyncClient() as client:
        for item in golden:
            answer, citations = await run_query(client, item["question"], base_url)
            questions.append(item["question"])
            answers.append(answer)
            contexts.append([c.get("text", "") for c in citations])
            ground_truths.append(item["ground_truth"])

    return {
        "question":   questions,
        "answer":     answers,
        "contexts":   contexts,
        "ground_truth": ground_truths,
    }


def run_evaluation(golden_path: str, base_url: str = "http://localhost:8000") -> dict:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    with open(golden_path) as f:
        golden = json.load(f)

    raw = asyncio.run(build_ragas_dataset(golden, base_url))
    dataset = Dataset.from_dict(raw)

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    print(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--api", default="http://localhost:8000")
    args = parser.parse_args()
    run_evaluation(args.dataset, args.api)
