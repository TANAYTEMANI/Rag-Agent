"""
Batch ingestion test — submits N documents in parallel and measures:
  - Submission throughput
  - Time to first indexed document
  - Time to all documents indexed
  - Per-stage breakdown (parsing → enriching → indexing → indexed)

Usage:
  python3 test_batch_ingest.py --docs /tmp/rag_test_docs --api http://localhost:8000
"""
import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

import httpx

BASE = "http://localhost:8000"


async def submit_doc(client: httpx.AsyncClient, path: Path,
                     department: str = "test") -> dict:
    """Submit one document for ingestion, return {doc_id, file_name, status}."""
    with open(path, "rb") as f:
        r = await client.post(
            f"{BASE}/ingest",
            files={"file": (path.name, f, "application/pdf")},
            data={"department": department},
            timeout=30,
        )
    r.raise_for_status()
    d = r.json()
    return {"doc_id": d["doc_id"], "file_name": path.name,
            "status": d["status"], "submitted_at": time.time()}


async def poll_status(client: httpx.AsyncClient, doc_id: str) -> dict:
    """Fetch current status of one document."""
    r = await client.get(f"{BASE}/ingest/{doc_id}", timeout=10)
    r.raise_for_status()
    return r.json()


async def wait_for_all(doc_ids: list[str], timeout: int = 600,
                       poll_interval: float = 5.0) -> dict[str, dict]:
    """Poll all jobs until all reach a terminal state or timeout."""
    terminal = {"indexed", "failed", "duplicate"}
    results = {}
    start = time.time()

    async with httpx.AsyncClient() as client:
        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                print(f"\n  Timeout after {timeout}s — some docs may still be processing")
                break

            pending = [d for d in doc_ids if d not in results
                       or results[d]["status"] not in terminal]
            if not pending:
                break

            tasks = [poll_status(client, doc_id) for doc_id in pending]
            statuses = await asyncio.gather(*tasks, return_exceptions=True)

            counts = defaultdict(int)
            for doc_id, status in zip(pending, statuses):
                if isinstance(status, Exception):
                    continue
                results[doc_id] = status
                counts[status["status"]] += 1

            total = len(doc_ids)
            done  = sum(1 for r in results.values() if r.get("status") in terminal)
            print(f"\r  [{elapsed:5.0f}s]  "
                  f"indexed={counts.get('indexed',0):3d}  "
                  f"enriching={counts.get('enriching',0):3d}  "
                  f"parsing={counts.get('parsing',0):3d}  "
                  f"failed={counts.get('failed',0):3d}  "
                  f"→ {done}/{total} complete",
                  end="", flush=True)

            if done == total:
                break
            await asyncio.sleep(poll_interval)

    print()
    return results


async def run_batch_test(doc_dir: str, api: str, concurrency: int = 20) -> None:
    global BASE
    BASE = api.rstrip("/")

    docs = sorted(Path(doc_dir).glob("*.pdf"))
    if not docs:
        print(f"No PDFs found in {doc_dir}")
        return

    print(f"\n{'='*60}")
    print(f"  BATCH INGESTION TEST")
    print(f"  Documents : {len(docs)}")
    print(f"  API       : {BASE}")
    print(f"  Concurrency: {concurrency} parallel submissions")
    print(f"{'='*60}\n")

    # ── Step 1: Submit all documents in parallel ──────────────────────────
    print(f"[1/3] Submitting {len(docs)} documents in parallel...")
    t_submit_start = time.time()

    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)

        async def bounded_submit(p: Path) -> dict:
            async with sem:
                try:
                    return await submit_doc(client, p)
                except Exception as e:
                    return {"doc_id": None, "file_name": p.name,
                            "status": "error", "error": str(e)}

        submitted = await asyncio.gather(*[bounded_submit(p) for p in docs])

    t_submit_end = time.time()
    submit_duration = t_submit_end - t_submit_start

    queued     = [s for s in submitted if s["status"] == "queued"]
    duplicates = [s for s in submitted if s["status"] == "duplicate"]
    errors     = [s for s in submitted if s["status"] == "error"]

    print(f"  Submitted  : {len(queued)} new  |  {len(duplicates)} duplicates  |  {len(errors)} errors")
    print(f"  Submit time: {submit_duration:.2f}s  ({len(docs)/submit_duration:.1f} docs/sec)\n")

    if not queued and not duplicates:
        print("  Nothing to wait for — all errored. Check API logs.")
        return

    # All doc_ids to track (new + duplicates that were already indexed)
    all_doc_ids = [s["doc_id"] for s in submitted if s["doc_id"]]

    # ── Step 2: Poll until all indexed ───────────────────────────────────
    print(f"[2/3] Waiting for {len(queued)} new documents to be indexed...")
    print(f"      (duplicates return immediately)\n")
    t_index_start = time.time()
    final_statuses = await wait_for_all(all_doc_ids)
    t_index_end = time.time()
    total_elapsed = t_index_end - t_submit_start

    # ── Step 3: Print results ─────────────────────────────────────────────
    print(f"\n[3/3] Results\n{'─'*60}")

    indexed  = [v for v in final_statuses.values() if v.get("status") == "indexed"]
    failed   = [v for v in final_statuses.values() if v.get("status") == "failed"]
    still_on = [v for v in final_statuses.values() if v.get("status") not in
                {"indexed", "failed", "duplicate"}]

    print(f"  {'Indexed':12s}: {len(indexed)}")
    print(f"  {'Duplicates':12s}: {len(duplicates)}")
    print(f"  {'Failed':12s}: {len(failed)}")
    print(f"  {'Still pending':12s}: {len(still_on)}")

    # Timing breakdown for successfully indexed docs
    durations = []
    for v in indexed:
        if v.get("started_at") and v.get("completed_at"):
            try:
                from datetime import datetime
                fmt = "%Y-%m-%dT%H:%M:%S.%f"
                s = datetime.fromisoformat(str(v["started_at"]).split("+")[0])
                e = datetime.fromisoformat(str(v["completed_at"]).split("+")[0])
                durations.append((e - s).total_seconds())
            except Exception:
                pass

    print(f"\n{'─'*60}")
    print(f"  TIMING SUMMARY")
    print(f"{'─'*60}")
    print(f"  Submit {len(docs)} docs     : {submit_duration:.1f}s")
    print(f"  Total wall-clock time : {total_elapsed:.0f}s  ({total_elapsed/60:.1f} min)")

    if durations:
        durations.sort()
        print(f"\n  Per-document parse+index time (from DB timestamps):")
        print(f"    Min    : {min(durations):.1f}s")
        print(f"    Median : {durations[len(durations)//2]:.1f}s")
        print(f"    p95    : {durations[int(len(durations)*0.95)]:.1f}s")
        print(f"    Max    : {max(durations):.1f}s")
        if len(queued) > 0:
            print(f"\n  Effective throughput: {len(indexed)/max(total_elapsed,1):.2f} docs/sec")
            print(f"  (with {4} parse workers running in parallel)")

    if failed:
        print(f"\n  Failed documents:")
        for v in failed:
            print(f"    {v.get('file_name','?'):40s} {v.get('error_message','')[:60]}")

    # ── Step 4: Quick retrieval spot-check ───────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  RETRIEVAL SPOT-CHECKS (queries against indexed corpus)")
    print(f"{'─'*60}")

    spot_queries = [
        ("What is the annual leave entitlement for employees?",
         ["20 days", "leave"]),
        ("What was the cloud infrastructure budget approved by the board?",
         ["2.4 million", "budget"]),
        ("What is the NPS score for Q3?",
         ["67", "NPS", "promoter"]),
        ("What is the enterprise pricing per user?",
         ["40 dollars", "enterprise", "user"]),
        ("What happened during the October 2024 outage?",
         ["47 minutes", "outage", "authentication"]),
    ]

    async with httpx.AsyncClient() as client:
        for query, expected_keywords in spot_queries:
            try:
                r = await client.post(
                    f"{BASE}/query",
                    json={"query": query, "stream": False},
                    timeout=60,
                )
                data = r.json()
                answer = data.get("answer", "")
                citations = data.get("citations", [])
                cache = data.get("from_cache", "none")
                hit = any(k.lower() in answer.lower() for k in expected_keywords)
                status_str = "✓ HIT" if hit else "✗ MISS"
                cache_str  = f" [{cache}]" if cache else ""
                print(f"\n  {status_str}{cache_str}")
                print(f"  Q: {query}")
                print(f"  A: {answer[:160]}{'...' if len(answer)>160 else ''}")
                if citations:
                    c = citations[0]
                    print(f"  Source: {c.get('doc_id','?')[:36]}  page={c.get('page')}  "
                          f"score={c.get('reranker_score',0):.3f}")
            except Exception as e:
                print(f"\n  ERROR on query: {e}")

    print(f"\n{'='*60}")
    print(f"  Test complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="/tmp/rag_test_docs")
    parser.add_argument("--api",  default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(run_batch_test(args.docs, args.api, args.concurrency))
