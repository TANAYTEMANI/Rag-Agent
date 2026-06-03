# RAG Agent — Backend

Production-grade multimodal RAG system. Ingests thousands of documents (PDF, DOCX, PPTX, Excel, CSV, images), retrieves with hybrid dense+sparse search, and generates grounded answers with source citations.

## Quick Start

### 1. Copy env file and fill in API keys
```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY and JINA_API_KEY
```

### 2. Start infrastructure
```bash
docker-compose up qdrant postgres redis -d
```

### 3. Run database migrations
```bash
# Wait for postgres to be healthy, then:
docker exec -i $(docker-compose ps -q postgres) psql -U rag -d ragdb < db/migrations.sql
```

### 4. Start the API and workers
```bash
# Option A: Docker (production)
docker-compose up --build

# Option B: Local development
pip install poetry
poetry install
uvicorn api.main:app --reload --port 8000 &
celery -A ingestion.worker worker -Q parse --concurrency=4 &
celery -A ingestion.worker worker -Q embed --concurrency=16 &
```

### 5. Verify everything is up
```bash
curl http://localhost:8000/health
```

Expected:
```json
{"status": "ok", "qdrant": "ok", "postgres": "ok", "redis": "ok"}
```

---

## API Reference

### Ingest a document
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@report.pdf" \
  -F "department=finance"
```
Response:
```json
{"doc_id": "uuid...", "file_name": "report.pdf", "status": "queued", "message": "..."}
```

### Check ingestion status
```bash
# Single document
curl http://localhost:8000/ingest/{doc_id}

# Batch summary
curl http://localhost:8000/ingest/status
```

### Query (streaming)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the Q3 revenue?", "stream": true}'
```
Returns SSE stream of `{"type":"token","text":"..."}` events, followed by `{"type":"citations","data":[...]}` and `{"type":"done"}`.

### Query (non-streaming)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the Q3 revenue?", "stream": false}'
```

---

## Architecture

```
POST /ingest
  → SHA-256 dedup check (Redis)
  → Celery parse task (parse queue, 4 workers)
      → Docling parser (warm-loaded, one instance per worker)
      → Content router: text / tables / images / Excel
      → Celery embed tasks (embed queue, 16 workers)
          → Contextual enrichment: Claude Haiku (concurrent, prompt-cached)
          → Jina late-chunking embed (batch)
          → Jina v4 image embed (batch)
          → Qdrant upsert (dense + BM25 text index)
          → PostgreSQL metadata

POST /query
  → L1 exact Redis cache (<1ms)
  → Embed query (Jina) + route + filter (parallel)
  → L2 semantic Qdrant cache (~15ms)
  → L3 retrieval result cache (Redis)
  → HyDE: hypothetical passage → embed
  → Hybrid retrieval: dense + sparse + image (parallel, RRF fusion)
  → Jina Reranker v3 (top-20 → top-5, score threshold gate)
  → Parent chunk expansion
  → Claude Sonnet streaming (prompt-cached system prompt)
```

---

## Key Design Decisions

| Decision | Why |
|---|---|
| Two Celery queues (parse / embed) | Parse is CPU-bound (4 workers); embed is I/O-bound (16 workers). Prevents large docs from blocking small ones. |
| Docling warm-loaded per worker | Avoids ~5-10s cold-start per document. Models loaded once at worker startup. |
| Concurrent contextual enrichment | `asyncio.gather` fires all Haiku calls simultaneously — ~20x faster than serial for a 20-chunk batch. |
| Jina late chunking API | `late_chunking=True` flag gives each chunk full-document context awareness. No local GPU needed. |
| Qdrant sparse (BM25 text index) | No separate Elasticsearch needed. Dense + sparse in one collection. |
| Jina Reranker v3 as relevance gate | Score threshold replaces CRAG (per-chunk LLM calls) with zero extra latency. |
| Pandas Query Engine | Excel/CSV → Parquet DataFrames. LLM generates Pandas expressions for exact numerical answers. |
| Three-level cache (L1/L2/L3) | L1 exact: <1ms. L2 semantic: ~15ms. L3 retrieval: skips ~300ms of retrieval+reranking. |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | — | Claude Haiku (enrichment) + Sonnet (generation) |
| `JINA_API_KEY` | ✅ | — | Embeddings + reranker (10M free tokens) |
| `POSTGRES_URL` | — | localhost | Async SQLAlchemy URL |
| `REDIS_URL` | — | localhost | Celery broker + cache |
| `QDRANT_HOST` | — | localhost | Vector DB |
| `GENERATION_MODEL` | — | claude-sonnet-4-6 | LLM for answers |
| `ENRICHMENT_MODEL` | — | claude-haiku-4-5-20251001 | LLM for chunk context |

---

## Running Evaluation

```bash
# Create a golden Q&A JSON file:
# [{"question": "...", "ground_truth": "...", "relevant_doc_ids": ["..."]}]

python -m evaluation.ragas_eval --dataset golden.json --api http://localhost:8000
```

Metrics: Faithfulness, Answer Relevance, Context Precision, Context Recall.
