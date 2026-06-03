# RAG Agent — Command Reference

All commands run from the `rag-agent/` directory.

---

## SETUP

### Docker (simplest, use for submission)
```bash
# First time only — run migrations after containers start
docker compose up qdrant postgres redis -d
docker cp db/migrations.sql $(docker compose ps -q postgres):/tmp/migrations.sql
docker exec $(docker compose ps -q postgres) psql -U rag -d ragdb -f /tmp/migrations.sql

# Start everything
docker compose up --build -d

# Check all 6 services are healthy
docker compose ps
curl http://localhost:8000/health
```

### Local (faster, more parallel — recommended for development)
```bash
# One-time: install infrastructure
brew install redis postgresql@15
brew services start redis
brew services start postgresql@15
createuser -s rag && createdb -U rag ragdb

# Qdrant (no brew formula — use Docker just for this one service)
docker run -d --name qdrant -p 6333:6333 -v /tmp/qdrant_data:/qdrant/storage qdrant/qdrant:latest

# Install Python dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt docling

# Apply migrations
psql -U rag -d ragdb -f db/migrations.sql

# Start everything with one command
bash run_local.sh
```

---

## WIPE ALL DATA (start fresh for demo)

```bash
# 1. Wipe Qdrant vectors
curl -s -X DELETE http://localhost:6333/collections/text_chunks
curl -s -X DELETE http://localhost:6333/collections/image_pages
curl -s -X DELETE http://localhost:6333/collections/query_cache

# 2. Wipe PostgreSQL records
docker exec $(docker compose ps -q postgres) \
  psql -U rag -d ragdb -c "TRUNCATE ingestion_jobs, documents, tabular_tables RESTART IDENTITY CASCADE;"
# OR for local:
psql -U rag -d ragdb -c "TRUNCATE ingestion_jobs, documents, tabular_tables RESTART IDENTITY CASCADE;"

# 3. Wipe Redis dedup cache
redis-cli DEL indexed_hashes hash_to_doc_id
redis-cli FLUSHDB   # wipes ALL redis keys including L1/L2/L3 caches

# 4. Delete uploaded files (optional)
rm -rf uploads/*

# 5. Collections are recreated automatically on next API start
# Restart the API to trigger setup_collections()
docker compose restart api   # Docker
# OR: Ctrl+C the uvicorn process and re-run bash run_local.sh
```

---

## INGEST DOCUMENTS

### Single document
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/document.pdf" \
  -F "department=finance"
```

### Batch ingest a whole folder (runs in parallel)
```bash
python3 test_batch_ingest.py --docs /tmp/rag_test_docs
```

### Watch ingestion progress in real time
```bash
# Overall counts by status
watch -n 2 'curl -s http://localhost:8000/ingest/status | python3 -m json.tool'

# Single document detail
curl http://localhost:8000/ingest/<doc_id> | python3 -m json.tool

# Live parse worker log (Docker)
docker compose logs -f worker-parse

# Live embed worker log (Docker)
docker compose logs -f worker-embed

# Live parse worker log (local)
tail -f /tmp/rag_worker_parse.log

# Live embed worker log (local)
tail -f /tmp/rag_worker_embed.log
```

### Check what's actually indexed in Qdrant
```bash
# Total vector count
curl -s http://localhost:6333/collections/text_chunks | \
  python3 -c "import sys,json; r=json.load(sys.stdin)['result']; print(f'points={r[\"points_count\"]} status={r[\"status\"]}')"

# Sample 5 points to see chunk types
curl -s -X POST http://localhost:6333/collections/text_chunks/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"limit":5,"with_payload":["chunk_type","doc_id","page_no"],"with_vector":false}' | \
  python3 -m json.tool
```

---

## QUERY THE SYSTEM

### Single query (non-streaming)
```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the Q3 2024 total revenue?", "stream": false}' | \
  python3 -m json.tool
```

### Streaming query (SSE)
```bash
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the enterprise pricing per user?", "stream": true}'
```

### Run full end-to-end test suite
```bash
bash test_e2e.sh /tmp/rag_test_docs/board_resolution_feb2024.pdf
```

### Run batch ingestion + retrieval test
```bash
python3 test_batch_ingest.py --docs /tmp/rag_test_docs
```

---

## MONITOR

### Memory and CPU usage
```bash
docker stats --no-stream   # Docker
# OR local: top -o cpu
```

### Cache hit rates
```bash
redis-cli INFO stats | grep -E "keyspace_hits|keyspace_misses"
```

### Qdrant dashboard
```
http://localhost:6333/dashboard
```

### API interactive docs (Swagger)
```
http://localhost:8000/docs
```

---

## TROUBLESHOOTING

### Document stuck in "parsing" for more than 5 min
```bash
# Check parse worker logs
docker compose logs worker-parse --tail=50

# Check if worker is OOM-killed (SIGKILL)
docker compose logs worker-parse | grep -i "sigkill\|killed\|oom"

# Reduce concurrency if OOM
# Edit docker-compose.yml: --concurrency=2 for parse worker
# OR run locally (no Docker memory limit)
```

### "No chunks indexed" after indexing completes
```bash
# Check Qdrant point count
curl -s http://localhost:6333/collections/text_chunks | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])"

# Check embed worker logs for errors
docker compose logs worker-embed | grep -i "error\|fail\|traceback"
```

### Reset and start completely fresh
```bash
docker compose down -v   # destroys all volumes including Qdrant + Postgres data
docker compose up --build -d
docker cp db/migrations.sql $(docker compose ps -q postgres):/tmp/migrations.sql
docker exec $(docker compose ps -q postgres) psql -U rag -d ragdb -f /tmp/migrations.sql
```
