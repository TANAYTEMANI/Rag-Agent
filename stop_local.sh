#!/usr/bin/env bash
# Stop all local RAG processes
echo "Stopping workers..."
celery -A ingestion.worker control shutdown 2>/dev/null || true
[ -f /tmp/rag_worker_parse.pid ] && kill "$(cat /tmp/rag_worker_parse.pid)" 2>/dev/null; rm -f /tmp/rag_worker_parse.pid
[ -f /tmp/rag_worker_embed.pid ] && kill "$(cat /tmp/rag_worker_embed.pid)" 2>/dev/null; rm -f /tmp/rag_worker_embed.pid
pkill -f "celery.*rag" 2>/dev/null || true
pkill -f "uvicorn api.main" 2>/dev/null || true
echo "Done. Infrastructure (Redis/Postgres/Qdrant) left running."
