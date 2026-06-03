#!/bin/bash
# Single-process startup for Render free tier.
# Runs API + both Celery workers in the same container.
# No set -e — we want the API to start even if workers have issues.

mkdir -p /tmp/uploads /tmp/table_store

echo "Starting Celery parse worker (detached)..."
celery -A ingestion.worker worker -Q parse --concurrency=2 --loglevel=info \
  --logfile=/tmp/worker-parse.log --pidfile=/tmp/worker-parse.pid --detach || \
  echo "Warning: parse worker failed to start, continuing..."

echo "Starting Celery embed worker (detached)..."
celery -A ingestion.worker worker -Q embed --concurrency=4 --loglevel=info \
  --logfile=/tmp/worker-embed.log --pidfile=/tmp/worker-embed.pid --detach || \
  echo "Warning: embed worker failed to start, continuing..."

echo "Workers started. Launching FastAPI on port ${PORT:-8000}..."
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
