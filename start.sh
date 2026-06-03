#!/bin/bash
# Single-process startup for Render free tier.
# Runs API + both Celery workers in the same container.
set -e

mkdir -p /tmp/uploads /tmp/table_store

echo "Starting Celery parse worker..."
celery -A ingestion.worker worker -Q parse --concurrency=2 --loglevel=info \
  --logfile=/tmp/worker-parse.log --detach

echo "Starting Celery embed worker..."
celery -A ingestion.worker worker -Q embed --concurrency=4 --loglevel=info \
  --logfile=/tmp/worker-embed.log --detach

echo "Starting FastAPI..."
exec uvicorn api.main:app --host 0.0.0.0 --port $PORT
