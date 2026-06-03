#!/usr/bin/env bash
# =============================================================================
# LOCAL DEVELOPMENT RUNNER
# Runs everything natively on macOS — no Docker overhead.
# Requires: brew install redis postgresql@15 qdrant
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${YELLOW}→${NC} $1"; }
ok()    { echo -e "${GREEN}✓${NC} $1"; }
fail()  { echo -e "${RED}✗${NC} $1"; exit 1; }

VENV="$ROOT/.venv"

# ── 1. Check / create virtualenv ─────────────────────────────────────────────
if [ ! -f "$VENV/bin/activate" ]; then
  info "Creating virtualenv..."
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# ── 2. Install dependencies if needed ────────────────────────────────────────
if ! python3 -c "import fastapi" 2>/dev/null; then
  info "Installing Python dependencies (this takes ~2 min first time)..."
  pip install --upgrade pip -q
  pip install -r requirements.txt -q
  # docling pulls torch — install CPU-only on macOS
  pip install docling -q
  ok "Dependencies installed"
fi

# ── 3. Source .env ────────────────────────────────────────────────────────────
if [ ! -f .env ]; then fail ".env not found — copy .env.example and fill in API keys"; fi
set -a; source .env; set +a

export PYTHONPATH="$ROOT"
export POSTGRES_URL="postgresql+asyncpg://rag:rag@localhost:5432/ragdb"
export REDIS_URL="redis://localhost:6379/0"
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"

# ── 4. Start infrastructure ───────────────────────────────────────────────────
info "Starting Redis..."
if ! redis-cli ping &>/dev/null; then
  redis-server --daemonize yes --logfile /tmp/rag_redis.log
  sleep 1
fi
redis-cli ping | grep -q PONG && ok "Redis running" || fail "Redis failed to start"

info "Starting PostgreSQL..."
if ! pg_isready -h localhost -p 5432 -U rag &>/dev/null; then
  # Try brew service first, fall back to pg_ctl
  brew services start postgresql@15 2>/dev/null || \
    pg_ctl -D "$(brew --prefix)/var/postgresql@15" start -l /tmp/rag_postgres.log 2>/dev/null || true
  sleep 2
fi
if ! pg_isready -h localhost -p 5432 -U rag &>/dev/null; then
  # Create user/db if not exists
  createuser -s rag 2>/dev/null || true
  createdb -U rag ragdb 2>/dev/null || true
fi
pg_isready -h localhost -p 5432 &>/dev/null && ok "PostgreSQL running" || fail "PostgreSQL failed"

# Run migrations (idempotent)
psql -U rag -d ragdb -f db/migrations.sql &>/dev/null
ok "Migrations applied"

info "Starting Qdrant..."
if ! curl -sf http://localhost:6333/healthz &>/dev/null; then
  if command -v qdrant &>/dev/null; then
    nohup qdrant --config-path /dev/null > /tmp/rag_qdrant.log 2>&1 &
  else
    # Try Docker just for Qdrant if not installed natively
    docker run -d --name rag_qdrant_local -p 6333:6333 \
      -v /tmp/qdrant_storage:/qdrant/storage \
      qdrant/qdrant:latest &>/dev/null || true
  fi
  sleep 3
fi
curl -sf http://localhost:6333/healthz &>/dev/null && ok "Qdrant running" || fail "Qdrant failed — run: docker run -d -p 6333:6333 qdrant/qdrant:latest"

mkdir -p uploads table_store

# ── 5. Start workers ──────────────────────────────────────────────────────────
info "Starting Celery parse worker (4 concurrent — native Mac gets full RAM)..."
celery -A ingestion.worker worker \
  -Q parse \
  --concurrency=4 \
  --loglevel=warning \
  --logfile=/tmp/rag_worker_parse.log \
  --pidfile=/tmp/rag_worker_parse.pid \
  --detach

info "Starting Celery embed worker (16 concurrent — I/O bound)..."
celery -A ingestion.worker worker \
  -Q embed \
  --concurrency=16 \
  --loglevel=warning \
  --logfile=/tmp/rag_worker_embed.log \
  --pidfile=/tmp/rag_worker_embed.pid \
  --detach

sleep 3
ok "Workers started"

# ── 6. Start API ──────────────────────────────────────────────────────────────
info "Starting FastAPI server on http://localhost:8000 ..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!
sleep 3

curl -sf http://localhost:8000/health | python3 -c "
import sys,json; d=json.load(sys.stdin)
ok = d['status']=='ok'
print(f'Health: {d}')
sys.exit(0 if ok else 1)
" && ok "API healthy" || fail "API not healthy"

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  RAG Agent running locally${NC}"
echo -e "${GREEN}============================================================${NC}"
echo "  API:       http://localhost:8000"
echo "  Docs:      http://localhost:8000/docs"
echo "  Qdrant UI: http://localhost:6333/dashboard"
echo ""
echo "  Logs:"
echo "    Parse worker:  tail -f /tmp/rag_worker_parse.log"
echo "    Embed worker:  tail -f /tmp/rag_worker_embed.log"
echo "    API:           shown above (stdout)"
echo ""
echo "  Stop everything: bash stop_local.sh"
echo -e "${GREEN}============================================================${NC}"

# Keep API in foreground (Ctrl+C to stop)
wait $API_PID
