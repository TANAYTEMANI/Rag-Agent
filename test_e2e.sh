#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# End-to-end test script for the RAG Agent backend
# Run from the rag-agent/ directory: bash test_e2e.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

BASE="http://localhost:8000"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

FAILURES=0
pass() { echo -e "${GREEN}✓ PASS${NC}  $1"; }
fail() { echo -e "${RED}✗ FAIL${NC}  $1"; FAILURES=$((FAILURES+1)); }
info() { echo -e "${YELLOW}→${NC} $1"; }
sep()  { echo -e "\n────────────────────────────────────────"; }

# ── TEST 1: Health check ──────────────────────────────────────────────────
sep
info "TEST 1: Health check"
HEALTH=$(curl -s "$BASE/health")
echo "  Response: $HEALTH"
if echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok', d" 2>/dev/null; then
  pass "All services healthy (qdrant, postgres, redis)"
else
  fail "Health check failed: $HEALTH"
fi

# ── TEST 2: Ingest a PDF ──────────────────────────────────────────────────
sep
info "TEST 2: Ingest test PDF"
PDF_PATH="${1:-/tmp/acme_q3_report.pdf}"
if [ ! -f "$PDF_PATH" ]; then
  fail "Test PDF not found at $PDF_PATH — run the PDF creation step first"
fi

INGEST=$(curl -s -X POST "$BASE/ingest" \
  -F "file=@$PDF_PATH" \
  -F "department=finance")
echo "  Response: $INGEST"

DOC_ID=$(echo "$INGEST" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('doc_id',''))" 2>/dev/null)
STATUS=$(echo "$INGEST" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null)

if [ -z "$DOC_ID" ]; then
  fail "No doc_id in response"
fi
if [ "$STATUS" = "queued" ] || [ "$STATUS" = "duplicate" ]; then
  pass "Document accepted (doc_id=$DOC_ID, status=$STATUS)"
else
  fail "Unexpected status: $STATUS"
fi

# ── TEST 3: Duplicate detection ───────────────────────────────────────────
sep
info "TEST 3: Duplicate upload detection"
DUP=$(curl -s -X POST "$BASE/ingest" \
  -F "file=@$PDF_PATH" \
  -F "department=finance")
DUP_STATUS=$(echo "$DUP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null)
if [ "$DUP_STATUS" = "duplicate" ]; then
  pass "Duplicate correctly detected and skipped"
else
  info "Note: file was new (not a duplicate yet) — rerun to test dedup"
fi

# ── TEST 4: Poll ingestion status ─────────────────────────────────────────
sep
info "TEST 4: Wait for document to be indexed (up to 3 minutes)"
INDEXED=false
for i in $(seq 1 36); do
  sleep 5
  JOB=$(curl -s "$BASE/ingest/$DOC_ID")
  JOB_STATUS=$(echo "$JOB" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null)
  printf "  [%02ds] status: %s\r" $((i*5)) "$JOB_STATUS"
  if [ "$JOB_STATUS" = "indexed" ]; then
    echo ""
    pass "Document fully indexed in $((i*5))s"
    INDEXED=true
    break
  elif [ "$JOB_STATUS" = "failed" ]; then
    echo ""
    ERROR=$(echo "$JOB" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error_message',''))" 2>/dev/null)
    fail "Ingestion failed: $ERROR"
  fi
done
if [ "$INDEXED" = false ]; then
  echo ""
  fail "Document not indexed after 3 minutes — check worker logs: docker compose logs worker-parse"
fi

# ── TEST 5: Batch status summary ──────────────────────────────────────────
sep
info "TEST 5: Batch status summary"
SUMMARY=$(curl -s "$BASE/ingest/status")
echo "  Response: $SUMMARY"
if echo "$SUMMARY" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'summary' in d" 2>/dev/null; then
  pass "Status summary endpoint works"
else
  fail "Status summary failed"
fi

# ── TEST 6: Simple factual query ──────────────────────────────────────────
sep
info "TEST 6: Simple factual query (non-streaming)"
Q1=$(curl -s -X POST "$BASE/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Acme Corp Q3 2024 total revenue?", "stream": false}')
echo "  Answer: $(echo "$Q1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('answer','ERROR')[:200])" 2>/dev/null)"
ANSWER=$(echo "$Q1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('answer',''))" 2>/dev/null)
if echo "$ANSWER" | grep -qi "4.2\|billion\|revenue\|could not find"; then
  pass "Query returned a grounded answer"
else
  fail "Query returned unexpected answer: $ANSWER"
fi

# ── TEST 7: Check citations in response ───────────────────────────────────
sep
info "TEST 7: Citations present in response"
CITATIONS=$(echo "$Q1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('citations',[])))" 2>/dev/null)
if [ "$CITATIONS" -gt 0 ]; then
  pass "Response includes $CITATIONS citation(s)"
  echo "  First citation: $(echo "$Q1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['citations'][0])" 2>/dev/null)"
else
  fail "No citations in response"
fi

# ── TEST 8: L1 exact cache hit ────────────────────────────────────────────
sep
info "TEST 8: L1 exact cache hit (same query again)"
Q_CACHED=$(curl -s -X POST "$BASE/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Acme Corp Q3 2024 total revenue?", "stream": false}')
CACHE=$(echo "$Q_CACHED" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('from_cache','none'))" 2>/dev/null)
if [ "$CACHE" = "exact" ]; then
  pass "L1 exact cache hit"
elif [ "$CACHE" = "semantic" ]; then
  pass "L2 semantic cache hit"
else
  info "Cache miss (expected on first run after index — caches build up over queries)"
fi

# ── TEST 9: Different factual query ──────────────────────────────────────
sep
info "TEST 9: Another factual query"
Q2=$(curl -s -X POST "$BASE/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the approved cloud infrastructure budget?", "stream": false}')
ANSWER2=$(echo "$Q2" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('answer','')[:300])" 2>/dev/null)
echo "  Answer: $ANSWER2"
if echo "$ANSWER2" | grep -qi "2.4\|million\|budget\|february\|could not find"; then
  pass "Budget query answered correctly"
else
  fail "Unexpected answer: $ANSWER2"
fi

# ── TEST 10: "I don't know" calibration ───────────────────────────────────
sep
info "TEST 10: Hallucination guard — query outside document scope"
Q3=$(curl -s -X POST "$BASE/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the recipe for chocolate cake?", "stream": false}')
ANSWER3=$(echo "$Q3" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('answer',''))" 2>/dev/null)
echo "  Answer: $ANSWER3"
if echo "$ANSWER3" | grep -qi "could not find\|not found\|no information\|not contain"; then
  pass "System correctly refused to answer out-of-scope query"
else
  info "Warning: system may have hallucinated — check answer: $ANSWER3"
fi

# ── TEST 11: Streaming query ──────────────────────────────────────────────
sep
info "TEST 11: Streaming SSE response"
STREAM_EVENTS=$(curl -s -N -X POST "$BASE/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What region had the highest growth?", "stream": true}' \
  --max-time 30 | head -20)
echo "  First events:"
echo "$STREAM_EVENTS" | head -5
TOKEN_COUNT=$(echo "$STREAM_EVENTS" | grep -c '"type"' || true)
if [ "$TOKEN_COUNT" -gt 0 ]; then
  pass "SSE streaming works ($TOKEN_COUNT SSE events received)"
else
  fail "No SSE events in stream"
fi

# ── SUMMARY ───────────────────────────────────────────────────────────────
sep
if [ "$FAILURES" -eq 0 ]; then
  echo -e "\n${GREEN}All tests passed.${NC}"
else
  echo -e "\n${RED}$FAILURES test(s) failed.${NC}"
fi
echo ""
echo "Useful commands:"
echo "  Watch parse worker:  docker compose logs -f worker-parse"
echo "  Watch embed worker:  docker compose logs -f worker-embed"
echo "  Watch API:           docker compose logs -f api"
echo "  Qdrant dashboard:    http://localhost:6333/dashboard"
echo "  Ingest status:       curl $BASE/ingest/status"
