#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="examples/vllm_postgres/docker-compose.yml"
PROMPT="Smoke test prompt $(date +%s)"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Expected compose file not found: $COMPOSE_FILE" >&2
  exit 1
fi

echo "[1/4] Starting stack"
docker compose -f "$COMPOSE_FILE" up -d

echo "[2/4] Running inference"
response=$(curl -sS http://localhost:8000/v2/models/ensemble/generate \
  -H 'Content-Type: application/json' \
  -d "{\"text_input\":\"$PROMPT\",\"max_tokens\":32,\"stream\":false}" ) || true

if [[ -z "${response// }" ]]; then
  echo "Inference returned empty response" >&2
  exit 1
fi

echo "[3/4] Waiting for sink write"
sleep 2

echo "[4/4] Validating Postgres record"
count=$(docker compose -f "$COMPOSE_FILE" exec -T postgres \
  psql -U app -d inference -t -A -c "SELECT COUNT(*) FROM public.inference_results WHERE prompt = '$PROMPT';" | tr -d '\r')

if [[ "$count" =~ ^[0-9]+$ ]] && (( count > 0 )); then
  echo "Smoke test passed: Postgres contains $count record(s) for prompt"
else
  echo "Smoke test failed: No Postgres record found for prompt" >&2
  exit 1
fi
