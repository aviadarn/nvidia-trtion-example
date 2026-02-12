#!/usr/bin/env bash
set -euo pipefail

echo "[smoke] checking compose files"
docker compose -f deploy/docker-compose.mongodb.yml config >/dev/null
docker compose -f deploy/docker-compose.postgres.yml config >/dev/null
docker compose -f deploy/docker-compose.kafka.yml config >/dev/null

echo "[smoke] static checks done"
