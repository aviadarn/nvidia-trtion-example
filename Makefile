PYTHON ?= python3

.PHONY: run-mongodb run-postgres run-kafka down infer smoke

run-mongodb:
	docker compose -f deploy/docker-compose.mongodb.yml up -d --build

run-postgres:
	docker compose -f deploy/docker-compose.postgres.yml up -d --build

run-kafka:
	docker compose -f deploy/docker-compose.kafka.yml up -d --build

down:
	docker compose -f deploy/docker-compose.mongodb.yml down -v || true
	docker compose -f deploy/docker-compose.postgres.yml down -v || true
	docker compose -f deploy/docker-compose.kafka.yml down -v || true

infer:
	$(PYTHON) shared/clients/triton_infer.py --model $${SAMPLE_MODEL:-ensemble_pytorch} --text "hello triton"

smoke:
	bash scripts/smoke_all.sh
