SHELL := /bin/bash

EXAMPLES := pytorch tensorflow onnx python_backend tensorrt_llm vllm

.PHONY: help build run infer teardown $(foreach e,$(EXAMPLES),build-$(e) run-$(e) infer-$(e) teardown-$(e))

help:
	@echo "Usage: make <target>"
	@echo
	@echo "Global targets:"
	@echo "  build      Build all examples"
	@echo "  run        Run all examples"
	@echo "  infer      Run inference for all examples"
	@echo "  teardown   Stop/clean all examples"
	@echo
	@echo "Per-example targets:"
	@for e in $(EXAMPLES); do \
		echo "  build-$$e run-$$e infer-$$e teardown-$$e"; \
	done

build: $(foreach e,$(EXAMPLES),build-$(e))
run: $(foreach e,$(EXAMPLES),run-$(e))
infer: $(foreach e,$(EXAMPLES),infer-$(e))
teardown: $(foreach e,$(EXAMPLES),teardown-$(e))

define EXAMPLE_RULES
build-$(1):
	@echo "[build] $(1)"
	@echo "TODO: add build workflow for examples/$(1)"

run-$(1):
	@echo "[run] $(1)"
	@echo "TODO: add runtime workflow for examples/$(1)"

infer-$(1):
	@echo "[infer] $(1)"
	@echo "TODO: add inference workflow for examples/$(1)"

teardown-$(1):
	@echo "[teardown] $(1)"
	@echo "TODO: add cleanup workflow for examples/$(1)"
endef

$(foreach e,$(EXAMPLES),$(eval $(call EXAMPLE_RULES,$(e))))
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
