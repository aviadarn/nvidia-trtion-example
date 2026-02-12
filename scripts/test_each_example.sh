#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-8000}"

run_one() {
  local example="$1"
  local model="$2"
  local image="$3"
  local name="triton-test-${example}"

  echo "[test:${example}] starting Triton container"
  docker rm -f "$name" >/dev/null 2>&1 || true

  docker run -d --rm \
    --name "$name" \
    -p "${TRITON_HTTP_PORT}:8000" \
    -v "$ROOT_DIR/examples/${example}/model_repository:/models:ro" \
    "$image" \
    tritonserver --model-repository=/models --strict-model-config=false >/dev/null

  cleanup() {
    docker rm -f "$name" >/dev/null 2>&1 || true
  }
  trap cleanup RETURN

  echo "[test:${example}] waiting for readiness"
  for _ in $(seq 1 60); do
    if curl -fsS "http://localhost:${TRITON_HTTP_PORT}/v2/health/ready" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done

  echo "[test:${example}] running endpoint/inference assertions"
  TRITON_URL="http://localhost:${TRITON_HTTP_PORT}" TRITON_TEST_MODELS="$model" \
    python3 -m unittest tests/test_example_endpoints.py

  cleanup
  trap - RETURN
}

run_one pytorch ensemble_pytorch nvcr.io/nvidia/tritonserver:24.08-py3
run_one tensorflow ensemble_tensorflow nvcr.io/nvidia/tritonserver:24.08-py3
run_one onnx ensemble_onnx nvcr.io/nvidia/tritonserver:24.08-py3
run_one python_backend ensemble_python nvcr.io/nvidia/tritonserver:24.08-py3

if [[ "${RUN_LLM_EXAMPLES:-0}" == "1" ]]; then
  run_one tensorrt_llm ensemble_trtllm nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3
  run_one vllm ensemble_vllm nvcr.io/nvidia/tritonserver:24.10-vllm-python-py3
else
  echo "[test] Skipping LLM examples. Set RUN_LLM_EXAMPLES=1 to include tensorrt_llm and vllm."
fi
