# TensorRT-LLM + MongoDB Example

This example runs a TensorRT-LLM-backed Triton deployment and persists inference records to MongoDB.

## GPU prerequisites

- NVIDIA GPU with **SM80+** (A100/H100/L40/L4 class recommended).
- NVIDIA driver and container runtime compatible with your CUDA stack.
- Tested baseline:
  - Driver: `>= 535`
  - CUDA runtime/container: `12.x`
  - TensorRT-LLM image aligned with Triton server release.
- Ensure MIG profiles expose enough memory for the selected model if MIG is enabled.

## Known TensorRT-LLM compatibility constraints

- TensorRT-LLM engines are tightly coupled to:
  - GPU architecture (SM capability),
  - TensorRT-LLM/TensorRT versions,
  - precision and plugin set used during engine build.
- Engines built on one architecture (for example SM90) are generally not portable to another (for example SM80).
- Quantization support depends on both TensorRT-LLM version and model architecture.
- If you upgrade Triton/TensorRT-LLM images, rebuild engines and revalidate outputs.

## Model artifact preparation

1. Download model weights (example uses Llama-family checkpoint) into `./models/input/`.
2. Convert checkpoint to TensorRT-LLM checkpoint format:

   ```bash
   docker compose -f examples/trtllm_mongodb/docker-compose.yml run --rm trtllm-builder \
     python3 /opt/scripts/convert_checkpoint.py \
       --input_dir /workspace/models/input \
       --output_dir /workspace/models/converted
   ```

3. Build TensorRT-LLM engine artifacts:

   ```bash
   docker compose -f examples/trtllm_mongodb/docker-compose.yml run --rm trtllm-builder \
     trtllm-build \
       --checkpoint_dir /workspace/models/converted \
       --output_dir /workspace/model_repository/tensorrt_llm/1 \
       --max_batch_size 8 \
       --max_input_len 2048 \
       --max_output_len 512
   ```

4. Verify model repository contains a TensorRT-LLM version directory and `config.pbtxt`.

## Docker Compose commands

Start services:

```bash
docker compose -f examples/trtllm_mongodb/docker-compose.yml up -d --build
```

Check health/logs:

```bash
docker compose -f examples/trtllm_mongodb/docker-compose.yml ps
docker compose -f examples/trtllm_mongodb/docker-compose.yml logs -f triton sink-writer mongo
```

Stop services:

```bash
docker compose -f examples/trtllm_mongodb/docker-compose.yml down -v
```

## Sample inference command

```bash
curl -sS http://localhost:8000/v2/models/ensemble/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text_input": "Summarize Triton Inference Server in one sentence.",
    "max_tokens": 64,
    "temperature": 0.2,
    "stream": false
  }'
```

## Expected persisted records in sink (MongoDB)

After one successful inference, expect one new document in `inference.results`, similar to:

```json
{
  "request_id": "<uuid>",
  "model": "ensemble",
  "prompt": "Summarize Triton Inference Server in one sentence.",
  "completion": "...",
  "token_count": 42,
  "latency_ms": 123,
  "created_at": "2026-01-01T00:00:00Z"
}
```

Quick verification:

```bash
docker compose -f examples/trtllm_mongodb/docker-compose.yml exec -T mongo \
  mongosh --quiet --eval 'db.getSiblingDB("inference").results.find().sort({created_at:-1}).limit(1).pretty()'
```

## Smoke test

Run:

```bash
scripts/smoke_trtllm_mongodb.sh
```
