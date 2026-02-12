# vLLM + Postgres Example

This example runs a vLLM-backed Triton deployment and persists inference records to PostgreSQL.

## GPU prerequisites

- NVIDIA GPU with sufficient memory for chosen model and context length.
- For production performance, prefer data center GPUs (A10/A100/H100/L4/L40).
- NVIDIA driver/container toolkit compatible with selected CUDA image.
- Tested baseline:
  - Driver: `>= 535`
  - CUDA runtime/container: `12.x`
  - vLLM version matched with PyTorch/CUDA build in image.

## Known vLLM compatibility constraints

- vLLM support matrix is sensitive to CUDA + PyTorch + GPU architecture combinations.
- PagedAttention kernels and some quantization backends may require newer GPUs/toolchains.
- Tensor parallel and long context settings can cause OOM without reducing:
  - `max_model_len`,
  - batch size,
  - or enabling/switching quantization.
- Upgrading vLLM often changes memory planning behavior; re-baseline throughput and latency.

## Model artifact preparation

1. Pre-download model weights/tokenizer for offline startup:

   ```bash
   docker compose -f examples/vllm_postgres/docker-compose.yml run --rm vllm \
     python3 -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.1-8B-Instruct', local_dir='/models/Llama-3.1-8B-Instruct')"
   ```

2. Ensure model directory is mounted to the inference container.
3. Set runtime parameters (`tensor_parallel_size`, `max_model_len`, dtype) for your GPU memory profile.

## Docker Compose commands

Start services:

```bash
docker compose -f examples/vllm_postgres/docker-compose.yml up -d --build
```

Check health/logs:

```bash
docker compose -f examples/vllm_postgres/docker-compose.yml ps
docker compose -f examples/vllm_postgres/docker-compose.yml logs -f triton sink-writer postgres
```

Stop services:

```bash
docker compose -f examples/vllm_postgres/docker-compose.yml down -v
```

## Sample inference command

```bash
curl -sS http://localhost:8000/v2/models/ensemble/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text_input": "Write a two-line haiku about GPUs.",
    "max_tokens": 48,
    "temperature": 0.7,
    "stream": false
  }'
```

## Expected persisted records in sink (PostgreSQL)

After one successful inference, expect one row in `public.inference_results`:

| column | type | description |
|---|---|---|
| request_id | uuid/text | request correlation id |
| model | text | served model name |
| prompt | text | input prompt |
| completion | text | generated output |
| token_count | integer | generated token count |
| latency_ms | integer | end-to-end latency |
| created_at | timestamptz | insertion timestamp |

Quick verification:

```bash
docker compose -f examples/vllm_postgres/docker-compose.yml exec -T postgres \
  psql -U app -d inference -c "SELECT request_id, model, created_at FROM public.inference_results ORDER BY created_at DESC LIMIT 1;"
```

## Smoke test

Run:

```bash
scripts/smoke_vllm_postgres.sh
```
