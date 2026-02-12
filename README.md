# NVIDIA Triton End-to-End Examples Monorepo

This repository provides practical Triton Inference Server examples for:

- PyTorch backend
- TensorFlow backend
- ONNX Runtime backend
- Python backend
- TensorRT-LLM backend
- vLLM backend

All examples include **ensemble models** and deployment examples with result persistence to:

- MongoDB
- PostgreSQL
- Kafka

## Repository layout

- `examples/*/model_repository`: Triton model repositories per backend.
- `shared/clients`: Python request + sink adapter utilities.
- `deploy`: Docker Compose stacks for Triton + sink(s).
- `scripts`: Basic smoke checks and sink verification commands.

## Quick start

```bash
make run-mongodb
make infer SAMPLE_MODEL=ensemble_pytorch
```

## Notes

- This scaffold is designed to be extended with real model artifacts.
- `TensorRT-LLM` and `vLLM` examples include deployment/config placeholders because exact engines/tokenizers vary by model.
