# NVIDIA Triton model repository examples

This repository provides example `model_repository/` layouts for common Triton backends.

## Included examples

- `examples/pytorch`
- `examples/tensorflow`
- `examples/onnx`
- `examples/python`
- `examples/tensorrt_llm`
- `examples/vllm`

Each example includes the Triton-required structure:

- `model_repository/<model_name>/1/` for versioned model artifacts
- `model_repository/<model_name>/config.pbtxt` for model metadata and backend configuration
