# NVIDIA Triton Example Workspace

This repository is organized around multiple Triton inference examples with shared client utilities and deployment assets.

## Repository layout

- `examples/pytorch/`
- `examples/tensorflow/`
- `examples/onnx/`
- `examples/python_backend/`
- `examples/tensorrt_llm/`
- `examples/vllm/`
- `shared/clients/`
- `deploy/`

## Example compatibility matrix

> These version targets are practical baseline recommendations. Pin exact image digests and framework wheels in each example as you implement them.

| Example | Backend / Runtime | Minimum GPU | CUDA | TensorRT | Triton Server Tag |
|---|---|---|---|---|---|
| `examples/pytorch` | PyTorch (LibTorch backend or ONNX export path) | NVIDIA T4 / A10 or newer | 12.2 | 10.0 | `24.08-py3` |
| `examples/tensorflow` | TensorFlow backend | NVIDIA T4 / A10 or newer | 12.2 | 10.0 | `24.08-py3` |
| `examples/onnx` | ONNX Runtime backend | NVIDIA T4 / A10 or newer | 12.2 | 10.0 | `24.08-py3` |
| `examples/python_backend` | Triton Python backend | NVIDIA T4 / A10 or newer | 12.2 | 10.0 | `24.08-py3` |
| `examples/tensorrt_llm` | TensorRT-LLM backend | NVIDIA A100 / H100 / L40S | 12.4 | 10.1+ | `24.10-trtllm-python-py3` |
| `examples/vllm` | vLLM backend integration | NVIDIA A100 / H100 / L40S | 12.4 | 10.1+ | `24.10-vllm-python-py3` |

## Quick start via Makefile

Each example has standard lifecycle targets:

- `build-<example>`
- `run-<example>`
- `infer-<example>`
- `teardown-<example>`

Where `<example>` is one of:

- `pytorch`
- `tensorflow`
- `onnx`
- `python_backend`
- `tensorrt_llm`
- `vllm`

You can also run grouped targets:

- `make build`
- `make run`
- `make infer`
- `make teardown`

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

## Example smoke tests

Use the endpoint smoke tests to verify each example environment comes up and inference returns non-empty output:

```bash
python3 -m unittest tests/test_example_endpoints.py
```

To automatically spin up Triton per example and run checks:

```bash
make test-examples
```

By default this runs `pytorch`, `tensorflow`, `onnx`, and `python_backend`.
To also include `tensorrt_llm` and `vllm`, set `RUN_LLM_EXAMPLES=1`:

```bash
RUN_LLM_EXAMPLES=1 make test-examples
```
