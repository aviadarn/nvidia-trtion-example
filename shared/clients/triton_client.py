from __future__ import annotations

import json
import time
from typing import Any, Iterable

import requests

from .schema import NormalizedInferenceResult, build_normalized_result


class TritonRequestClient:
    """Utility for running inference against NVIDIA Triton over HTTP or gRPC."""

    def __init__(self, endpoint: str, timeout_s: float = 30.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = timeout_s

    def infer_http(
        self,
        *,
        model: str,
        inputs: list[dict[str, Any]],
        outputs: Iterable[str] | None = None,
        request_id: str | None = None,
    ) -> NormalizedInferenceResult:
        payload: dict[str, Any] = {
            "id": request_id,
            "inputs": inputs,
        }
        if outputs:
            payload["outputs"] = [{"name": name} for name in outputs]

        url = f"{self.endpoint}/v2/models/{model}/infer"
        start = time.perf_counter()
        response = requests.post(url, json=payload, timeout=self.timeout_s)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        body = response.json()

        normalized_outputs = body.get("outputs", [])

        return build_normalized_result(
            request_id=body.get("id") or request_id,
            model=model,
            latency_ms=elapsed_ms,
            inputs=inputs,
            outputs=normalized_outputs,
        )

    def infer_grpc(
        self,
        *,
        model: str,
        inputs: list[dict[str, Any]],
        outputs: Iterable[str] | None = None,
        request_id: str | None = None,
    ) -> NormalizedInferenceResult:
        """Run inference through tritonclient.grpc.

        Requires `tritonclient[grpc]` to be installed.
        """
        try:
            import numpy as np
            from tritonclient.grpc import InferInput, InferRequestedOutput, InferenceServerClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "gRPC inference requires optional dependency `tritonclient[grpc]`"
            ) from exc

        target = self.endpoint.removeprefix("http://").removeprefix("https://")
        client = InferenceServerClient(url=target)

        grpc_inputs = []
        for item in inputs:
            infer_input = InferInput(item["name"], item["shape"], item["datatype"])
            infer_input.set_data_from_numpy(np.asarray(item["data"]))
            grpc_inputs.append(infer_input)

        grpc_outputs = (
            [InferRequestedOutput(name) for name in outputs] if outputs else None
        )

        start = time.perf_counter()
        result = client.infer(
            model_name=model,
            inputs=grpc_inputs,
            outputs=grpc_outputs,
            request_id=request_id,
            client_timeout=self.timeout_s,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        normalized_outputs = []
        output_names = outputs or result.get_response(as_json=True).get("outputs", [])
        if isinstance(output_names, list) and output_names and isinstance(output_names[0], dict):
            output_names = [item["name"] for item in output_names]

        for name in output_names:
            np_data = result.as_numpy(name)
            normalized_outputs.append(
                {
                    "name": name,
                    "datatype": str(np_data.dtype),
                    "shape": list(np_data.shape),
                    "data": np_data.tolist(),
                }
            )

        return build_normalized_result(
            request_id=request_id,
            model=model,
            latency_ms=elapsed_ms,
            inputs=inputs,
            outputs=normalized_outputs,
        )

    @staticmethod
    def to_json(result: NormalizedInferenceResult) -> str:
        return json.dumps(result.to_dict(), separators=(",", ":"))
