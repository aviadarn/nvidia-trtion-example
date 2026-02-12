import json
import os
import unittest
from typing import Iterable
from urllib import error, request

TRITON_URL = os.environ.get("TRITON_URL", "http://localhost:8000").rstrip("/")

EXAMPLE_MODELS = {
    "pytorch": "ensemble_pytorch",
    "tensorflow": "ensemble_tensorflow",
    "onnx": "ensemble_onnx",
    "python_backend": "ensemble_python",
    "tensorrt_llm": "ensemble_trtllm",
    "vllm": "ensemble_vllm",
}


def _selected_models() -> Iterable[tuple[str, str]]:
    raw = os.environ.get("TRITON_TEST_MODELS", "").strip()
    if not raw:
        return EXAMPLE_MODELS.items()

    requested = {part.strip() for part in raw.split(",") if part.strip()}
    selected: list[tuple[str, str]] = []
    for example, model in EXAMPLE_MODELS.items():
        if example in requested or model in requested:
            selected.append((example, model))
    return selected


def _http_json(method: str, url: str, body: dict | None = None) -> tuple[int, dict]:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=10) as resp:
            payload = resp.read().decode("utf-8").strip()
            return resp.status, json.loads(payload) if payload else {}
    except error.HTTPError as exc:
        payload = exc.read().decode("utf-8").strip()
        maybe_json = {}
        if payload:
            try:
                maybe_json = json.loads(payload)
            except json.JSONDecodeError:
                maybe_json = {"raw": payload}
        return exc.code, maybe_json


class TritonExampleEndpointsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.selected = list(_selected_models())
        if not cls.selected:
            raise unittest.SkipTest("No examples selected via TRITON_TEST_MODELS")

        try:
            with request.urlopen(f"{TRITON_URL}/v2/health/ready", timeout=5) as resp:
                if resp.status != 200:
                    raise unittest.SkipTest(
                        f"Triton is not ready at {TRITON_URL} (status={resp.status})"
                    )
        except Exception as exc:
            raise unittest.SkipTest(f"Triton is not reachable at {TRITON_URL}: {exc}") from exc

    def test_model_metadata_available_for_each_example(self) -> None:
        for example, model in self.selected:
            with self.subTest(example=example, model=model):
                status, payload = _http_json("GET", f"{TRITON_URL}/v2/models/{model}")
                self.assertEqual(
                    status,
                    200,
                    msg=f"Expected model metadata for {example}:{model}, got status={status} payload={payload}",
                )
                self.assertEqual(payload.get("name"), model)

    def test_inference_result_non_empty_for_each_example(self) -> None:
        for example, model in self.selected:
            with self.subTest(example=example, model=model):
                prompt = f"smoke:{example}"
                infer_payload = {
                    "inputs": [
                        {
                            "name": "RAW_TEXT",
                            "shape": [1, 1],
                            "datatype": "BYTES",
                            "data": [[prompt]],
                        }
                    ],
                    "outputs": [{"name": "FINAL_TEXT"}],
                }
                status, payload = _http_json(
                    "POST",
                    f"{TRITON_URL}/v2/models/{model}/infer",
                    body=infer_payload,
                )
                self.assertEqual(
                    status,
                    200,
                    msg=f"Inference failed for {example}:{model}; status={status} payload={payload}",
                )

                outputs = payload.get("outputs", [])
                self.assertTrue(outputs, msg=f"No outputs returned for {example}:{model}")
                final_text_output = next((item for item in outputs if item.get("name") == "FINAL_TEXT"), None)
                self.assertIsNotNone(
                    final_text_output,
                    msg=f"FINAL_TEXT output missing for {example}:{model}; outputs={outputs}",
                )
                returned_data = final_text_output.get("data")
                self.assertTrue(returned_data, msg=f"FINAL_TEXT data empty for {example}:{model}")


if __name__ == "__main__":
    unittest.main()
