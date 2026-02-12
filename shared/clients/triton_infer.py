import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import tritonclient.http as httpclient


def run(model: str, text: str, url: str = "localhost:8000"):
    client = httpclient.InferenceServerClient(url=url)
    infer_input = httpclient.InferInput("RAW_TEXT", [1, 1], "BYTES")
    infer_input.set_data_from_numpy(np.array([[text.encode("utf-8")]], dtype=object))
    requested_output = httpclient.InferRequestedOutput("FINAL_TEXT")

    t0 = time.time()
    result = client.infer(model_name=model, inputs=[infer_input], outputs=[requested_output])
    latency_ms = (time.time() - t0) * 1000

    out = result.as_numpy("FINAL_TEXT")
    decoded = out[0][0].decode("utf-8") if hasattr(out[0][0], "decode") else str(out[0][0])
    payload = {
        "request_id": f"req-{int(time.time() * 1000)}",
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": round(latency_ms, 3),
        "inputs": {"RAW_TEXT": text},
        "outputs": {"FINAL_TEXT": decoded},
    }
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ensemble_pytorch")
    parser.add_argument("--text", default="hello")
    parser.add_argument("--url", default="localhost:8000")
    args = parser.parse_args()
    run(args.model, args.text, args.url)
