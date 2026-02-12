from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any
import uuid


@dataclass(slots=True)
class NormalizedInferenceResult:
    """Normalized schema for inference responses across transport layers."""

    request_id: str
    model: str
    latency_ms: float
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()



def build_normalized_result(
    *,
    model: str,
    latency_ms: float,
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    request_id: str | None = None,
    timestamp: str | None = None,
) -> NormalizedInferenceResult:
    return NormalizedInferenceResult(
        request_id=request_id or str(uuid.uuid4()),
        model=model,
        latency_ms=latency_ms,
        inputs=inputs,
        outputs=outputs,
        timestamp=timestamp or utc_timestamp(),
    )
