from __future__ import annotations

import json
from typing import Any

from .schema import NormalizedInferenceResult


class KafkaProducerAdapter:
    """Publishes inference results as JSON events to Kafka."""

    def __init__(self, bootstrap_servers: str, topic: str) -> None:
        try:
            from kafka import KafkaProducer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Kafka producer requires `kafka-python`") from exc

        self._topic = topic
        self._producer = KafkaProducer(
            bootstrap_servers=[s.strip() for s in bootstrap_servers.split(",")],
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            key_serializer=lambda key: key.encode("utf-8") if key else None,
        )

    def send(self, result: NormalizedInferenceResult | dict[str, Any]) -> None:
        payload = result.to_dict() if isinstance(result, NormalizedInferenceResult) else result
        self._producer.send(
            topic=self._topic,
            key=payload.get("request_id"),
            value=payload,
        )
        self._producer.flush()
