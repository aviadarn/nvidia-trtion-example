from __future__ import annotations

import json
import logging
import os
from typing import Any

from shared.clients.mongodb_writer import MongoDBWriter
from shared.clients.postgres_writer import PostgresWriter

logger = logging.getLogger(__name__)


class InferenceResultConsumer:
    """Consumes Kafka inference events and fans out to storage sinks."""

    def __init__(
        self,
        *,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        mongo_writer: MongoDBWriter | None = None,
        postgres_writer: PostgresWriter | None = None,
    ) -> None:
        try:
            from kafka import KafkaConsumer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Kafka consumer requires `kafka-python`") from exc

        self._consumer = KafkaConsumer(
            topic,
            bootstrap_servers=[s.strip() for s in bootstrap_servers.split(",")],
            group_id=group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        )
        self._mongo_writer = mongo_writer
        self._postgres_writer = postgres_writer

    def run(self) -> None:
        logger.info("Starting inference result consumer")
        for message in self._consumer:
            payload = message.value
            self._write(payload)

    def _write(self, payload: dict[str, Any]) -> None:
        if self._mongo_writer is not None:
            self._mongo_writer.write(payload)
        if self._postgres_writer is not None:
            self._postgres_writer.write(payload)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_TOPIC", "triton.inference.results")
    group_id = os.getenv("KAFKA_GROUP_ID", "inference-results-consumer")

    mongo_writer = None
    if _env_bool("MONGO_ENABLED", True):
        mongo_writer = MongoDBWriter(
            mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            database=os.getenv("MONGO_DATABASE", "triton"),
            collection=os.getenv("MONGO_COLLECTION", "inference_results"),
        )

    postgres_writer = None
    if _env_bool("POSTGRES_ENABLED", True):
        postgres_writer = PostgresWriter(
            dsn=os.getenv(
                "POSTGRES_DSN",
                "dbname=triton user=postgres password=postgres host=localhost port=5432",
            ),
            table=os.getenv("POSTGRES_TABLE", "inference_results"),
        )

    consumer = InferenceResultConsumer(
        bootstrap_servers=bootstrap,
        topic=topic,
        group_id=group_id,
        mongo_writer=mongo_writer,
        postgres_writer=postgres_writer,
    )
    consumer.run()


if __name__ == "__main__":
    main()
