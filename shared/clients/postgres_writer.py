from __future__ import annotations

import json
from typing import Any

from .schema import NormalizedInferenceResult


class PostgresWriter:
    """Persists normalized inference results into PostgreSQL."""

    def __init__(self, dsn: str, table: str = "inference_results") -> None:
        try:
            import psycopg2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Postgres writer requires `psycopg2`") from exc

        self._conn = psycopg2.connect(dsn)
        self._table = table
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn, self._conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    request_id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    latency_ms DOUBLE PRECISION NOT NULL,
                    inputs JSONB NOT NULL,
                    outputs JSONB NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL
                )
                """
            )

    def write(self, result: NormalizedInferenceResult | dict[str, Any]) -> None:
        payload = result.to_dict() if isinstance(result, NormalizedInferenceResult) else result

        with self._conn, self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self._table}
                    (request_id, model, latency_ms, inputs, outputs, timestamp)
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::timestamptz)
                ON CONFLICT (request_id)
                DO UPDATE SET
                    model = EXCLUDED.model,
                    latency_ms = EXCLUDED.latency_ms,
                    inputs = EXCLUDED.inputs,
                    outputs = EXCLUDED.outputs,
                    timestamp = EXCLUDED.timestamp
                """,
                (
                    payload["request_id"],
                    payload["model"],
                    payload["latency_ms"],
                    json.dumps(payload["inputs"]),
                    json.dumps(payload["outputs"]),
                    payload["timestamp"],
                ),
            )
