import json
import os
import psycopg


def write(doc):
    dsn = os.getenv("POSTGRES_DSN", "postgresql://postgres:postgres@postgres:5432/triton")
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS inference_results (
                    request_id TEXT PRIMARY KEY,
                    model TEXT,
                    timestamp TEXT,
                    latency_ms DOUBLE PRECISION,
                    payload JSONB
                )
                """
            )
            cur.execute(
                """
                INSERT INTO inference_results(request_id, model, timestamp, latency_ms, payload)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (request_id) DO NOTHING
                """,
                (
                    doc.get("request_id"),
                    doc.get("model"),
                    doc.get("timestamp"),
                    doc.get("latency_ms"),
                    json.dumps(doc),
                ),
            )
        conn.commit()
    print(json.dumps({"status": "inserted", "table": "inference_results"}))


if __name__ == "__main__":
    payload = json.loads(os.getenv("PAYLOAD_JSON", "{}"))
    write(payload)
