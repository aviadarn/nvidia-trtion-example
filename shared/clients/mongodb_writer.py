from __future__ import annotations

from typing import Any

from .schema import NormalizedInferenceResult


class MongoDBWriter:
    """Persists normalized inference results into MongoDB."""

    def __init__(self, mongo_uri: str, database: str, collection: str = "inference_results") -> None:
        try:
            from pymongo import MongoClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("MongoDB writer requires `pymongo`") from exc

        self._client = MongoClient(mongo_uri)
        self._collection = self._client[database][collection]
        self._collection.create_index("request_id", unique=True)
        self._collection.create_index("timestamp")

    def write(self, result: NormalizedInferenceResult | dict[str, Any]) -> None:
        payload = result.to_dict() if isinstance(result, NormalizedInferenceResult) else result
        self._collection.replace_one(
            {"request_id": payload["request_id"]},
            payload,
            upsert=True,
        )
import json
import os
from pymongo import MongoClient


def write(doc):
    uri = os.getenv("MONGODB_URI", "mongodb://mongodb:27017")
    db_name = os.getenv("MONGODB_DB", "triton")
    coll_name = os.getenv("MONGODB_COLLECTION", "inference_results")
    client = MongoClient(uri)
    coll = client[db_name][coll_name]
    coll.insert_one(doc)
    print(json.dumps({"status": "inserted", "collection": coll_name}))


if __name__ == "__main__":
    payload = json.loads(os.getenv("PAYLOAD_JSON", "{}"))
    write(payload)
