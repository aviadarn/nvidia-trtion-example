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
