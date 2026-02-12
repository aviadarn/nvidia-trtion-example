import json
import os
from kafka import KafkaConsumer
from pymongo import MongoClient


def main():
    topic = os.getenv("KAFKA_TOPIC", "triton.inference")
    consumer = KafkaConsumer(topic, bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "kafka:9092"), value_deserializer=lambda m: json.loads(m.decode("utf-8")))
    mongo = MongoClient(os.getenv("MONGODB_URI", "mongodb://mongodb:27017"))
    coll = mongo[os.getenv("MONGODB_DB", "triton")][os.getenv("MONGODB_COLLECTION", "inference_results")]
    for msg in consumer:
        coll.insert_one(msg.value)
        print(json.dumps({"status": "consumed", "request_id": msg.value.get("request_id")}))


if __name__ == "__main__":
    main()
