import json
import os
from kafka import KafkaProducer


def publish(doc):
    bootstrap = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
    topic = os.getenv("KAFKA_TOPIC", "triton.inference")
    producer = KafkaProducer(bootstrap_servers=bootstrap, value_serializer=lambda v: json.dumps(v).encode("utf-8"))
    producer.send(topic, doc)
    producer.flush()
    print(json.dumps({"status": "published", "topic": topic}))


if __name__ == "__main__":
    payload = json.loads(os.getenv("PAYLOAD_JSON", "{}"))
    publish(payload)
