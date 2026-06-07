import json
import os
from google.cloud import pubsub_v1


def publish_json(topic_name: str, payload: dict):
    project_id = os.environ["PROJECT_ID"]
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_name)

    data = json.dumps(payload).encode("utf-8")
    future = publisher.publish(topic_path, data=data)

    message_id = future.result()
    print(f"Published to {topic_name}: session={payload.get('session_id')} message_id={message_id}")
