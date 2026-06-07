import json
import time
from pathlib import Path

from app.pubsub_publish_helper import publish_json


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data/live_fake_sequence.json"

    samples = json.loads(data_path.read_text(encoding="utf-8"))

    for sample in samples:
        payload = {
            "stream": "steering",
            "session_id": sample["session_id"],
            "timestamp": sample["timestamp"],
            **sample["steering"],
        }
        publish_json("steering-features-topic", payload)
        time.sleep(1)


if __name__ == "__main__":
    main()
