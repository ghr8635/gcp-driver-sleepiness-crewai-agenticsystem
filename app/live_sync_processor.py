import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from google.api_core.exceptions import DeadlineExceeded

from google.cloud import pubsub_v1
from google.cloud import bigquery
from google import genai

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.fatigue_logic import estimate_fatigue
from app.faiss_rag_retriever import FaissVertexRAGRetriever


PROJECT_ID = os.environ["PROJECT_ID"]
DATASET = os.environ.get("DATASET", "driver_sleepiness_ai")
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or PROJECT_ID
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

VISION_SUB = "vision-features-sub"
LANE_SUB = "lane-features-sub"
STEERING_SUB = "steering-features-sub"

MAX_MESSAGES_PER_PULL = 10


def parse_llm_output(text: str) -> dict:
    fan_match = re.search(r"Fan:\s*Level\s*(\d+)", text, re.IGNORECASE)
    music_match = re.search(r"Music:\s*(On|Off)", text, re.IGNORECASE)
    vibration_match = re.search(r"Vibration:\s*(On|Off)", text, re.IGNORECASE)
    reason_match = re.search(r"Reason:\s*(.*)", text, re.IGNORECASE | re.DOTALL)

    return {
        "fan_level": int(fan_match.group(1)) if fan_match else None,
        "music": music_match.group(1).capitalize() if music_match else None,
        "vibration": vibration_match.group(1).capitalize() if vibration_match else None,
        "reason": reason_match.group(1).strip() if reason_match else None,
    }


def build_prompt(row: dict, retrieved_context: str) -> str:
    return f"""
You are an intelligent in-cabin assistant. Based on the following driving behavior and fatigue indicators, generate an appropriate intervention to help the driver stay alert.

Strictly follow this format:
Fan: Level X
Music: On/Off
Vibration: On/Off
Reason: <short explanation of why this intervention is needed>

Return only the following four lines.
Do not use markdown.
Do not add extra explanation.
Do not add bullet points.

<retrieved_safety_context>
{retrieved_context}
</retrieved_safety_context>

<vision_features>
blink_rate: {float(row['blink_rate']):.1f} per minute
yawning_rate: {float(row['yawning_rate']):.1f} per minute
perclos: {float(row['perclos']):.2f}%
</vision_features>

<lane_features>
sdlp: {float(row['sdlp']):.2f} m
lane_keeping_ratio: {float(row['lane_keeping_ratio']):.1f}
lane_departure_frequency: {float(row['lane_departure_frequency']):.1f} per minute
</lane_features>

<steering_features>
steering_entropy: {float(row['steering_entropy']):.1f}
steering_reversal_rate: {float(row['steering_reversal_rate']):.1f} per minute
steering_angle_variability: {float(row['steering_angle_variability']):.2f}°
</steering_features>

<fatigue_levels>
fatigue_camera: {row['fatigue_camera']}
fatigue_steering: {row['fatigue_steering']}
fatigue_lane: {row['fatigue_lane']}
</fatigue_levels>

<Expected Intervention>
""".strip()


def vertex_gemini_llm(prompt: str) -> str:
    client = genai.Client(
        vertexai=True,
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
    )

    last_error = None

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            last_error = e
            wait_time = min(2 ** attempt, 16)
            print(f"Gemini call failed on attempt {attempt + 1}/5: {e}")
            print(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)

    raise last_error


def mock_llm(row: dict) -> str:
    risk_level = row["risk_level"]

    if risk_level == "low":
        return (
            "Fan: Level 1\n"
            "Music: Off\n"
            "Vibration: Off\n"
            "Reason: Fatigue indicators are low, so passive monitoring is enough."
        )

    if risk_level == "medium":
        return (
            "Fan: Level 2\n"
            "Music: On\n"
            "Vibration: Off\n"
            "Reason: Moderate fatigue indicators suggest increasing driver fatigue."
        )

    return (
        "Fan: Level 3\n"
        "Music: On\n"
        "Vibration: On\n"
        "Reason: High fatigue indicators require stronger alert intervention."
    )


def insert_fatigue_feature(client: bigquery.Client, row: dict):
    table_id = f"{PROJECT_ID}.{DATASET}.fatigue_features"

    payload = [{
        "session_id": row["session_id"],
        "timestamp": row["timestamp"],
        "camera_frame_id": row.get("camera_frame_id", "live_fake"),
        "sync_window_ms": int(row.get("sync_window_ms", 0)),
        "blink_rate": float(row["blink_rate"]),
        "yawning_rate": float(row["yawning_rate"]),
        "perclos": float(row["perclos"]),
        "sdlp": float(row["sdlp"]),
        "lane_keeping_ratio": float(row["lane_keeping_ratio"]),
        "lane_departure_frequency": float(row["lane_departure_frequency"]),
        "steering_entropy": float(row["steering_entropy"]),
        "steering_reversal_rate": float(row["steering_reversal_rate"]),
        "steering_angle_variability": float(row["steering_angle_variability"]),
        "fatigue_camera": row["fatigue_camera"],
        "fatigue_steering": row["fatigue_steering"],
        "fatigue_lane": row["fatigue_lane"],
        "fatigue_score": float(row["fatigue_score"]),
        "risk_level": row["risk_level"],
    }]

    errors = client.insert_rows_json(table_id, payload)
    if errors:
        raise RuntimeError(f"BigQuery fatigue_features insert error: {errors}")


def insert_decision_log(client: bigquery.Client, row: dict, context: str, prompt: str, llm_output: str, parsed: dict, model_name: str, latency_ms: int):
    table_id = f"{PROJECT_ID}.{DATASET}.agent_decision_logs"

    payload = [{
        "request_id": str(uuid.uuid4()),
        "session_id": row["session_id"],
        "timestamp": row["timestamp"],
        "risk_level": row["risk_level"],
        "fatigue_score": float(row["fatigue_score"]),
        "retrieved_context": context,
        "prompt": prompt,
        "llm_output": llm_output,
        "fan_level": parsed["fan_level"],
        "music": parsed["music"],
        "vibration": parsed["vibration"],
        "reason": parsed["reason"],
        "model_name": model_name,
        "prompt_version": "live-pubsub-faiss-rag-v1",
        "max_length": 256,
        "latency_ms": latency_ms,
        "safety_flag": True,
    }]

    errors = client.insert_rows_json(table_id, payload)
    if errors:
        raise RuntimeError(f"BigQuery agent_decision_logs insert error: {errors}")


def pull_messages(subscriber, subscription_path):
    messages = []

    try:
        response = subscriber.pull(
            request={
                "subscription": subscription_path,
                "max_messages": MAX_MESSAGES_PER_PULL,
            },
            timeout=10,
        )
    except DeadlineExceeded:
        return messages
    except Exception as e:
        print(f"Pub/Sub pull warning for {subscription_path}: {e}")
        return messages

    if response.received_messages:
        ack_ids = []

        for received in response.received_messages:
            try:
                data = json.loads(received.message.data.decode("utf-8"))
                messages.append(data)
                ack_ids.append(received.ack_id)
            except Exception as e:
                print(f"Failed to parse message: {e}")

        if ack_ids:
            subscriber.acknowledge(
                request={
                    "subscription": subscription_path,
                    "ack_ids": ack_ids,
                }
            )

    return messages


def main():
    print("Starting live sync processor...")
    print("This simulates ROS2 sync using three Pub/Sub streams: vision, lane, steering.")

    subscriber = pubsub_v1.SubscriberClient()
    bq_client = bigquery.Client(project=PROJECT_ID)

    root = Path(__file__).resolve().parents[1]
    retriever = FaissVertexRAGRetriever(
        kb_path=str(root / "data/intervention_knowledge_base.jsonl"),
        store_dir=str(root / "data/faiss_vdb"),
        max_size=50,
        novelty_threshold=0.78,
    )

    subscriptions = {
        "vision": subscriber.subscription_path(PROJECT_ID, VISION_SUB),
        "lane": subscriber.subscription_path(PROJECT_ID, LANE_SUB),
        "steering": subscriber.subscription_path(PROJECT_ID, STEERING_SUB),
    }

    buffer = {}

    while True:
        total_pulled = 0

        for stream_name, sub_path in subscriptions.items():
            messages = pull_messages(subscriber, sub_path)
            total_pulled += len(messages)

            for msg in messages:
                session_id = msg["session_id"]
                buffer.setdefault(session_id, {})
                buffer[session_id][stream_name] = msg
                print(f"Received {stream_name} message for {session_id}")

        ready_sessions = [
            sid for sid, parts in buffer.items()
            if all(k in parts for k in ["vision", "lane", "steering"])
        ]

        for session_id in ready_sessions:
            parts = buffer.pop(session_id)

            combined = {
                "session_id": session_id,
                "timestamp": parts["vision"]["timestamp"],
                "sync_window_ms": 50,
                **{k: v for k, v in parts["vision"].items() if k not in ["stream", "session_id", "timestamp"]},
                **{k: v for k, v in parts["lane"].items() if k not in ["stream", "session_id", "timestamp"]},
                **{k: v for k, v in parts["steering"].items() if k not in ["stream", "session_id", "timestamp"]},
            }

            print(f"\nSynced complete driver state for {session_id}")

            row = estimate_fatigue(combined)
            insert_fatigue_feature(bq_client, row)

            start = time.perf_counter()
            context, best_retrieval_score = retriever.retrieve_with_scores(row, top_k=3)
            prompt = build_prompt(row, context)

            try:
                llm_output = vertex_gemini_llm(prompt)
                model_name = "gemini-2.5-flash-lite-vertex-ai"
            except Exception as e:
                print(f"Vertex AI Gemini failed. Using fallback. Error: {e}")
                llm_output = mock_llm(row)
                model_name = "mock-llm-fallback"

            parsed = parse_llm_output(llm_output)
            latency_ms = int((time.perf_counter() - start) * 1000)
            vdb_update = retriever.maybe_add_intervention(
                row=row,
                parsed=parsed,
                llm_output=llm_output,
                best_retrieval_score=best_retrieval_score,
            )

            print(f"VDB update status: {vdb_update}")

            insert_decision_log(
                bq_client,
                row=row,
                context=context,
                prompt=prompt,
                llm_output=llm_output,
                parsed=parsed,
                model_name=model_name,
                latency_ms=latency_ms,
            )

            print(f"Intervention for {session_id}:")
            print(llm_output)
            print(f"Saved to BigQuery. Model: {model_name}, latency_ms={latency_ms}\n")

            time.sleep(8)

        if total_pulled == 0:
            print("No new messages. Waiting...")
            time.sleep(3)


if __name__ == "__main__":
    main()
