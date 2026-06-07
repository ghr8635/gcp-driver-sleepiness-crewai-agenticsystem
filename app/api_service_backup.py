import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from google.cloud import bigquery
from google.cloud import pubsub_v1
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

app = FastAPI(title="Driver Sleepiness Agent API")

bq_client = bigquery.Client(project=PROJECT_ID)
publisher = pubsub_v1.PublisherClient()

retriever = None

def get_retriever():
    global retriever
    if retriever is None:
        retriever = FaissVertexRAGRetriever(
            kb_path=str(ROOT_DIR / "data/intervention_knowledge_base.jsonl"),
            store_dir=str(ROOT_DIR / "data/faiss_vdb"),
            max_size=50,
            novelty_threshold=0.78,
        )
    return retriever


class DriverState(BaseModel):
    session_id: str
    timestamp: str
    blink_rate: float
    yawning_rate: float
    perclos: float
    sdlp: float
    lane_keeping_ratio: float
    lane_departure_frequency: float
    steering_entropy: float
    steering_reversal_rate: float
    steering_angle_variability: float
    camera_frame_id: Optional[str] = "api_request"
    sync_window_ms: Optional[int] = 50


class VisionMessage(BaseModel):
    session_id: str
    timestamp: str
    blink_rate: float
    yawning_rate: float
    perclos: float


class LaneMessage(BaseModel):
    session_id: str
    timestamp: str
    sdlp: float
    lane_keeping_ratio: float
    lane_departure_frequency: float


class SteeringMessage(BaseModel):
    session_id: str
    timestamp: str
    steering_entropy: float
    steering_reversal_rate: float
    steering_angle_variability: float


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


def insert_fatigue_feature(row: dict):
    table_id = f"{PROJECT_ID}.{DATASET}.fatigue_features"

    payload = [{
        "session_id": row["session_id"],
        "timestamp": row["timestamp"],
        "camera_frame_id": row.get("camera_frame_id", "api_request"),
        "sync_window_ms": int(row.get("sync_window_ms", 50)),
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

    errors = bq_client.insert_rows_json(table_id, payload)
    if errors:
        raise RuntimeError(f"BigQuery fatigue_features insert error: {errors}")


def insert_decision_log(row: dict, context: str, prompt: str, llm_output: str, parsed: dict, model_name: str, latency_ms: int):
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
        "prompt_version": "api-faiss-rag-v1",
        "max_length": 256,
        "latency_ms": latency_ms,
        "safety_flag": True,
    }]

    errors = bq_client.insert_rows_json(table_id, payload)
    if errors:
        raise RuntimeError(f"BigQuery agent_decision_logs insert error: {errors}")


def publish_to_topic(topic_name: str, payload: dict):
    topic_path = publisher.topic_path(PROJECT_ID, topic_name)
    future = publisher.publish(topic_path, data=json.dumps(payload).encode("utf-8"))
    return future.result()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "project_id": PROJECT_ID,
        "dataset": DATASET,
        "rag": "faiss_vertex_embeddings",
        "llm": "vertex_ai_gemini",
    }


@app.post("/predict")
def predict(driver_state: DriverState):
    row = estimate_fatigue(driver_state.model_dump())

    insert_fatigue_feature(row)

    start = time.perf_counter()
    rag = get_retriever()
    context, best_retrieval_score = rag.retrieve_with_scores(row, top_k=3)
    prompt = build_prompt(row, context)

    try:
        llm_output = vertex_gemini_llm(prompt)
        model_name = "gemini-2.5-flash-lite-vertex-ai"
    except Exception as e:
        llm_output = mock_llm(row)
        model_name = f"mock-llm-fallback: {type(e).__name__}"

    parsed = parse_llm_output(llm_output)
    latency_ms = int((time.perf_counter() - start) * 1000)
    vdb_update = rag.maybe_add_intervention(
        row=row,
        parsed=parsed,
        llm_output=llm_output,
        best_retrieval_score=best_retrieval_score,
    )

    print(f"VDB update status: {vdb_update}")

    insert_decision_log(
        row=row,
        context=context,
        prompt=prompt,
        llm_output=llm_output,
        parsed=parsed,
        model_name=model_name,
        latency_ms=latency_ms,
    )

    return {
    "session_id": row["session_id"],
    "risk_level": row["risk_level"],
    "fatigue_score": row["fatigue_score"],
    "fan_level": parsed["fan_level"],
    "music": parsed["music"],
    "vibration": parsed["vibration"],
    "reason": parsed["reason"],
    "model_name": model_name,
    "latency_ms": latency_ms,
    "best_retrieval_score": best_retrieval_score,
    "vdb_update": vdb_update,
}

@app.post("/publish/vision")
def publish_vision(message: VisionMessage):
    payload = {"stream": "vision", **message.model_dump()}
    message_id = publish_to_topic("vision-features-topic", payload)
    return {"status": "published", "topic": "vision-features-topic", "message_id": message_id}


@app.post("/publish/lane")
def publish_lane(message: LaneMessage):
    payload = {"stream": "lane", **message.model_dump()}
    message_id = publish_to_topic("lane-features-topic", payload)
    return {"status": "published", "topic": "lane-features-topic", "message_id": message_id}


@app.post("/publish/steering")
def publish_steering(message: SteeringMessage):
    payload = {"stream": "steering", **message.model_dump()}
    message_id = publish_to_topic("steering-features-topic", payload)
    return {"status": "published", "topic": "steering-features-topic", "message_id": message_id}
