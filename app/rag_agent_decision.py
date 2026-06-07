"""
RAG + LLM-style decision layer.

This version follows the actual prompt style used in the driver sleepiness project.

It:
1. Reads processed fatigue rows from BigQuery.
2. Retrieves relevant safety context.
3. Builds a prompt with:
   - vision_features
   - lane_features
   - steering_features
   - fatigue_levels
4. Produces deterministic LLM-style output in the required format:
   Fan: Level X
   Music: On/Off
   Vibration: On/Off
   Reason: ...
5. Writes the decision log back to BigQuery.

Usage:
    python app/rag_agent_decision.py \
      --project YOUR_PROJECT \
      --dataset driver_sleepiness_ai
"""
import argparse
import os
import re
import time
import uuid
from pathlib import Path
from google.cloud import bigquery
from google import genai

GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID")
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")


MAX_LENGTH = 256


def load_guidelines(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    chunks = {}
    current_key = None
    buffer = []
    for line in text.splitlines():
        line_strip = line.strip()
        if not line_strip:
            continue
        if line_strip.endswith(":"):
            if current_key and buffer:
                chunks[current_key] = " ".join(buffer)
            current_key = line_strip.replace(":", "").lower()
            buffer = []
        else:
            buffer.append(line_strip)
    if current_key and buffer:
        chunks[current_key] = " ".join(buffer)
    return chunks


def retrieve_context(risk_level: str, guidelines: dict) -> str:
    key = {
        "low": "low fatigue",
        "medium": "medium fatigue",
        "high": "high fatigue",
    }.get(risk_level, "medium fatigue")
    return guidelines.get(key, "") + " " + guidelines.get("policy", "")


def build_prompt(row: dict, retrieved_context: str) -> str:
    """
    Mirrors the user's original prompt structure.
    The retrieved_context is added as safety policy context for the RAG part.
    """
    blink_rate = float(row["blink_rate"])
    yawning_rate = float(row["yawning_rate"])
    perclos = float(row["perclos"])
    sdlp = float(row["sdlp"])
    lane_keeping_ratio = float(row["lane_keeping_ratio"])
    lane_departure_freq = float(row["lane_departure_frequency"])
    steering_entropy = float(row["steering_entropy"])
    srr = float(row["steering_reversal_rate"])
    sav = float(row["steering_angle_variability"])

    fatigue_list = [
        row["fatigue_camera"],
        row["fatigue_steering"],
        row["fatigue_lane"],
    ]

    prompt = f"""
You are an intelligent in-cabin assistant. Based on the following driving behavior and fatigue indicators, generate an appropriate intervention to help the driver stay alert.

Strictly follow this format:
Fan: Level X
Music: On/Off
Vibration: On/Off
Reason: <short explanation of why this intervention is needed>

Example:
Fan: Level 2
Music: On
Vibration: Off
Reason: High PERCLOS and blinking suggest moderate fatigue.

<retrieved_safety_context>
{retrieved_context}
</retrieved_safety_context>

<vision_features>
blink_rate: {blink_rate:.1f} per minute
yawning_rate: {yawning_rate:.1f} per minute
perclos: {perclos:.2f}%
</vision_features>

<lane_features>
sdlp: {sdlp:.2f} m
lane_keeping_ratio: {lane_keeping_ratio:.1f}
lane_departure_frequency: {lane_departure_freq:.1f} per minute
</lane_features>

<steering_features>
steering_entropy: {steering_entropy:.1f}
steering_reversal_rate: {srr:.1f} per minute
steering_angle_variability: {sav:.2f}°
</steering_features>

<fatigue_levels>
fatigue_camera: {fatigue_list[0]}
fatigue_steering: {fatigue_list[1]}
fatigue_lane: {fatigue_list[2]}
</fatigue_levels>

<Expected Intervention>
""".strip()
    return prompt


def vertex_gemini_llm(prompt: str) -> str:
    """
    Real LLM call using Gemini through Vertex AI.
    The prompt already contains retrieved safety context and fatigue features.
    """
    if not GOOGLE_CLOUD_PROJECT:
        raise ValueError("GOOGLE_CLOUD_PROJECT is missing.")

    client = genai.Client(
        vertexai=True,
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )

    return response.text.strip()

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--dataset", default="driver_sleepiness_ai")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    guidelines = load_guidelines(root / "data/safety_guidelines.txt")
    client = bigquery.Client(project=args.project)

    source_table = f"`{args.project}.{args.dataset}.fatigue_features`"
    target_table = f"{args.project}.{args.dataset}.agent_decision_logs"

    query = f"""
    SELECT *
    FROM {source_table}
    ORDER BY timestamp
    LIMIT {args.limit}
    """
    rows = [dict(row) for row in client.query(query).result()]
    output_rows = []

    for row in rows:
        start = time.perf_counter()
        context = retrieve_context(row["risk_level"], guidelines)
        prompt = build_prompt(row, context)

        try:
            llm_output = vertex_gemini_llm(prompt)
            model_name = "gemini-2.5-flash-lite-vertex-ai"
        except Exception as e:
            print(f"Vertex AI call failed, falling back to mock LLM: {e}")
            llm_output = mock_llm(prompt, row["risk_level"], float(row["fatigue_score"]))
            model_name = "mock-llm-fallback"

        parsed = parse_llm_output(llm_output)
        latency_ms = int((time.perf_counter() - start) * 1000)

        output_rows.append({
            "request_id": str(uuid.uuid4()),
            "session_id": row["session_id"],
            "timestamp": row["timestamp"].isoformat(),
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
            "prompt_version": "driver-fatigue-original-format-v1",
            "max_length": MAX_LENGTH,
            "latency_ms": latency_ms,
            "safety_flag": True,
        })

    if not output_rows:
        print("No rows found in fatigue_features table. Run the Dataflow/Beam pipeline first.")
        return

    errors = client.insert_rows_json(target_table, output_rows)
    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors}")

    print(f"Inserted {len(output_rows)} agent decision logs into {target_table}")


if __name__ == "__main__":
    main()
