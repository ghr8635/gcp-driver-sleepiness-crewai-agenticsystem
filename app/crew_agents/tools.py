import time

from app.fatigue_logic import estimate_fatigue
from app.services import (
    get_retriever,
    build_prompt,
    vertex_gemini_llm,
    mock_llm,
    parse_llm_output,
    insert_fatigue_feature,
    insert_decision_log,
)


def calculate_fatigue(driver_state: dict) -> dict:
    row = estimate_fatigue(driver_state)
    return row


def retrieve_rag_context(row: dict) -> dict:
    rag = get_retriever()
    context, best_retrieval_score = rag.retrieve_with_scores(row, top_k=3)

    return {
        "context": context,
        "best_retrieval_score": best_retrieval_score,
    }


def generate_intervention(row: dict, context: str) -> dict:
    prompt = build_prompt(row, context)

    try:
        llm_output = vertex_gemini_llm(prompt)
        model_name = "gemini-2.5-flash-lite-vertex-ai"
    except Exception as e:
        llm_output = mock_llm(row)
        model_name = f"mock-llm-fallback: {type(e).__name__}"

    parsed = parse_llm_output(llm_output)

    return {
        "prompt": prompt,
        "llm_output": llm_output,
        "parsed": parsed,
        "model_name": model_name,
    }


def validate_intervention(row: dict, parsed: dict) -> dict:
    risk_level = row["risk_level"]

    fan_level = parsed.get("fan_level")
    music = parsed.get("music")
    vibration = parsed.get("vibration")
    reason = parsed.get("reason")

    if fan_level not in [1, 2, 3]:
        fan_level = None

    if music not in ["On", "Off"]:
        music = None

    if vibration not in ["On", "Off"]:
        vibration = None

    if not reason:
        reason = "Fallback intervention applied due to invalid or incomplete model output."

    if fan_level is None or music is None or vibration is None:
        if risk_level == "low":
            fan_level = 1
            music = "Off"
            vibration = "Off"
        elif risk_level == "medium":
            fan_level = 2
            music = "On"
            vibration = "Off"
        else:
            fan_level = 3
            music = "On"
            vibration = "On"

    if risk_level == "high" and fan_level < 3:
        fan_level = 3
        music = "On"
        vibration = "On"
        reason = "High fatigue risk detected, so stronger alert intervention was enforced."

    return {
        "fan_level": fan_level,
        "music": music,
        "vibration": vibration,
        "reason": reason,
    }


def update_vector_memory(
    row: dict,
    parsed: dict,
    llm_output: str,
    best_retrieval_score: float,
) -> dict:
    rag = get_retriever()

    vdb_update = rag.maybe_add_intervention(
        row=row,
        parsed=parsed,
        llm_output=llm_output,
        best_retrieval_score=best_retrieval_score,
    )

    return vdb_update


def log_prediction(
    row: dict,
    context: str,
    prompt: str,
    llm_output: str,
    parsed: dict,
    model_name: str,
    latency_ms: int,
):
    insert_fatigue_feature(row)

    insert_decision_log(
        row=row,
        context=context,
        prompt=prompt,
        llm_output=llm_output,
        parsed=parsed,
        model_name=model_name,
        latency_ms=latency_ms,
    )


def run_tool_pipeline(driver_state: dict) -> dict:
    start = time.perf_counter()

    agent_progress = []

    row = calculate_fatigue(driver_state)
    agent_progress.append({
        "step": 1,
        "agent": "Fatigue Analysis Agent",
        "status": "completed",
        "action": "Calculated camera, lane, steering fatigue scores and final risk level.",
        "output_summary": {
            "risk_level": row["risk_level"],
            "fatigue_score": row["fatigue_score"],
            "fatigue_camera": row["fatigue_camera"],
            "fatigue_lane": row["fatigue_lane"],
            "fatigue_steering": row["fatigue_steering"],
        },
    })

    rag_result = retrieve_rag_context(row)
    context = rag_result["context"]
    best_retrieval_score = rag_result["best_retrieval_score"]
    agent_progress.append({
        "step": 2,
        "agent": "RAG Retrieval Agent",
        "status": "completed",
        "action": "Retrieved relevant intervention context from persistent FAISS vector memory.",
        "output_summary": {
            "best_retrieval_score": best_retrieval_score,
            "top_k": 3,
        },
    })

    intervention_result = generate_intervention(row, context)
    prompt = intervention_result["prompt"]
    llm_output = intervention_result["llm_output"]
    parsed_raw = intervention_result["parsed"]
    model_name = intervention_result["model_name"]
    agent_progress.append({
        "step": 3,
        "agent": "Intervention Decision Agent",
        "status": "completed",
        "action": "Generated intervention decision using fatigue state, retrieved context, and Vertex AI Gemini.",
        "output_summary": {
            "model_name": model_name,
            "raw_fan_level": parsed_raw.get("fan_level"),
            "raw_music": parsed_raw.get("music"),
            "raw_vibration": parsed_raw.get("vibration"),
        },
    })

    parsed = validate_intervention(row, parsed_raw)
    agent_progress.append({
        "step": 4,
        "agent": "Safety Validation Agent",
        "status": "completed",
        "action": "Validated intervention format and enforced safety fallback rules where needed.",
        "output_summary": {
            "fan_level": parsed["fan_level"],
            "music": parsed["music"],
            "vibration": parsed["vibration"],
        },
    })

    latency_ms = int((time.perf_counter() - start) * 1000)

    vdb_update = update_vector_memory(
        row=row,
        parsed=parsed,
        llm_output=llm_output,
        best_retrieval_score=best_retrieval_score,
    )
    agent_progress.append({
        "step": 5,
        "agent": "Memory Update Agent",
        "status": "completed",
        "action": "Checked semantic novelty and updated FAISS vector memory if needed.",
        "output_summary": vdb_update,
    })

    log_prediction(
        row=row,
        context=context,
        prompt=prompt,
        llm_output=llm_output,
        parsed=parsed,
        model_name=model_name,
        latency_ms=latency_ms,
    )
    agent_progress.append({
        "step": 6,
        "agent": "Logging Agent",
        "status": "completed",
        "action": "Logged fatigue features and agent decision output to BigQuery.",
        "output_summary": {
            "bigquery_logging": "completed",
            "latency_ms": latency_ms,
        },
    })

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
        "agent_progress": agent_progress,
        "orchestration": "crewai_multi_agent_workflow",
    }