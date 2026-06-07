"""
Apache Beam / Dataflow pipeline for driver sleepiness migration prototype.

This version follows the real project structure more closely:
- synchronized camera / lane / steering features are the input
- fatigue is estimated separately for camera, lane, and steering signals
- the combined risk level is written to BigQuery

Run locally:
    python pipelines/dataflow_fatigue_pipeline.py \
      --input data/raw_synced_driver_events.jsonl \
      --output_table YOUR_PROJECT:driver_sleepiness_ai.fatigue_features

Run on Dataflow:
    python pipelines/dataflow_fatigue_pipeline.py \
      --runner DataflowRunner \
      --project YOUR_PROJECT \
      --region europe-west3 \
      --temp_location gs://YOUR_BUCKET/temp \
      --staging_location gs://YOUR_BUCKET/staging \
      --input gs://YOUR_BUCKET/raw/raw_synced_driver_events.jsonl \
      --output_table YOUR_PROJECT:driver_sleepiness_ai.fatigue_features
"""
import argparse
import json
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions


def clamp(x, low=0.0, high=1.0):
    return max(low, min(high, x))


def classify(score: float) -> str:
    if score < 0.35:
        return "low"
    if score < 0.65:
        return "medium"
    return "high"


def level_to_number(level: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}[level]


def parse_event(line: str):
    return json.loads(line)


def extract_features_and_estimate_fatigue(event: dict):
    """
    In the actual project, these features came after synchronization and feature extraction.
    Here we use dummy synchronized feature records to simulate the same stage.

    Vision features:
        blink_rate, yawning_rate, perclos

    Lane features:
        sdlp, lane_keeping_ratio, lane_departure_frequency

    Steering features:
        steering_entropy, steering_reversal_rate, steering_angle_variability
    """
    blink_rate = float(event.get("blink_rate", 0.0))
    yawning_rate = float(event.get("yawning_rate", 0.0))
    perclos = float(event.get("perclos", 0.0))

    sdlp = float(event.get("sdlp", 0.0))
    lane_keeping_ratio = float(event.get("lane_keeping_ratio", 1.0))
    lane_departure_frequency = float(event.get("lane_departure_frequency", 0.0))

    steering_entropy = float(event.get("steering_entropy", 0.0))
    steering_reversal_rate = float(event.get("steering_reversal_rate", 0.0))
    steering_angle_variability = float(event.get("steering_angle_variability", 0.0))

    # Normalize each feature group.
    camera_score = clamp(
        0.25 * clamp(blink_rate / 40.0)
        + 0.25 * clamp(yawning_rate / 3.0)
        + 0.50 * clamp(perclos / 50.0)
    )

    lane_score = clamp(
        0.40 * clamp(sdlp / 0.70)
        + 0.30 * clamp((1.0 - lane_keeping_ratio) / 0.40)
        + 0.30 * clamp(lane_departure_frequency / 2.0)
    )

    steering_score = clamp(
        0.35 * clamp(steering_entropy / 5.0)
        + 0.30 * clamp(steering_reversal_rate / 10.0)
        + 0.35 * clamp(steering_angle_variability / 9.0)
    )

    fatigue_camera = classify(camera_score)
    fatigue_lane = classify(lane_score)
    fatigue_steering = classify(steering_score)

    # Combined fatigue score.
    fatigue_score = clamp(
        0.45 * camera_score
        + 0.30 * lane_score
        + 0.25 * steering_score
    )

    # Risk level is combined from score and individual modality levels.
    fatigue_levels = [fatigue_camera, fatigue_steering, fatigue_lane]
    if fatigue_levels.count("high") >= 2:
        risk_level = "high"
    elif "high" in fatigue_levels or fatigue_score >= 0.65:
        risk_level = "high"
    elif fatigue_levels.count("medium") >= 2 or fatigue_score >= 0.35:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "session_id": event["session_id"],
        "timestamp": event["timestamp"],
        "camera_frame_id": event.get("camera_frame_id"),
        "sync_window_ms": int(event.get("sync_window_ms", 0)),
        "blink_rate": blink_rate,
        "yawning_rate": yawning_rate,
        "perclos": perclos,
        "sdlp": sdlp,
        "lane_keeping_ratio": lane_keeping_ratio,
        "lane_departure_frequency": lane_departure_frequency,
        "steering_entropy": steering_entropy,
        "steering_reversal_rate": steering_reversal_rate,
        "steering_angle_variability": steering_angle_variability,
        "fatigue_camera": fatigue_camera,
        "fatigue_steering": fatigue_steering,
        "fatigue_lane": fatigue_lane,
        "fatigue_score": round(fatigue_score, 4),
        "risk_level": risk_level,
    }


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL path, local or gs://")
    parser.add_argument("--output_table", required=True, help="BigQuery table: PROJECT:DATASET.TABLE")
    known_args, pipeline_args = parser.parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args, save_main_session=True)

    schema = (
        "session_id:STRING,timestamp:TIMESTAMP,camera_frame_id:STRING,"
        "sync_window_ms:INTEGER,blink_rate:FLOAT,yawning_rate:FLOAT,perclos:FLOAT,"
        "sdlp:FLOAT,lane_keeping_ratio:FLOAT,lane_departure_frequency:FLOAT,"
        "steering_entropy:FLOAT,steering_reversal_rate:FLOAT,"
        "steering_angle_variability:FLOAT,fatigue_camera:STRING,"
        "fatigue_steering:STRING,fatigue_lane:STRING,fatigue_score:FLOAT,"
        "risk_level:STRING"
    )

    with beam.Pipeline(options=pipeline_options) as p:
        (
            p
            | "ReadRawEvents" >> beam.io.ReadFromText(known_args.input)
            | "ParseJSON" >> beam.Map(parse_event)
            | "ExtractFeaturesAndEstimateFatigue" >> beam.Map(extract_features_and_estimate_fatigue)
            | "WriteFatigueFeaturesToBigQuery" >> beam.io.WriteToBigQuery(
                known_args.output_table,
                schema=schema,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            )
        )


if __name__ == "__main__":
    run()
