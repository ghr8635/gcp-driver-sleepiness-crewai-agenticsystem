def clamp(x, low=0.0, high=1.0):
    return max(low, min(high, x))


def classify(score: float) -> str:
    if score < 0.35:
        return "low"
    if score < 0.65:
        return "medium"
    return "high"


def estimate_fatigue(event: dict) -> dict:
    blink_rate = float(event.get("blink_rate", 0.0))
    yawning_rate = float(event.get("yawning_rate", 0.0))
    perclos = float(event.get("perclos", 0.0))

    sdlp = float(event.get("sdlp", 0.0))
    lane_keeping_ratio = float(event.get("lane_keeping_ratio", 1.0))
    lane_departure_frequency = float(event.get("lane_departure_frequency", 0.0))

    steering_entropy = float(event.get("steering_entropy", 0.0))
    steering_reversal_rate = float(event.get("steering_reversal_rate", 0.0))
    steering_angle_variability = float(event.get("steering_angle_variability", 0.0))

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

    fatigue_score = clamp(
        0.45 * camera_score
        + 0.30 * lane_score
        + 0.25 * steering_score
    )

    fatigue_levels = [fatigue_camera, fatigue_steering, fatigue_lane]

    if fatigue_levels.count("high") >= 2:
        risk_level = "high"
    elif "high" in fatigue_levels or fatigue_score >= 0.65:
        risk_level = "high"
    elif fatigue_levels.count("medium") >= 2 or fatigue_score >= 0.35:
        risk_level = "medium"
    else:
        risk_level = "low"

    output = dict(event)
    output.update({
        "fatigue_camera": fatigue_camera,
        "fatigue_steering": fatigue_steering,
        "fatigue_lane": fatigue_lane,
        "fatigue_score": round(fatigue_score, 4),
        "risk_level": risk_level,
    })

    return output
