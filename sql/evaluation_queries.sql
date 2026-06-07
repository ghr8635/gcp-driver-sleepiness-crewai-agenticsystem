-- Inspect processed fatigue features
SELECT
  session_id,
  blink_rate,
  yawning_rate,
  perclos,
  sdlp,
  lane_keeping_ratio,
  lane_departure_frequency,
  steering_entropy,
  steering_reversal_rate,
  steering_angle_variability,
  fatigue_camera,
  fatigue_steering,
  fatigue_lane,
  fatigue_score,
  risk_level
FROM `YOUR_PROJECT.driver_sleepiness_ai.fatigue_features`
ORDER BY timestamp;

-- Inspect LLM-style outputs in the same format as the original project
SELECT
  session_id,
  risk_level,
  fatigue_score,
  fan_level,
  music,
  vibration,
  reason
FROM `YOUR_PROJECT.driver_sleepiness_ai.agent_decision_logs`
ORDER BY timestamp;

-- Risk and intervention distribution
SELECT
  risk_level,
  COUNT(*) AS n,
  AVG(fan_level) AS avg_fan_level,
  COUNTIF(music = 'On') AS music_on_count,
  COUNTIF(vibration = 'On') AS vibration_on_count,
  AVG(latency_ms) AS avg_latency_ms
FROM `YOUR_PROJECT.driver_sleepiness_ai.agent_decision_logs`
GROUP BY risk_level
ORDER BY risk_level;
