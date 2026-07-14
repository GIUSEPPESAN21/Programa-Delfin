CREATE TABLE IF NOT EXISTS ed_metrics (
  snapshot_id STRING NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  patients_in_system INT64,
  mean_occupancy_projected FLOAT64,
  action_recommended STRING,
  mode STRING,
  model_version STRING
);

CREATE TABLE IF NOT EXISTS ed_shadow_logs (
  log_id STRING NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  action STRING,
  confidence FLOAT64,
  patients_in_system INT64,
  accepted BOOL
);
