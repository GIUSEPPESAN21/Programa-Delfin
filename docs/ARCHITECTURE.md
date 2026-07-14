# ED Orchestrator — Production Architecture

## 1. Overview

The **ED Orchestrator** is a hybrid cyber-physical system for real-time emergency department (ED) operational orchestration. It integrates four modules:

| Module | Technology | Responsibility |
|---|---|---|
| Digital Twin | SimPy | Patient flow simulation, bidirectional sync |
| RL Agent | Gymnasium + PyTorch/RLlib | Dynamic dispatch policy optimization |
| XAI Layer | SHAP | Transparent decision explanations |
| UI | FastAPI + Streamlit | Clinical/operational dashboards |

## 2. System Architecture

```
┌─────────────┐     RTLS/EHR      ┌──────────────────┐
│  Physical   │ ───────────────►  │  State Manager   │
│  ED Plane   │                   │  (FastAPI)       │
└─────────────┘                   └────────┬─────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
            ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
            │ Bidirectional│      │ DQN Inference│      │  Firestore   │
            │ Sync + SimPy │      │ + SHAP XAI   │      │  / BigQuery  │
            └──────────────┘      └──────────────┘      └──────────────┘
                                           │
                                           ▼
                                  ┌──────────────┐
                                  │  Streamlit   │
                                  │  Dashboard   │
                                  └──────────────┘
```

## 3. Module Specifications

### 3.1 Digital Twin (SimPy)

**Location:** `simulation/ed_digital_twin/` (research), `src/ed_orchestrator/digital_twin/` (production)

**Key classes:**
- `EmergencyDepartmentTwin`: SimPy environment with patient processes, arrival NHPP, metrics
- `BidirectionalSync`: pull/push/feedback with Δt ≤ 30s
- `StateManager`: async telemetry ingest, look-ahead projection

**Data contract — `EDStateSnapshot`:**
```python
{
  "timestamp": "ISO8601",
  "queue_by_esi": {1: int, ..., 5: int},
  "wait_times_by_esi": {1: float, ..., 5: float},
  "beds_occupied": int,
  "consult_occupied": int,
  "imaging_occupied": int,
  "patients_in_system": int,
  "vitals_summary": {"spo2": float, "heart_rate": float}
}
```

**SLA:** sync latency ≤ 30s; look-ahead p95 ≤ 5s (15 scenarios) / ≤ 60s (50 scenarios)

### 3.2 RL Agent (Gymnasium + DQN)

**MDP formulation:**
- State: R^17 (normalized operational + physiological features)
- Action: Discrete(6) — composite dispatch policies
- Reward: multi-objective (critical wait, LOS, occupancy, throughput, equity)
- γ = 0.99

**Training pipeline:**
1. Offline training on SimPy environment (`simulation/experiments/train_dqn.py`)
2. RLlib training with MLflow (`src/ed_orchestrator/rl/train_rllib.py`)
3. Model export to `models/dqn_model.pt`

**Inference:** `InferenceService.predict(state_vector)` → action + Q-values + SHAP

### 3.3 XAI Layer

**Flow:** state → DQN → SHAP KernelExplainer → clinical/manager narratives

**Output contract — `RecommendResponse`:**
```json
{
  "action": 1,
  "action_name": "Refuerzo casos críticos (ESI 1-2)",
  "q_values": [0.12, 0.45, 0.08, 0.11, 0.09, 0.15],
  "confidence": 0.31,
  "clinical_narrative": "...",
  "manager_narrative": "...",
  "top_features": [{"feature": "espera_ESI2", "value": 0.18, "attribution": 0.34}],
  "projected_impact": {"los_delta_min": -8.5, "horizon_hours": 2},
  "model_version": "dqn_model",
  "mode": "shadow"
}
```

### 3.4 API (FastAPI)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check + model status |
| `/api/v1/telemetry` | POST | Ingest ED snapshot + run look-ahead |
| `/api/v1/recommend` | POST | Get DQN recommendation + XAI |
| `/api/v1/twin/state` | GET | Current twin state + sync log |

**Shadow mode:** Header `X-Mode: shadow` logs recommendation without execution.

### 3.5 Dashboard (Streamlit)

Three role-based panels:
- **Clinical:** ESI queues, recommendations, SHAP features
- **Manager:** KPIs, look-ahead projections, sync status
- **MLOps:** model version, drift score, retrain trigger

## 4. MLOps Lifecycle

### Phase 1: Prototyping (Pure Simulation)
- Build SimPy twin, validate KS/MAPE
- Script: `simulation/experiments/evaluate_baselines.py`

### Phase 2: Offline RL Training
- Train DQN on historical/synthetic data
- DVC pipeline: `mlops/dvc.yaml`
- MLflow experiment tracking

### Phase 3: Shadow Mode
- Connect trained agent to twin
- Human physicians validate recommendations
- Logs in Firestore `shadow_logs` collection

### Phase 4: Deployment & Monitoring
- Docker → Cloud Run
- Drift detection via BigQuery analytics
- Auto-retrain trigger via GitHub Actions `train.yml`

## 5. GCP Deployment

| Service | Purpose |
|---|---|
| Cloud Run (api) | FastAPI inference |
| Cloud Run (dashboard) | Streamlit UI |
| Firestore | Real-time state, shadow logs |
| BigQuery | Historical metrics, drift analytics |
| Artifact Registry | Docker images |
| Secret Manager | GCP credentials |
| Cloud Scheduler | Drift check every 6h |

**Deploy:** `bash scripts/deploy_gcp.sh [tag]`

**Local dev:** `docker compose -f infra/docker/docker-compose.yml up`

## 6. Code Structure

```
Research Delifn/
├── src/ed_orchestrator/     # Production package
├── simulation/              # Research simulation + experiments
├── manuscript/              # Scientific manuscript (DOCX)
├── infra/                   # Docker, GCP, CI/CD
├── mlops/                   # DVC pipeline
├── data/                    # DVC-tracked datasets
├── models/                  # Exported model artifacts
├── tests/                   # pytest suite
└── docs/ARCHITECTURE.md     # This document
```

## 7. Validation Strategy

| Phase | Validation | Pass Criteria |
|---|---|---|
| Prototype | KS test, MAPE | p > 0.05 per station |
| Offline RL | Convergence curve | Stable reward, no Q divergence |
| Shadow | Clinical checklist | >80% coherence rating |
| Deploy | API latency | p95 < 200ms |
| XAI | Perturbation correlation | r > 0.7 vs manual attribution |

## 8. Security (HIPAA-aligned)

- PHI encrypted at rest in Firestore (GCP default encryption)
- IAM least-privilege for Cloud Run service accounts
- No PHI in application logs
- Secret Manager for credentials (never in code/env files committed)
- Shadow mode default in production until clinical validation

## 9. End-to-End State Flow Example

1. RTLS detects 8 ESI-2 patients waiting >20 min → POST `/api/v1/telemetry`
2. StateManager syncs twin, runs 50 look-ahead scenarios → 92% projected occupancy in 2h
3. POST `/api/v1/recommend` with `X-Mode: shadow`
4. DQN selects action 1 (critical boost); SHAP attributes espera_ESI2 (0.34), ocupacion_camas (0.21)
5. Streamlit displays clinical narrative; physician accepts/rejects → logged to Firestore
6. BigQuery aggregates outcomes for drift monitoring

## 10. Library Versions

| Component | Library | Version |
|---|---|---|
| Simulation | simpy | ≥4.1.1 |
| RL Env | gymnasium | ≥1.0.0 |
| RL Training | ray[rllib] | ≥2.9.0 |
| Deep Learning | torch | ≥2.2.0 |
| API | fastapi | ≥0.110.0 |
| XAI | shap | ≥0.44.0 |
| MLOps | mlflow, dvc | ≥2.10, ≥3.0 |
| Dashboard | streamlit | ≥1.32.0 |
| GCP | google-cloud-firestore | ≥2.16.0 |
