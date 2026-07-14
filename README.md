# Sistema Ciberfísico Híbrido de Orquestación Hospitalaria en Urgencias

Investigación y sistema de producción para orquestación operativa en departamentos de urgencias mediante Gemelo Digital activo-bidireccional, Deep Reinforcement Learning (DQN) e Explainable AI (XAI).

## Estructura del proyecto

```
Research Delifn/
├── src/ed_orchestrator/       # Paquete de producción (FastAPI, Streamlit, RL)
├── simulation/                # Simulación reproducible (SimPy, experimentos)
├── manuscript/                # Manuscrito científico DOCX
├── infra/                     # Docker, GCP, GitHub Actions
├── mlops/                     # DVC pipeline, MLflow
├── docs/ARCHITECTURE.md       # Documentación técnica de arquitectura
└── tests/                     # Tests pytest
```

## Instalación

```bash
pip install -e .
pip install -r requirements/dev.txt
cd simulation && pip install -r requirements.txt
```

## Ejecución reproducible (investigación)

```bash
cd simulation
python experiments/evaluate_baselines.py
python experiments/train_dqn.py
python experiments/robustness.py
python generate_manuscript_assets.py
python ../manuscript/build_docx.py
```

## Despliegue local (Docker)

```bash
docker compose -f infra/docker/docker-compose.yml up --build
# API: http://localhost:8080/health
# Dashboard: http://localhost:8501
```

## Despliegue GCP

```bash
export GCP_PROJECT_ID=your-project
bash scripts/deploy_gcp.sh latest
```

## API endpoints

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado del servicio y modelo |
| `POST /api/v1/telemetry` | Ingesta telemetría + look-ahead |
| `POST /api/v1/recommend` | Recomendación DQN + XAI (shadow mode) |
| `GET /api/v1/twin/state` | Estado del gemelo digital |

## Manuscrito

El manuscrito científico se genera en:
`manuscript/Orquestacion_Urgencias_Manuscrito.docx`

## Referencias clave

- Soster et al. (2025) JAMIA Open — MLOps + optimización de turnos
- Liu et al. (2020) — DRL en urgencias (PMC7349722)
- Vural et al. (2025) JMIR — Predicción de hacinamiento
- Applied Sciences (2024) — DES + ML scheduling
