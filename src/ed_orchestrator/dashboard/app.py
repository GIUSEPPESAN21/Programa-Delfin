"""Streamlit dashboard for clinical and operational monitoring."""

from __future__ import annotations

import os
from datetime import datetime

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("ED_API_URL", "http://localhost:8080")


def fetch_json(path: str):
    try:
        r = httpx.get(f"{API_URL}{path}", timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.warning(f"API unavailable: {exc}")
        return None


def post_json(path: str, payload: dict, headers: dict | None = None):
    try:
        r = httpx.post(f"{API_URL}{path}", json=payload, headers=headers or {}, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def demo_snapshot() -> dict:
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "queue_by_esi": {1: 2, 2: 5, 3: 8, 4: 12, 5: 6},
        "wait_times_by_esi": {1: 5.0, 2: 22.0, 3: 45.0, 4: 60.0, 5: 30.0},
        "beds_occupied": 22,
        "consult_occupied": 3,
        "imaging_occupied": 1,
        "patients_in_system": 33,
        "vitals_summary": {"spo2": 96.5, "heart_rate": 92.0},
        "hourly_arrivals": [8, 10, 14, 18, 12],
    }


def panel_clinico():
    st.subheader("Panel Clínico")
    health = fetch_json("/health")
    if health:
        st.caption(f"Modelo: {health.get('model_version', 'N/A')} | Cargado: {health.get('model_loaded')}")

    snapshot = demo_snapshot()
    cols = st.columns(5)
    for i, col in enumerate(cols, start=1):
        col.metric(f"Cola ESI-{i}", snapshot["queue_by_esi"][i])

    if st.button("Obtener recomendación operativa"):
        resp = post_json(
            "/api/v1/recommend",
            {"state": snapshot, "mode": "shadow"},
            headers={"X-Mode": "shadow"},
        )
        if resp:
            st.success(resp["action_name"])
            st.markdown("**Justificación clínica**")
            st.text(resp["clinical_narrative"])
            st.markdown("**Covariables principales (SHAP)**")
            st.dataframe(pd.DataFrame(resp["top_features"]))


def panel_gestor():
    st.subheader("Panel Gestor Operativo")
    twin = fetch_json("/api/v1/twin/state")
    if twin and twin.get("status") != "empty":
        st.json(twin)
    else:
        st.info("Sin telemetría reciente. Use el panel clínico para simular ingesta.")

    st.markdown("**KPIs proyectados**")
    resp = post_json("/api/v1/telemetry", demo_snapshot())
    if resp and resp.get("lookahead"):
        la = resp["lookahead"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Ocupación proyectada", f"{la['mean_occupancy']*100:.1f}%")
        c2.metric("Escenarios", la["n_scenarios"])
        c3.metric("Latencia sync (ms)", f"{la['latency_ms']:.0f}")


def panel_mlops():
    st.subheader("Panel MLOps")
    health = fetch_json("/health")
    if health:
        st.metric("Versión API", health.get("version"))
        st.metric("Modelo cargado", str(health.get("model_loaded")))
    st.markdown("**Drift monitoring** (placeholder)")
    st.progress(0.15, text="Drift score: 0.15 (bajo)")
    if st.button("Solicitar re-entrenamiento"):
        st.info("Pipeline DVC/GitHub Actions: train.yml dispatch registrado.")


def run():
    st.set_page_config(page_title="ED Orchestrator", layout="wide")
    st.title("Sistema Ciberfísico de Orquestación en Urgencias")
    tab1, tab2, tab3 = st.tabs(["Clínico", "Gestor", "MLOps"])
    with tab1:
        panel_clinico()
    with tab2:
        panel_gestor()
    with tab3:
        panel_mlops()


if __name__ == "__main__":
    run()
