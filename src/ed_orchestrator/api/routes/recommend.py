"""Recommendation and shadow-mode routes."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Header, Request

from ed_orchestrator.api.schemas import DeploymentMode, FeatureAttribution, RecommendRequest, RecommendResponse

router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(
    request: Request,
    body: RecommendRequest,
    x_mode: str | None = Header(default=None, alias="X-Mode"),
) -> RecommendResponse:
    ctx = request.app.state.ctx
    mode = DeploymentMode(x_mode or body.mode.value)

    await ctx.state_manager.ingest_telemetry(body.state)
    state_vec = ctx.state_manager.get_current_state_vector()
    result = ctx.inference.predict(state_vec)

    if mode == DeploymentMode.SHADOW:
        await ctx.firestore.log_shadow_recommendation(body.state, result, mode.value)

    return RecommendResponse(
        action=result["action"],
        action_name=result["action_name"],
        q_values=result["q_values"],
        confidence=result["confidence"],
        clinical_narrative=result["clinical_narrative"],
        manager_narrative=result["manager_narrative"],
        top_features=[FeatureAttribution(**f) for f in result["top_features"]],
        projected_impact=result["projected_impact"],
        model_version=result["model_version"],
        mode=mode,
    )
