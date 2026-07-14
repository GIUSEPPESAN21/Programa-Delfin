"""Telemetry and twin state routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ed_orchestrator.api.schemas import EDStateSnapshot, TelemetryResponse

router = APIRouter()


@router.post("/telemetry", response_model=TelemetryResponse)
async def ingest_telemetry(request: Request, snapshot: EDStateSnapshot) -> TelemetryResponse:
    ctx = request.app.state.ctx
    snapshot_id = await ctx.state_manager.ingest_telemetry(snapshot)
    lookahead = await ctx.state_manager.run_lookahead(horizon_minutes=120)
    await ctx.firestore.save_snapshot(snapshot_id, snapshot, lookahead)
    return TelemetryResponse(status="accepted", snapshot_id=snapshot_id, lookahead=lookahead)


@router.get("/twin/state")
async def get_twin_state(request: Request):
    ctx = request.app.state.ctx
    snapshot = ctx.state_manager.latest_snapshot
    if snapshot is None:
        return {"status": "empty"}
    return {
        "snapshot": snapshot.model_dump(),
        "state_vector": ctx.state_manager.get_current_state_vector().tolist(),
        "sync_log": ctx.state_manager.sync.sync_log[-5:],
    }
