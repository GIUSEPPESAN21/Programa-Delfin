"""Pydantic data contracts for API and telemetry."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DeploymentMode(str, Enum):
    SHADOW = "shadow"
    ACTIVE = "active"


class EDStateSnapshot(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    queue_by_esi: Dict[int, int] = Field(default_factory=lambda: {i: 0 for i in range(1, 6)})
    wait_times_by_esi: Dict[int, float] = Field(default_factory=lambda: {i: 0.0 for i in range(1, 6)})
    beds_occupied: int = 0
    consult_occupied: int = 0
    imaging_occupied: int = 0
    patients_in_system: int = 0
    vitals_summary: Dict[str, float] = Field(default_factory=dict)
    hourly_arrivals: List[int] = Field(default_factory=list)


class LookaheadResult(BaseModel):
    mean_occupancy: float
    std_occupancy: float
    n_scenarios: int
    latency_ms: float
    horizon_minutes: int = 120


class FeatureAttribution(BaseModel):
    feature: str
    value: float
    attribution: float


class RecommendRequest(BaseModel):
    state: EDStateSnapshot
    model_version: Optional[str] = "latest"
    mode: DeploymentMode = DeploymentMode.SHADOW


class RecommendResponse(BaseModel):
    action: int
    action_name: str
    q_values: List[float]
    confidence: float
    clinical_narrative: str
    manager_narrative: str
    top_features: List[FeatureAttribution]
    projected_impact: Dict[str, Any]
    model_version: str
    mode: DeploymentMode


class TelemetryResponse(BaseModel):
    status: str
    snapshot_id: str
    lookahead: Optional[LookaheadResult] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    model_version: Optional[str] = None
