"""Health check routes."""

from fastapi import APIRouter, Request

from ed_orchestrator import __version__
from ed_orchestrator.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    ctx = request.app.state.ctx
    return HealthResponse(
        status="ok",
        version=__version__,
        model_loaded=ctx.inference.is_loaded,
        model_version=ctx.inference.model_version if ctx.inference.is_loaded else None,
    )
