"""FastAPI application entrypoint."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ed_orchestrator import __version__
from ed_orchestrator.api.routes import health, recommend, twin_state
from ed_orchestrator.digital_twin.state_manager import StateManager
from ed_orchestrator.rl.inference import InferenceService
from ed_orchestrator.services.firestore_client import FirestoreClient


class AppState:
    def __init__(self):
        self.state_manager = StateManager()
        self.inference = InferenceService()
        self.firestore = FirestoreClient()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.ctx = AppState()
    yield


app = FastAPI(
    title="ED Orchestrator API",
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(recommend.router, prefix="/api/v1", tags=["recommend"])
app.include_router(twin_state.router, prefix="/api/v1", tags=["twin"])


def run():
    import uvicorn

    uvicorn.run(
        "ed_orchestrator.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8080")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
    )


if __name__ == "__main__":
    run()
