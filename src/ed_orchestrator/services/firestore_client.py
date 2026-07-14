"""Firestore integration with in-memory fallback for local dev."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ed_orchestrator.api.schemas import EDStateSnapshot, LookaheadResult


class FirestoreClient:
    """Persists telemetry and shadow logs; falls back to local JSONL."""

    def __init__(self):
        self.use_gcp = os.getenv("USE_GCP", "false").lower() == "true"
        self.project = os.getenv("GCP_PROJECT", "ed-orchestrator-dev")
        self._local_log = Path(os.getenv("LOCAL_DATA_DIR", "data/processed"))
        self._local_log.mkdir(parents=True, exist_ok=True)
        self._client = None
        if self.use_gcp:
            try:
                from google.cloud import firestore

                self._client = firestore.Client(project=self.project)
            except Exception:
                self.use_gcp = False

    async def save_snapshot(
        self,
        snapshot_id: str,
        snapshot: EDStateSnapshot,
        lookahead: Optional[LookaheadResult],
    ) -> None:
        doc = {
            "snapshot_id": snapshot_id,
            "timestamp": snapshot.timestamp.isoformat(),
            "patients_in_system": snapshot.patients_in_system,
            "lookahead": lookahead.model_dump() if lookahead else None,
        }
        await self._write("telemetry", snapshot_id, doc)

    async def log_shadow_recommendation(
        self,
        snapshot: EDStateSnapshot,
        result: Dict[str, Any],
        mode: str,
    ) -> None:
        doc = {
            "timestamp": datetime.utcnow().isoformat(),
            "mode": mode,
            "action": result["action_name"],
            "confidence": result["confidence"],
            "patients_in_system": snapshot.patients_in_system,
        }
        await self._write("shadow_logs", datetime.utcnow().isoformat(), doc)

    async def _write(self, collection: str, doc_id: str, data: Dict[str, Any]) -> None:
        if self.use_gcp and self._client:
            self._client.collection(collection).document(doc_id).set(data)
            return
        path = self._local_log / f"{collection}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": doc_id, **data}) + "\n")
