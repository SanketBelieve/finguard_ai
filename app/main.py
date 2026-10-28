"""
main.py — FastAPI Application & REST/SSE Endpoints for FinGuard AI
==================================================================

"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List

from engine import FinGuardTriageEngine
from schemas import DisputeIntakeRequest, TriageVerdict

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore
    FASTAPI_AVAILABLE = False


engine = FinGuardTriageEngine()

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="FinGuard AI — RBI Compliance & Dispute Triage API",
        version="1.0.0",
        description=(
            "Modular FastAPI service combining a sub-15ms Deterministic Router, "
            "Hybrid BM25+Vector RAG Retriever over RBI Master Directions, and "
            "a Parameterized SQL Audit Store."
        ),
    )

    # Enable CORS for React / Next.js frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/v1/disputes/triage", response_model=TriageVerdict)
    async def triage_dispute_endpoint(payload: DisputeIntakeRequest) -> TriageVerdict:
        """Synchronously triages a payment dispute and logs the audit record to SQL."""
        try:
            return engine.process_dispute(payload)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/v1/disputes/stream")
    async def stream_dispute_triage(payload: DisputeIntakeRequest) -> StreamingResponse:
        """Streams step-by-step triage progress via Server-Sent Events (SSE) for JavaScript UIs."""

        async def event_generator() -> AsyncGenerator[str, None]:
            yield f"data: {json.dumps({'step': 'ROUTER_CHECK', 'transaction_id': payload.transaction_id})}\n\n"
            verdict = engine.process_dispute(payload)
            yield f"data: {json.dumps({'step': 'CITATIONS_RETRIEVED', 'count': len(verdict.citations)})}\n\n"
            verdict_dict = verdict.model_dump() if hasattr(verdict, "model_dump") else verdict.dict()
            yield f"data: {json.dumps({'step': 'VERDICT_READY', 'verdict': verdict_dict})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.get("/v1/audit/history")
    async def audit_history_endpoint(limit: int = Query(default=20, ge=1, le=100)) -> List[Dict[str, Any]]:
        """Returns recent parameterized SQL audit records."""
        return engine.sql_store.get_recent_audit_logs(limit=limit)

    @app.get("/v1/metrics")
    async def metrics_endpoint() -> Dict[str, Any]:
        """Returns aggregated SQL latency, cost per 1,000 queries, and router bypass metrics."""
        return engine.sql_store.get_operational_metrics()

    @app.get("/health")
    async def health_endpoint() -> Dict[str, str]:
        return {"status": "healthy", "service": "finguard-ai", "region": "ap-south-1 (Pune/Mumbai)"}
else:
    app = None