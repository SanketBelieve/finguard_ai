"""
main.py — FastAPI Application & REST/SSE Endpoints for FinGuard AI
==================================================================
Includes Day 3 LLM Synthesis & Fallback Telemetry Endpoints:
  - GET  /v1/llm/status
  - POST /v1/disputes/triage
  - POST /v1/disputes/stream
  - POST /v1/corpus/ingest-pdfs
  - GET  /v1/corpus/chunks
  - GET  /v1/audit/history
  - GET  /v1/metrics

Launch server:
    uvicorn main:app --reload --port 8000

Interactive Swagger UI:
    http://localhost:8000/docs
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
        version="1.3.0",
        description=(
            "Modular FastAPI service combining a sub-15ms Deterministic Router, "
            "`pdfplumber` RBI Circular Ingestion + Hybrid BM25/Vector Store, "
            "Day 3 `.with_structured_output()` LLM Synthesis with Deterministic Fallback, "
            "and a Parameterized SQL Audit Store."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/v1/disputes/triage", response_model=TriageVerdict)
    async def triage_dispute_endpoint(payload: DisputeIntakeRequest) -> TriageVerdict:
        """Synchronously triages a payment dispute and logs the audit record to SQL.

        Convert processing failures to HTTP 500 with the exception text as detail.
        """
        try:
            return engine.process_dispute(payload)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/v1/disputes/stream")
    async def stream_dispute_triage(payload: DisputeIntakeRequest) -> StreamingResponse:
        """Stream SSE data events for the router check, citation count, and verdict.

        Triage and audit logging finish after the first event and before the
        citation-count event. Processing and serialization errors terminate the
        stream without a structured error event.
        """

        async def event_generator() -> AsyncGenerator[str, None]:
            """Yield JSON SSE frames for router check, citations, and verdict.

            Process and audit the dispute after the first frame. Processing and
            JSON serialization errors propagate to the stream.
            """
            yield f"data: {json.dumps({'step': 'ROUTER_CHECK', 'transaction_id': payload.transaction_id})}\n\n"
            verdict = engine.process_dispute(payload)
            yield f"data: {json.dumps({'step': 'CITATIONS_RETRIEVED', 'count': len(verdict.citations)})}\n\n"
            verdict_dict = verdict.model_dump() if hasattr(verdict, "model_dump") else verdict.dict()
            yield f"data: {json.dumps({'step': 'VERDICT_READY', 'verdict': verdict_dict})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.get("/v1/llm/status")
    async def llm_status_endpoint() -> Dict[str, Any]:
        """Returns active LLM synthesis configuration, timeout budget, and fallback policy."""
        return {
            "live_llm_configured": engine.llm_synthesizer.is_configured,
            "active_provider": engine.llm_synthesizer.provider_name,
            "timeout_seconds": engine.llm_synthesizer.timeout_seconds,
            "max_retries": engine.llm_synthesizer.max_retries,
            "fallback_mode": "deterministic_rbi_policy_engine",
        }

    @app.post("/v1/corpus/ingest-pdfs")
    async def ingest_pdfs_endpoint(pdf_dir: str = Query(default="rbi_circulars")) -> Dict[str, Any]:
        """Re-scans `pdf_dir` with `pdfplumber`, chunks pages, and upserts vectors."""
        count = engine.retriever.refresh_pdf_corpus(pdf_dir=pdf_dir)
        return {
            "status": "indexed",
            "pdf_directory": pdf_dir,
            "vector_backend": engine.retriever.vector_store.backend,
            "chunks_indexed": count,
        }

    @app.get("/v1/corpus/chunks")
    async def list_indexed_chunks_endpoint() -> List[Dict[str, Any]]:
        """Lists all RBI PDF chunks currently stored in the vector database."""
        return engine.retriever.vector_store.list_all_chunks()

    @app.get("/v1/audit/history")
    async def audit_history_endpoint(limit: int = Query(default=20, ge=1, le=100)) -> List[Dict[str, Any]]:
        """Returns recent parameterized SQL audit records."""
        return engine.sql_store.get_recent_audit_logs(limit=limit)

    @app.get("/v1/metrics")
    async def metrics_endpoint() -> Dict[str, Any]:
        """Returns aggregated SQL latency, cost per 1,000 queries, router bypass, and LLM fallback metrics."""
        return engine.sql_store.get_operational_metrics()

    @app.get("/health")
    async def health_endpoint() -> Dict[str, str]:
        """Return static status, service, and region metadata without probing dependencies."""
        return {"status": "healthy", "service": "finguard-ai", "region": "ap-south-1 (Pune/Mumbai)"}
else:
    app = None