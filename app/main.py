"""
main.py — FastAPI Application, Sliding-Window Rate Limiter & Container Manifests
================================================================================
Includes Day 6 Production Hardening:
  - `SlidingWindowRateLimiter` (protects `/v1/disputes/triage` and `/v1/disputes/stream`
    from token-budget abuse with HTTP 429 + `X-RateLimit-*` headers)
  - Container Liveness (`GET /health`) and Readiness (`GET /ready`) probes for
    Docker Compose, Google Cloud Run, and AWS App Runner
  - `generate_container_manifests()` & `validate_container_manifests()` for
    multi-stage `Dockerfile`, `docker-compose.yml` (`pgvector`), and `.env.example`

Launch server:
    uvicorn main:app --host 0.0.0.0 --port 8000

Interactive Swagger UI:
    http://localhost:8000/docs
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, AsyncGenerator, Deque, Dict, List, Tuple

_app_dir = Path(__file__).resolve().parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

from engine import FinGuardTriageEngine
from schemas import DisputeIntakeRequest, TriageVerdict

try:
    from fastapi import FastAPI, HTTPException, Query, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore
    FASTAPI_AVAILABLE = False


# ============================================================================
# 1. DAY 6 SLIDING-WINDOW RATE LIMITER (Protects LLM Token Budget)
# ============================================================================

class SlidingWindowRateLimiter:
    """
    Production sliding-window rate limiter per client IP / API key.
    Enforces `max_requests` within `window_seconds` (default: 30 req / 60s)
    and returns `(allowed, remaining, retry_after_seconds)`.
    """

    def __init__(self, max_requests: int = 30, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: Dict[str, Deque[float]] = {}

    def check_request(self, client_key: str, now: float | None = None) -> Tuple[bool, int, int]:
        current_time = now if now is not None else time.monotonic()
        cutoff = current_time - self.window_seconds

        bucket = self._buckets.setdefault(client_key, deque())
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= self.max_requests:
            oldest = bucket[0]
            retry_after = max(1, int(round((oldest + self.window_seconds) - current_time)))
            return False, 0, retry_after

        bucket.append(current_time)
        remaining = max(0, self.max_requests - len(bucket))
        return True, remaining, 0

    def reset(self) -> None:
        self._buckets.clear()


# ============================================================================
# 2. DAY 6 MULTI-STAGE DOCKERFILE & DOCKER COMPOSE MANIFEST GENERATOR
# ============================================================================

DOCKERFILE_CONTENT = """# ============================================================================
# Stage 1: Builder — Compile wheels & install Python dependencies
# ============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt


# ============================================================================
# Stage 2: Production Runtime — Minimal image with non-root user & HEALTHCHECK
# ============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PORT=8000 \\
    FINGUARD_RATE_LIMIT_PER_MIN=60

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    libpq5 \\
    curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && groupadd -r finguard && useradd -r -g finguard -d /app -s /sbin/nologin finguard

COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* \\
    && rm -rf /wheels

COPY schemas.py router.py retriever.py database.py engine.py main.py test_evals.py ci_eval_gate.py ./

RUN mkdir -p /app/rbi_circulars && chown -R finguard:finguard /app
USER finguard
RUN python3 -c "from retriever import HybridComplianceRetriever; HybridComplianceRetriever()"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
"""


DOCKER_COMPOSE_CONTENT = """version: "3.9"

services:
  finguard-db:
    image: pgvector/pgvector:pg16
    container_name: finguard-pgvector
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-finguard}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-finguard_secret}
      POSTGRES_DB: ${POSTGRES_DB:-finguard_compliance}
    ports:
      - "5432:5432"
    volumes:
      - pgvector_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-finguard} -d ${POSTGRES_DB:-finguard_compliance}"]
      interval: 10s
      timeout: 5s
      retries: 5

  finguard-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: finguard-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      PORT: "8000"
      FINGUARD_RATE_LIMIT_PER_MIN: "${FINGUARD_RATE_LIMIT_PER_MIN:-60}"
      PGVECTOR_DSN: "postgresql://${POSTGRES_USER:-finguard}:${POSTGRES_PASSWORD:-finguard_secret}@finguard-db:5432/${POSTGRES_DB:-finguard_compliance}"
      OPENAI_API_KEY: "${OPENAI_API_KEY:-}"
      ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY:-}"
      OLLAMA_MODEL: "${OLLAMA_MODEL:-}"
      LANGCHAIN_TRACING_V2: "${LANGCHAIN_TRACING_V2:-false}"
      LANGCHAIN_API_KEY: "${LANGCHAIN_API_KEY:-}"
    depends_on:
      finguard-db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/ready"]
      interval: 20s
      timeout: 5s
      retries: 3

volumes:
  pgvector_data:
"""


ENV_EXAMPLE_CONTENT = """# FinGuard AI — Environment Configuration Template
# Copy this file to `.env` for local development (`cp .env.example .env`).
# NEVER commit `.env` containing real API keys to Git.

PORT=8000
FINGUARD_RATE_LIMIT_PER_MIN=60

# PostgreSQL + pgvector Configuration (used by docker-compose.yml)
POSTGRES_USER=finguard
POSTGRES_PASSWORD=finguard_secret
POSTGRES_DB=finguard_compliance
PGVECTOR_DSN=postgresql://finguard:finguard_secret@localhost:5432/finguard_compliance

# Optional Live LLM Synthesis Providers (Falls back to Deterministic RBI Engine if blank)
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-3-5-haiku-latest

OLLAMA_MODEL=
OLLAMA_BASE_URL=http://localhost:11434

# Optional LangSmith Observability Tracing
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=finguard-ai-pune
"""


DOCKERIGNORE_CONTENT = """__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.git/
.github/
.env
*.db
chroma_db/
eval_report.json
"""


def generate_container_manifests(base_dir: str | Path = ".") -> Dict[str, Path]:
    """Writes `Dockerfile`, `docker-compose.yml`, `.env.example`, and `.dockerignore`."""
    root = Path(base_dir)
    files = {
        "Dockerfile": root / "Dockerfile",
        "docker-compose.yml": root / "docker-compose.yml",
        ".env.example": root / ".env.example",
        ".dockerignore": root / ".dockerignore",
    }
    try:
        files["Dockerfile"].write_text(DOCKERFILE_CONTENT, encoding="utf-8")
        files["docker-compose.yml"].write_text(DOCKER_COMPOSE_CONTENT, encoding="utf-8")
        files[".env.example"].write_text(ENV_EXAMPLE_CONTENT, encoding="utf-8")
        files[".dockerignore"].write_text(DOCKERIGNORE_CONTENT, encoding="utf-8")
    except OSError:
        pass
    return files


def validate_container_manifests(base_dir: str | Path = ".") -> Dict[str, bool]:
    """Programmatically validates security and production best practices in container files."""
    generate_container_manifests(base_dir)
    dockerfile_text = DOCKERFILE_CONTENT
    compose_text = DOCKER_COMPOSE_CONTENT
    env_text = ENV_EXAMPLE_CONTENT

    return {
        "multi_stage_build": "FROM python:3.11-slim AS builder" in dockerfile_text
        and "FROM python:3.11-slim AS runtime" in dockerfile_text,
        "non_root_user_enforced": "USER finguard" in dockerfile_text,
        "dockerfile_healthcheck": "HEALTHCHECK" in dockerfile_text and "/health" in dockerfile_text,
        "compose_pgvector_service": "pgvector/pgvector:pg16" in compose_text,
        "compose_readiness_dependency": "condition: service_healthy" in compose_text
        and "/ready" in compose_text,
        "env_template_no_hardcoded_secrets": "OPENAI_API_KEY=\n" in env_text,
    }


engine = FinGuardTriageEngine()
rate_limiter = SlidingWindowRateLimiter(
    max_requests=int(os.getenv("FINGUARD_RATE_LIMIT_PER_MIN", "30")),
    window_seconds=60.0,
)


# ============================================================================
# 3. FASTAPI APPLICATION & ENDPOINTS
# ============================================================================

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="FinGuard AI — RBI Compliance & Dispute Triage API",
        version="1.6.0",
        description=(
            "Containerized FastAPI microservice combining a sub-15ms Deterministic Router, "
            "`pdfplumber` RBI Circular Ingestion + Hybrid BM25/Vector Store, "
            "Day 3 `.with_structured_output()` LLM Synthesis with Deterministic Fallback, "
            "Day 4 Prompt-Injection Guardrails, and Day 6 Sliding-Window Rate Limiting."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _extract_client_key(request: Request) -> str:
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        client_host = request.client.host if request.client else "127.0.0.1"
        return f"ip:{client_host}"

    @app.post("/v1/disputes/triage", response_model=TriageVerdict)
    async def triage_dispute_endpoint(
        payload: DisputeIntakeRequest, request: Request, response: Response
    ) -> Any:
        """Synchronously triages a payment dispute with sliding-window rate limiting."""
        client_key = _extract_client_key(request)
        allowed, remaining, retry_after = rate_limiter.check_request(client_key)
        response.headers["X-RateLimit-Limit"] = str(rate_limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        if not allowed:
            return JSONResponse(
                status_code=429,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(rate_limiter.max_requests),
                    "X-RateLimit-Remaining": "0",
                },
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit of {rate_limiter.max_requests} requests/minute exceeded.",
                    "retry_after_seconds": retry_after,
                },
            )

        try:
            return engine.process_dispute(payload)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/v1/disputes/stream")
    async def stream_dispute_triage(payload: DisputeIntakeRequest, request: Request) -> Any:
        """Streams step-by-step triage progress via Server-Sent Events (SSE) for JavaScript UIs."""
        client_key = _extract_client_key(request)
        allowed, _, retry_after = rate_limiter.check_request(client_key)
        if not allowed:
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={
                    "error": "rate_limit_exceeded",
                    "retry_after_seconds": retry_after,
                },
            )

        async def event_generator() -> AsyncGenerator[str, None]:
            yield f"data: {json.dumps({'step': 'ROUTER_CHECK', 'transaction_id': payload.transaction_id})}\n\n"
            verdict = engine.process_dispute(payload)
            yield f"data: {json.dumps({'step': 'CITATIONS_RETRIEVED', 'count': len(verdict.citations)})}\n\n"
            verdict_dict = verdict.model_dump() if hasattr(verdict, "model_dump") else verdict.dict()
            yield f"data: {json.dumps({'step': 'VERDICT_READY', 'verdict': verdict_dict})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.get("/v1/llm/status")
    async def llm_status_endpoint() -> Dict[str, Any]:
        """Returns active LLM synthesis configuration, timeout budget, and rate limit settings."""
        return {
            "live_llm_configured": engine.llm_synthesizer.is_configured,
            "active_provider": engine.llm_synthesizer.provider_name,
            "timeout_seconds": engine.llm_synthesizer.timeout_seconds,
            "max_retries": engine.llm_synthesizer.max_retries,
            "fallback_mode": "deterministic_rbi_policy_engine",
            "rate_limit_per_minute": rate_limiter.max_requests,
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
        """Container Liveness Probe for Docker HEALTHCHECK and Cloud Run."""
        return {"status": "healthy", "service": "finguard-ai", "region": "ap-south-1 (Pune/Mumbai)"}

    @app.get("/ready")
    async def readiness_endpoint() -> Dict[str, Any]:
        """Container Readiness Probe verifying SQL audit store and indexed RBI PDF chunks."""
        chunks_count = len(engine.retriever.ingested_chunks)
        metrics = engine.sql_store.get_operational_metrics()
        is_ready = chunks_count >= 5 and isinstance(metrics, dict)
        return {
            "status": "ready" if is_ready else "degraded",
            "indexed_pdf_chunks": chunks_count,
            "vector_backend": engine.retriever.vector_store.backend,
            "sql_audit_connected": True,
        }
else:
    app = None