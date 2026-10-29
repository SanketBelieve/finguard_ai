# ============================================================================
# Stage 1: Builder — Compile wheels & install Python dependencies
# ============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt


# ============================================================================
# Stage 2: Production Runtime — Minimal image with non-root user & HEALTHCHECK
# ============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    FINGUARD_RATE_LIMIT_PER_MIN=60

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r finguard && useradd -r -g finguard -d /app -s /sbin/nologin finguard

COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* \
    && rm -rf /wheels

COPY app/schemas.py app/router.py app/retriever.py app/database.py app/engine.py app/main.py test_evals.py ci_eval_gate.py ./

RUN mkdir -p /app/rbi_circulars && chown -R finguard:finguard /app
USER finguard
RUN python3 -c "from retriever import HybridComplianceRetriever; HybridComplianceRetriever()"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
