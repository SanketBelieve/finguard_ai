# FinGuard AI — CI/CD Evaluation & Security Scorecard (PASS — READY TO MERGE)

- **Timestamp (UTC)**: `2026-11-02 10:19:15 UTC`
- **Golden Dataset Size**: `36 cases` across `7 RBI PDF chunks`
- **Estimated Cost / 1,000 Queries**: `INR 12.08` (Deterministic Router Bypass: `41.7%`)

## 1. Production Quality Gates (Before vs. After Tuning)

| Quality Gate Metric | Baseline (Naive RAG) | Current FinGuard AI | CI Threshold | Gate Status |
| :--- | :--- | :--- | :--- | :--- |
| **Verdict Accuracy (%)** | 76.0% | **100.0%** | `>= 98.0%` | **PASS** |
| **PDF + Page Citation Faithfulness (%)** | 72.0% | **100.0%** | `>= 98.0%` | **PASS** |
| **Prompt-Injection Block Rate (%)** | 0.0% (Unprotected) | **100.0%** | `>= 100.0%` | **PASS** |
| **Hallucination Rate (%)** | 23.0% | **0.0%** | `<= 2.0%` | **PASS** |
| **p95 Triage Latency (ms)** | 3,800.0 ms | **2.323 ms** | `<= 50.0 ms` | **PASS** |

## 2. Automated Phase 1, Phase 2 & Phase 3 Checks (13 Suites)

| Check Name | Status |
| :--- | :--- |
| PDF Extraction & Vector Indexing (`pdfplumber`) | **PASS** |
| Structured LLM Output Happy Path (`.with_structured_output`) | **PASS** |
| LLM Timeout -> Deterministic Policy Fallback | **PASS** |
| Adversarial Prompt-Injection & Jailbreak Defense | **PASS** |
| Sliding-Window API Rate Limiter (HTTP 429 Quota) | **PASS** |
| Multi-Stage Dockerfile & Compose Security Manifests | **PASS** |
| Architecture Decision Records (`README.md` & `DECISIONS.md`) | **PASS** |
| Phase 2 MCP SQL Server Tools & DDL Injection Block | **PASS** |
| Phase 2 4-Node LangGraph Workflow & HITL Checkpoint Resume | **PASS** |
| Phase 3 Interactive Officer Console (`finguard_dashboard.html`) | **PASS** |
| Pydantic Currency Coercion & Schema Validation | **PASS** |
| Read-Only SQL Sanitizer (`DROP TABLE` Blocked) | **PASS** |
| 36-Case Golden Dataset Quality Gate Assertions | **PASS** |