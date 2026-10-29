# FinGuard AI — CI/CD Evaluation & Security Scorecard (PASS — READY TO MERGE)

- **Timestamp (UTC)**: `2026-10-29 06:31:01 UTC`
- **Golden Dataset Size**: `36 cases` across `7 RBI PDF chunks`
- **Estimated Cost / 1,000 Queries**: `INR 14.36` (Deterministic Router Bypass: `36.1%`)

## 1. Production Quality Gates (Before vs. After Tuning)

| Quality Gate Metric | Baseline (Naive RAG) | Current FinGuard AI | CI Threshold | Gate Status |
| :--- | :--- | :--- | :--- | :--- |
| **Verdict Accuracy (%)** | 76.0% | **100.0%** | `>= 98.0%` | **PASS** |
| **PDF + Page Citation Faithfulness (%)** | 72.0% | **100.0%** | `>= 98.0%` | **PASS** |
| **Prompt-Injection Block Rate (%)** | 0.0% (Unprotected) | **100.0%** | `>= 100.0%` | **PASS** |
| **Hallucination Rate (%)** | 23.0% | **0.0%** | `<= 2.0%` | **PASS** |
| **p95 Triage Latency (ms)** | 3,800.0 ms | **3.252 ms** | `<= 50.0 ms` | **PASS** |

## 2. Automated Unit, PDF, LLM Fallback & Security Checks

| Check Name | Status |
| :--- | :--- |
| PDF Extraction & Vector Indexing (`pdfplumber`) | **PASS** |
| Structured LLM Output Happy Path (`.with_structured_output`) | **PASS** |
| LLM Timeout -> Deterministic Policy Fallback | **PASS** |
| Adversarial Prompt-Injection & Jailbreak Defense | **PASS** |
| Pydantic Currency Coercion & Schema Validation | **PASS** |
| Read-Only SQL Sanitizer (`DROP TABLE` Blocked) | **PASS** |
| 36-Case Golden Dataset Quality Gate Assertions | **PASS** |
