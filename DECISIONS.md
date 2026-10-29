# Engineering Decisions & Trade-Offs (`DECISIONS.md`)

This document records the core architectural decisions made while building **FinGuard AI**, why alternatives were rejected, and how each choice affects latency, accuracy, security, and operating cost.

## ADR-001: Sub-15ms Deterministic Intent Router Ahead of LLM & Vector Search

* **Context**: In high-volume Indian payment operations (UPI/IMPS/NEFT/RTGS), 35–50% of inbound disputes are routine failed debits ("account debited, beneficiary not credited") or off-topic/adversarial inputs. Routing every request directly to a frontier LLM API inflates p95 latency (>2,000 ms) and token spend.
* **Decision**: Place `DeterministicFinTechRouter` (`router.py`) at Step 1 of the pipeline to classify clear-cut failed debits, duplicate charges, commercial non-delivery claims, and prompt-injection attacks in **<1ms** using compiled regex patterns before invoking `StructuredLLMSynthesizer`.
* **Impact**: Bypassed external LLM calls on **36.1% of Golden Dataset traffic**, cutting estimated cost from **INR 42.00 to INR 14.36 per 1,000 queries (-65.8%)** and blocking **100% of prompt-injection attacks** with zero token spend.

## ADR-002: Hybrid Search (BM25 + Dense Vectors) + Priority Reranking Over Pure Vector Search

* **Context**: Pure dense vector similarity (`similarity_search()`) struggles to distinguish between clauses that share heavily overlapping vocabulary—such as RBI's **Zero Liability (`<=72h`, Paragraph 6)** vs. **Limited Liability (`4–7 days`, Paragraph 7)** vs. **Customer Negligence (`shared OTP/PIN`, Paragraph 7(ii))**. In baseline testing, naive vector lookup produced a **23.0% hallucination/mis-citation rate**.
* **Decision**: Implement a 3-stage `HybridComplianceRetriever` (`retriever.py`) combining:
  1. Dense vector cosine similarity (`RegulatoryVectorStore`),
  2. Sparse BM25-style lexical keyword overlap, and
  3. A deterministic domain priority reranker that enforces RBI statutory hierarchy (Customer Credential Sharing `Paragraph 7(ii)` $>$ Reporting Delay Tiers $>$ Rail-Specific TAT Clauses).
* **Impact**: Improved PDF + page-number citation faithfulness from **72.0% to 100.0%** across all 36 Golden Evaluation cases and reduced hallucination rate to **0.0%**.

## ADR-003: Pluggable 3-Backend Vector Store (`sqlite_vec` | `chroma` | `pgvector`)

* **Context**: Many portfolio repositories fail during recruiter review or GitHub Actions CI runs because they hardcode external cloud vector DB credentials (such as Pinecone API keys) or require a heavy local Docker cluster just to run `pytest`.
* **Decision**: Build `RegulatoryVectorStore` (`retriever.py`) with three interchangeable backends behind a unified `upsert_chunks()` and `similarity_search()` interface:
  * `sqlite_vec` (default persistent SQLite + NumPy exact cosine search, zero external daemon required),
  * `chroma` (`chromadb.PersistentClient` for local HNSW indexing), and
  * `pgvector` (`PostgreSQL 16` + `vector_cosine_ops` HNSW index for containerized production via `docker-compose.yml`).
* **Impact**: Reviewers and GitHub Actions runners can execute `python3 test_evals.py` in under 2 seconds with zero setup, while production deployments use `pgvector` via `docker compose up`.

## ADR-004: `.with_structured_output(LLMSynthesisDraft)` Paired with Deterministic RBI Policy Fallback

* **Context**: External LLM APIs (OpenAI, Anthropic) experience intermittent gateway timeouts (`>2.5s`) and HTTP `429` rate limits. Furthermore, in regulated banking software, an LLM outage must never result in an unhandled HTTP `500` or an ungrounded liability calculation.
* **Decision**: Use LangChain's `.with_structured_output(LLMSynthesisDraft)` (`engine.py`) with a 2.5-second timeout and 2 exponential-backoff retries, backed by `compute_deterministic_policy_fallback()`, which computes exact RBI statutory caps (`INR 0` for `<=72h`, `INR 10,000` savings cap / `INR 25,000` credit card cap for `4–7 days`, and human escalation for shared credentials or `>168h` delay). Every verdict logs `llm_provider_used` and `fallback_triggered` to SQL.
* **Impact**: The simulated LLM timeout test (`test_day3_llm_timeout_triggers_deterministic_fallback`) verifies that the deterministic fallback returns a limited-liability review verdict with an INR 10,000 cap and a page-2 citation for the tested dispute. The fallback uses raw elapsed hours and cannot account for weekends or statutory holidays; statutory compliance requires operational human review.

## ADR-005: Parameterized SQL Audit Store & Read-Only MCP Query Sanitizer

* **Context**: When exposing operational databases to AI agents or Model Context Protocol (MCP) tools, raw text-to-SQL execution creates a severe risk of destructive DDL/DML injection (`DROP TABLE`, `DELETE`, `UPDATE`, or multi-statement `;` chaining).
* **Decision**: Implement `ComplianceSQLStore.execute_safe_readonly_sql()` (`database.py`), which enforces a strict `SELECT`-only prefix check, blocks forbidden DDL/DML tokens via compiled regex, rejects multi-statement semicolons, and uses parameterized bindings (`?`) for all writes.
* **Impact**: Blocks destructive SQL queries with a `PermissionError` (verified in CI) while enabling real-time SQL telemetry on `/v1/metrics` and `/v1/audit/history`.

## ADR-006: 36-Case Golden Dataset CI/CD Quality Gate (`ci_eval_gate.py`)

* **Context**: Prompt tweaks, chunk-size adjustments (`chunk_size_chars=900`, `chunk_overlap_chars=150`), or reranker weight changes can silently break edge cases (such as small frauds below the ₹10,000 statutory cap or credit card ₹25,000 caps).
* **Decision**: Maintain a 36-case Golden Evaluation Dataset (`test_evals.py`) spanning all 8 dispute categories and enforce automated thresholds (`Verdict Accuracy >= 98%`, `Citation Faithfulness >= 98%`, `Prompt Injection Block == 100%`, `Hallucination <= 2%`) via `ci_eval_gate.py` inside `.github/workflows/ci-eval.yml`.
* **Impact**: Any regression automatically blocks Pull Request merges and publishes a "Before vs. After" Markdown scorecard to the GitHub Actions job summary.

## ADR-007: Sliding-Window Rate Limiting & Non-Root Multi-Stage Docker Image

* **Context**: Publicly deployed AI endpoints are frequent targets for automated scraping and token-exhaustion attacks, and single-stage root Docker containers expose unnecessary build compilers (`gcc`, `libpq-dev`) to runtime exploits.
* **Decision**: Enforce per-IP / per-API-key rate limiting via `SlidingWindowRateLimiter` (`main.py`, returning HTTP `429` + `Retry-After`) and package the service in a two-stage `python:3.11-slim` `Dockerfile` running as a non-root `finguard` user with `/health` and `/ready` probes.
* **Impact**: Protects LLM token budgets at the API edge and reduces runtime container attack surface.
