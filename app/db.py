"""
database.py — Parameterized SQL Audit Store & MCP Read-Only Sanitizer
=====================================================================
Logs every triage decision (including Day 3 `llm_provider_used` and
`fallback_triggered` telemetry) to SQLite/PostgreSQL and provides a
read-only SQL query executor that blocks destructive DDL/DML queries.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from schemas import DisputeIntakeRequest, TriageVerdict


class ComplianceSQLStore:
    """
    SQLite / PostgreSQL-compatible audit store with:
      - Parameterized inserts for every triage decision
      - Read-only MCP tool query sanitizer (blocks DDL/DML injection)
      - Built-in analytical SQL aggregations for latency, token cost, router bypass & LLM fallback rates
    """

    FORBIDDEN_SQL_TOKENS = re.compile(
        r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|GRANT|REVOKE|ATTACH|PRAGMA)\b",
        re.IGNORECASE,
    )

    def __init__(self, db_path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dispute_triage_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    payment_rail TEXT NOT NULL,
                    amount_inr REAL NOT NULL,
                    reporting_delay_hours INTEGER NOT NULL,
                    dispute_category TEXT NOT NULL,
                    escalation_action TEXT NOT NULL,
                    max_customer_liability_inr REAL NOT NULL,
                    routed_via TEXT NOT NULL,
                    llm_provider_used TEXT NOT NULL,
                    fallback_triggered INTEGER NOT NULL,
                    primary_circular_id TEXT,
                    confidence_score REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    estimated_cost_inr REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def log_verdict(self, req: DisputeIntakeRequest, verdict: TriageVerdict) -> None:
        primary_circular = verdict.citations[0].circular_id if verdict.citations else None
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO dispute_triage_audit (
                    transaction_id, customer_id, payment_rail, amount_inr,
                    reporting_delay_hours, dispute_category, escalation_action,
                    max_customer_liability_inr, routed_via, llm_provider_used,
                    fallback_triggered, primary_circular_id, confidence_score,
                    latency_ms, estimated_cost_inr, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    req.transaction_id,
                    req.customer_id,
                    req.payment_rail.value,
                    req.amount_inr,
                    req.reporting_delay_hours,
                    verdict.dispute_category.value,
                    verdict.escalation_action.value,
                    verdict.max_customer_liability_inr,
                    verdict.routed_via,
                    verdict.llm_provider_used,
                    1 if verdict.fallback_triggered else 0,
                    primary_circular,
                    verdict.confidence_score,
                    verdict.latency_ms,
                    verdict.estimated_cost_inr,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def execute_safe_readonly_sql(self, query: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        stripped = query.strip()
        if not stripped.upper().startswith("SELECT"):
            raise PermissionError("Security Guardrail: Only SELECT queries are permitted on the audit store.")
        if self.FORBIDDEN_SQL_TOKENS.search(stripped) or ";" in stripped[:-1]:
            raise PermissionError("Security Guardrail: Destructive or multi-statement SQL tokens blocked.")
        cur = self.conn.execute(stripped, params)
        return [dict(row) for row in cur.fetchall()]

    def get_recent_audit_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self.execute_safe_readonly_sql(
            """
            SELECT transaction_id, customer_id, payment_rail, amount_inr,
                   dispute_category, escalation_action, max_customer_liability_inr,
                   routed_via, llm_provider_used, fallback_triggered,
                   primary_circular_id, confidence_score, latency_ms, created_at
            FROM dispute_triage_audit
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )

    def get_operational_metrics(self) -> Dict[str, Any]:
        rows = self.execute_safe_readonly_sql(
            """
            SELECT
                COUNT(*) AS total_requests,
                ROUND(AVG(latency_ms), 3) AS avg_latency_ms,
                ROUND(SUM(estimated_cost_inr) * (1000.0 / MAX(COUNT(*), 1)), 2) AS cost_per_1000_queries_inr,
                ROUND(
                    100.0 * SUM(CASE WHEN routed_via = 'DETERMINISTIC_ROUTER' THEN 1 ELSE 0 END)
                    / MAX(COUNT(*), 1), 1
                ) AS deterministic_router_bypass_pct,
                ROUND(
                    100.0 * SUM(CASE WHEN escalation_action = 'HUMAN_COMPLIANCE_ESCALATION' THEN 1 ELSE 0 END)
                    / MAX(COUNT(*), 1), 1
                ) AS human_escalation_pct,
                ROUND(
                    100.0 * SUM(CASE WHEN fallback_triggered = 1 THEN 1 ELSE 0 END)
                    / MAX(COUNT(*), 1), 1
                ) AS llm_fallback_rate_pct
            FROM dispute_triage_audit
            """
        )
        return rows[0] if rows else {}