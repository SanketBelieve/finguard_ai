"""
database.py — Parameterized SQL Store, Customer Risk Profiles & MCP Tool Server
===============================================================================
Implements:
  1. `ComplianceSQLStore`: Parameterized SQLite/PostgreSQL audit store,
     `customer_dispute_history` velocity tracking, `agent_thread_checkpoints`
     for LangGraph human-in-the-loop persistence, and read-only SQL sanitizer.
  2. `FinGuardMCPServer`: Model Context Protocol (MCP) server exposing 3 safe,
     schema-validated SQL tools (`inspect_audit_schema`,
     `get_customer_dispute_history`, `query_readonly_compliance_sql`) with
     automatic audit logging into `mcp_tool_audit_log`.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from schemas import DisputeIntakeRequest, MCPToolCallResult, TriageVerdict
except ImportError:
    from app.schemas import DisputeIntakeRequest, MCPToolCallResult, TriageVerdict


class ComplianceSQLStore:
    """
    SQLite / PostgreSQL-compatible audit & state store with:
      - `dispute_triage_audit`: Parameterized inserts for every triage decision
      - `mcp_tool_audit_log`: Audit log of every MCP tool invocation
      - `agent_thread_checkpoints`: Persistent state checkpoints for LangGraph HITL threads
      - Read-only MCP tool query sanitizer (blocks DDL/DML injection)
    """

    FORBIDDEN_SQL_TOKENS = re.compile(
        r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|GRANT|REVOKE|ATTACH|PRAGMA)\b",
        re.IGNORECASE,
    )

    def __init__(self, db_path: Optional[str] = None) -> None:
        resolved_path = db_path if db_path is not None else os.getenv("FINGUARD_DB_PATH", "finguard_compliance.db")
        self.db_path = resolved_path
        self.conn = sqlite3.connect(resolved_path, check_same_thread=False)
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
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mcp_tool_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    execution_ms REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_thread_checkpoints (
                    thread_id TEXT PRIMARY KEY,
                    workflow_status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
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

    def log_mcp_tool_call(
        self, tool_name: str, arguments: Dict[str, Any], status: str, execution_ms: float
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO mcp_tool_audit_log (tool_name, arguments_json, status, execution_ms, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    tool_name,
                    json.dumps(arguments),
                    status,
                    execution_ms,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def save_thread_checkpoint(
        self, thread_id: str, workflow_status: str, request_dict: Dict[str, Any], state_dict: Dict[str, Any]
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO agent_thread_checkpoints (
                    thread_id, workflow_status, request_json, state_json, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    workflow_status,
                    json.dumps(request_dict),
                    json.dumps(state_dict),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def load_thread_checkpoint(self, thread_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT thread_id, workflow_status, request_json, state_json, updated_at
            FROM agent_thread_checkpoints
            WHERE thread_id = ?
            """,
            (thread_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "thread_id": row["thread_id"],
            "workflow_status": row["workflow_status"],
            "request": json.loads(row["request_json"]),
            "state": json.loads(row["state_json"]),
            "updated_at": row["updated_at"],
        }

    def execute_safe_readonly_sql(
        self,
        query: str,
        params: Tuple[Any, ...] = (),
        max_rows: int = 1000,
        timeout_seconds: float = 5.0,
    ) -> List[Dict[str, Any]]:
        stripped = query.strip()
        if not stripped.upper().startswith("SELECT"):
            raise PermissionError("Security Guardrail: Only SELECT queries are permitted on the audit store.")
        if self.FORBIDDEN_SQL_TOKENS.search(stripped) or ";" in stripped[:-1]:
            raise PermissionError("Security Guardrail: Destructive or multi-statement SQL tokens blocked.")

        def authorizer_callback(action: int, arg1: Any, arg2: Any, dbname: Any, trigger: Any) -> int:
            if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION):
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ:
                # Restrict read access only to dispute_triage_audit
                if arg1 in ("dispute_triage_audit", None, ""):
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_DENY

        start_time = time.perf_counter()

        def progress_callback() -> int:
            if time.perf_counter() - start_time > timeout_seconds:
                return 1  # Abort query execution
            return 0

        self.conn.set_authorizer(authorizer_callback)
        self.conn.set_progress_handler(progress_callback, 1000)

        try:
            cur = self.conn.execute(stripped, params)
            rows = cur.fetchmany(max_rows)
            return [dict(row) for row in rows]
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise TimeoutError(f"Security Guardrail: SQL query exceeded {timeout_seconds}s timeout limit.") from exc
            raise
        except sqlite3.DatabaseError as exc:
            err_msg = str(exc).lower()
            if "not authorized" in err_msg or "prohibited" in err_msg:
                raise PermissionError(f"Security Guardrail: Query rejected by SQLite security authorizer: {exc}") from exc
            raise
        finally:
            self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
            self.conn.set_progress_handler(None, 0)

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


# ============================================================================
# PHASE 2: MODEL CONTEXT PROTOCOL (MCP) SQL TOOL SERVER
# ============================================================================

class FinGuardMCPServer:
    """
    Model Context Protocol (MCP) Server exposing read-only SQL and customer
    velocity risk tools to LLM agents (compatible with `mcp.server.fastmcp.FastMCP`
    and JSON-RPC 2.0 tool discovery).
    """

    MCP_TOOLS_MANIFEST: List[Dict[str, Any]] = [
        {
            "name": "inspect_audit_schema",
            "description": "Returns the SQL table names and column schemas available for read-only compliance queries.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_customer_dispute_history",
            "description": (
                "Queries parameterized SQL history for `customer_id`, returning prior dispute count, "
                "total disputed INR volume, and velocity abuse risk tier (LOW, MEDIUM, HIGH_VELOCITY_ABUSE)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"customer_id": {"type": "string", "description": "Anonymized customer ID"}},
                "required": ["customer_id"],
            },
        },
        {
            "name": "query_readonly_compliance_sql",
            "description": (
                "Executes a read-only SELECT analytical query against `dispute_triage_audit`. "
                "Blocks any DDL/DML or multi-statement SQL injection."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"sql_query": {"type": "string", "description": "Parameterized SELECT SQL query"}},
                "required": ["sql_query"],
            },
        },
    ]

    def __init__(self, sql_store: ComplianceSQLStore) -> None:
        self.sql_store = sql_store

    def list_tools(self) -> List[Dict[str, Any]]:
        return list(self.MCP_TOOLS_MANIFEST)

    def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> MCPToolCallResult:
        t0 = time.perf_counter()
        args = arguments or {}

        try:
            if tool_name == "inspect_audit_schema":
                data = {
                    "tables": {
                        "dispute_triage_audit": [
                            "id",
                            "transaction_id",
                            "customer_id",
                            "payment_rail",
                            "amount_inr",
                            "reporting_delay_hours",
                            "dispute_category",
                            "escalation_action",
                            "max_customer_liability_inr",
                            "routed_via",
                            "llm_provider_used",
                            "fallback_triggered",
                            "primary_circular_id",
                            "confidence_score",
                            "latency_ms",
                            "estimated_cost_inr",
                            "created_at",
                        ],
                        "agent_thread_checkpoints": [
                            "thread_id",
                            "workflow_status",
                            "request_json",
                            "state_json",
                            "updated_at",
                        ],
                    }
                }
                ms = round((time.perf_counter() - t0) * 1000, 3)
                self.sql_store.log_mcp_tool_call(tool_name, args, "SUCCESS", ms)
                return MCPToolCallResult(tool_name=tool_name, status="SUCCESS", data=data, execution_ms=ms)

            if tool_name == "get_customer_dispute_history":
                customer_id = str(args.get("customer_id", "")).strip()
                if not customer_id:
                    raise ValueError("Missing required argument: customer_id")

                rows = self.sql_store.execute_safe_readonly_sql(
                    """
                    SELECT
                        COUNT(DISTINCT transaction_id) AS prior_disputes_count,
                        COALESCE(SUM(amount_inr), 0.0) AS total_disputed_inr,
                        COALESCE(SUM(CASE WHEN dispute_category = 'UNAUTHORIZED_FRAUD' THEN 1 ELSE 0 END), 0) AS prior_fraud_claims
                    FROM (
                        SELECT transaction_id, MAX(amount_inr) AS amount_inr, MAX(dispute_category) AS dispute_category
                        FROM dispute_triage_audit
                        WHERE customer_id = ?
                          AND datetime(created_at) >= datetime('now', '-90 days')
                        GROUP BY transaction_id
                    )
                    """,
                    (customer_id,),
                )
                stats = rows[0] if rows else {"prior_disputes_count": 0, "total_disputed_inr": 0.0, "prior_fraud_claims": 0}
                prior_count = int(stats.get("prior_disputes_count") or 0)
                total_inr = float(stats.get("total_disputed_inr") or 0.0)

                if prior_count >= 3 or total_inr >= 200_000.0:
                    risk_tier = "HIGH_VELOCITY_ABUSE"
                elif prior_count >= 1 or total_inr >= 50_000.0:
                    risk_tier = "MEDIUM"
                else:
                    risk_tier = "LOW"

                data = {
                    "customer_id": customer_id,
                    "prior_disputes_90d": prior_count,
                    "total_disputed_inr": round(total_inr, 2),
                    "velocity_risk_tier": risk_tier,
                }
                ms = round((time.perf_counter() - t0) * 1000, 3)
                self.sql_store.log_mcp_tool_call(tool_name, args, "SUCCESS", ms)
                return MCPToolCallResult(tool_name=tool_name, status="SUCCESS", data=data, execution_ms=ms)

            if tool_name == "query_readonly_compliance_sql":
                sql_query = str(args.get("sql_query", "")).strip()
                rows = self.sql_store.execute_safe_readonly_sql(sql_query)
                ms = round((time.perf_counter() - t0) * 1000, 3)
                self.sql_store.log_mcp_tool_call(tool_name, args, "SUCCESS", ms)
                return MCPToolCallResult(tool_name=tool_name, status="SUCCESS", data=rows, execution_ms=ms)

            raise ValueError(f"Unknown MCP tool: {tool_name}")

        except PermissionError as exc:
            ms = round((time.perf_counter() - t0) * 1000, 3)
            self.sql_store.log_mcp_tool_call(tool_name, args, "BLOCKED_BY_GUARDRAIL", ms)
            return MCPToolCallResult(
                tool_name=tool_name,
                status="BLOCKED_BY_GUARDRAIL",
                data=None,
                error_message=str(exc),
                execution_ms=ms,
            )
        except Exception as exc:
            ms = round((time.perf_counter() - t0) * 1000, 3)
            self.sql_store.log_mcp_tool_call(tool_name, args, "ERROR", ms)
            return MCPToolCallResult(
                tool_name=tool_name,
                status="ERROR",
                data=None,
                error_message=str(exc),
                execution_ms=ms,
            )