"""
test_evals.py — Golden Dataset Evaluation Suite & Unit/Security Tests
=====================================================================
Run standalone benchmark & tests:
    python3 test_evals.py

Run via pytest:
    pytest test_evals.py -v
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from database import ComplianceSQLStore
from engine import FinGuardTriageEngine
from schemas import (
    DisputeCategory,
    DisputeIntakeRequest,
    EscalationAction,
    PaymentRail,
)


@dataclass(frozen=True)
class GoldenEvalCase:
    case_id: str
    request: DisputeIntakeRequest
    expected_category: DisputeCategory
    expected_action: EscalationAction
    expected_max_liability_inr: float
    expected_circular_prefix: Optional[str]


GOLDEN_EVAL_DATASET: List[GoldenEvalCase] = [
    GoldenEvalCase(
        case_id="EVAL-01 (UPI Failed Debit T+1)",
        request=DisputeIntakeRequest(
            customer_id="CUST-1001",
            transaction_id="UTR629100112",
            payment_rail=PaymentRail.UPI,
            amount_inr=4500.0,
            reporting_delay_hours=12,
            customer_narrative="My account was debited INR 4,500 via UPI but the merchant account was not credited and status shows failed.",
        ),
        expected_category=DisputeCategory.FAILED_DEBIT_NOT_CREDITED,
        expected_action=EscalationAction.AUTO_RESOLVE_T_PLUS_1,
        expected_max_liability_inr=0.0,
        expected_circular_prefix="RBI/2019-20/67",
    ),
    GoldenEvalCase(
        case_id="EVAL-02 (Unauthorized Fraud <72h Zero Liability)",
        request=DisputeIntakeRequest(
            customer_id="CUST-1002",
            transaction_id="UTR998811223",
            payment_rail=PaymentRail.IMPS,
            amount_inr=42000.0,
            reporting_delay_hours=18,
            customer_narrative="Unauthorized IMPS fraud transaction of Rs 42,000 happened from my account due to a third-party breach without my knowledge.",
        ),
        expected_category=DisputeCategory.UNAUTHORIZED_FRAUD,
        expected_action=EscalationAction.ZERO_LIABILITY_SHADOW_CREDIT,
        expected_max_liability_inr=0.0,
        expected_circular_prefix="RBI/2017-18/15",
    ),
    GoldenEvalCase(
        case_id="EVAL-03 (Unauthorized Fraud 4-7 Days Limited Liability)",
        request=DisputeIntakeRequest(
            customer_id="CUST-1003",
            transaction_id="UTR774411990",
            payment_rail=PaymentRail.UPI,
            amount_inr=35000.0,
            reporting_delay_hours=110,
            customer_narrative="I noticed an unauthorized fraud transaction on my savings account from 5 days ago due to a phishing attack.",
        ),
        expected_category=DisputeCategory.UNAUTHORIZED_FRAUD,
        expected_action=EscalationAction.LIMITED_LIABILITY_REVIEW,
        expected_max_liability_inr=10000.0,
        expected_circular_prefix="RBI/2017-18/15",
    ),
    GoldenEvalCase(
        case_id="EVAL-04 (Shared OTP Customer Negligence Escalation)",
        request=DisputeIntakeRequest(
            customer_id="CUST-1004",
            transaction_id="UTR552299881",
            payment_rail=PaymentRail.UPI,
            amount_inr=18500.0,
            reporting_delay_hours=6,
            customer_narrative="A scammer called pretending to be bank support and I shared OTP and entered UPI PIN, losing Rs 18,500.",
        ),
        expected_category=DisputeCategory.UNAUTHORIZED_FRAUD,
        expected_action=EscalationAction.HUMAN_COMPLIANCE_ESCALATION,
        expected_max_liability_inr=18500.0,
        expected_circular_prefix="RBI/2017-18/15",
    ),
    GoldenEvalCase(
        case_id="EVAL-05 (Out-of-Scope Crypto/Weather Guardrail)",
        request=DisputeIntakeRequest(
            customer_id="CUST-1005",
            transaction_id="UTR000011112",
            payment_rail=PaymentRail.UPI,
            amount_inr=100.0,
            reporting_delay_hours=1,
            customer_narrative="Can you give me a bitcoin crypto stock tip and tell me today's cricket score in Pune?",
        ),
        expected_category=DisputeCategory.OUT_OF_SCOPE_QUERY,
        expected_action=EscalationAction.REJECT_OUT_OF_SCOPE,
        expected_max_liability_inr=0.0,
        expected_circular_prefix=None,
    ),
]


def run_evaluation_suite() -> Dict[str, Any]:
    engine = FinGuardTriageEngine()
    results = []
    latencies = []
    exact_matches = 0
    citation_hits = 0
    hallucinations = 0

    for case in GOLDEN_EVAL_DATASET:
        verdict = engine.process_dispute(case.request)
        latencies.append(verdict.latency_ms)

        action_ok = (
            verdict.dispute_category == case.expected_category
            and verdict.escalation_action == case.expected_action
            and math.isclose(verdict.max_customer_liability_inr, case.expected_max_liability_inr, abs_tol=0.01)
        )
        if action_ok:
            exact_matches += 1

        if case.expected_circular_prefix is None:
            cit_ok = len(verdict.citations) == 0
        else:
            cit_ok = any(c.circular_id.startswith(case.expected_circular_prefix) for c in verdict.citations)
            if not cit_ok:
                hallucinations += 1
        if cit_ok:
            citation_hits += 1

        results.append(
            {
                "case_id": case.case_id,
                "routed_via": verdict.routed_via,
                "action": verdict.escalation_action.value,
                "liability_inr": verdict.max_customer_liability_inr,
                "citation": verdict.citations[0].circular_id if verdict.citations else "NONE (Guardrail)",
                "passed": action_ok and cit_ok,
            }
        )

    total = len(GOLDEN_EVAL_DATASET)
    sorted_lat = sorted(latencies)
    p50_ms = sorted_lat[len(sorted_lat) // 2]
    p95_ms = sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))]
    op_metrics = engine.sql_store.get_operational_metrics()

    return {
        "total_cases": total,
        "verdict_accuracy_pct": round(100.0 * exact_matches / total, 1),
        "citation_faithfulness_pct": round(100.0 * citation_hits / total, 1),
        "hallucination_rate_pct": round(100.0 * hallucinations / total, 1),
        "p50_latency_ms": round(p50_ms, 3),
        "p95_latency_ms": round(p95_ms, 3),
        "sql_operational_metrics": op_metrics,
        "case_breakdown": results,
    }


def test_eval_suite_passes_quality_gates() -> None:
    report = run_evaluation_suite()
    assert report["verdict_accuracy_pct"] == 100.0
    assert report["citation_faithfulness_pct"] == 100.0
    assert report["hallucination_rate_pct"] == 0.0


def test_pydantic_coerces_formatted_inr_strings() -> None:
    req = DisputeIntakeRequest(
        customer_id="CUST-9001",
        transaction_id="UTR123456789",
        payment_rail=PaymentRail.UPI,
        amount_inr="₹ 12,500.50",  # type: ignore[arg-type]
        reporting_delay_hours=8,
        customer_narrative="My account was debited ₹12,500.50 via UPI but merchant was not credited.",
    )
    assert math.isclose(req.amount_inr, 12500.50)


def test_sql_guardrail_blocks_destructive_queries() -> None:
    store = ComplianceSQLStore()
    try:
        store.execute_safe_readonly_sql("DROP TABLE dispute_triage_audit")
        assert False, "Expected PermissionError for DROP TABLE"
    except PermissionError:
        pass


if __name__ == "__main__":
    test_eval_suite_passes_quality_gates()
    test_pydantic_coerces_formatted_inr_strings()
    test_sql_guardrail_blocks_destructive_queries()

    summary = run_evaluation_suite()
    print("=" * 78)
    print("MODULAR FINGUARD AI — GOLDEN DATASET EVALUATION & SQL AUDIT BENCHMARK")
    print("=" * 78)
    print("Unit & SQL Security Tests  : 3 / 3 PASSED")
    print(f"Total Golden Eval Cases    : {summary['total_cases']}")
    print(f"Verdict Accuracy           : {summary['verdict_accuracy_pct']}%")
    print(f"Citation Faithfulness      : {summary['citation_faithfulness_pct']}%")
    print(f"Hallucination Rate         : {summary['hallucination_rate_pct']}%")
    print(f"Latency (p50 / p95)        : {summary['p50_latency_ms']} ms / {summary['p95_latency_ms']} ms")
    print(f"SQL Audit Store Metrics    : {json.dumps(summary['sql_operational_metrics'])}")
    print("-" * 78)
    for row in summary["case_breakdown"]:
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"[{status}] {row['case_id']:<48} | Route: {row['routed_via']:<20} "
            f"| Action: {row['action']:<28} | Max Liab: ₹{row['liability_inr']:,.0f}"
        )
    print("=" * 78)