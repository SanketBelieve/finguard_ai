"""
engine.py — Core FinGuard AI Triage & RBI Liability Orchestrator
================================================================
Coordinates the Deterministic Router, Hybrid RAG Retriever, RBI Liability
Rules, Pydantic Output Verification, and SQL Audit Store.
"""

from __future__ import annotations

import time

from database import ComplianceSQLStore
from retriever import HybridComplianceRetriever, RBI_REGULATORY_CORPUS
from router import DeterministicFinTechRouter
from schemas import (
    DisputeCategory,
    DisputeIntakeRequest,
    EscalationAction,
    PaymentRail,
    TriageVerdict,
)


class FinGuardTriageEngine:
    """
    Coordinates:
      Step 1 -> Deterministic Intent Router (<15ms path for standard TAT & out-of-scope)
      Step 2 -> Hybrid RAG + Cross-Encoder Reranking over RBI circulars
      Step 3 -> Liability & TAT policy engine + Pydantic schema verification
      Step 4 -> Parameterized SQL audit logging
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.retriever = HybridComplianceRetriever(RBI_REGULATORY_CORPUS)
        self.sql_store = ComplianceSQLStore(db_path=db_path)

    def process_dispute(self, req: DisputeIntakeRequest) -> TriageVerdict:
        t0 = time.perf_counter()
        narrative_lower = req.customer_narrative.lower()

        # Step 1: Try deterministic sub-15ms router first
        fast_cat = DeterministicFinTechRouter.classify_fast(req)
        if fast_cat == DisputeCategory.OUT_OF_SCOPE_QUERY:
            latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            verdict = TriageVerdict(
                transaction_id=req.transaction_id,
                dispute_category=DisputeCategory.OUT_OF_SCOPE_QUERY,
                escalation_action=EscalationAction.REJECT_OUT_OF_SCOPE,
                max_customer_liability_inr=0.0,
                provisional_credit_due_days=0,
                compensation_penalty_per_day_inr=0.0,
                confidence_score=0.99,
                routed_via="GUARDRAIL_FALLBACK",
                citations=[],
                rationale=(
                    "Out-of-scope query blocked by deterministic domain guardrail. "
                    "FinGuard AI only processes regulated Indian payment disputes and RBI compliance rules."
                ),
                latency_ms=latency_ms,
                estimated_cost_inr=0.0,
            )
            self.sql_store.log_verdict(req, verdict)
            return verdict

        # Step 2: Retrieve & rerank RBI compliance circulars
        citations = self.retriever.retrieve_and_rerank(
            query=req.customer_narrative,
            payment_rail=req.payment_rail,
            delay_hours=req.reporting_delay_hours,
            top_k=2,
        )

        if not citations:
            latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            verdict = TriageVerdict(
                transaction_id=req.transaction_id,
                dispute_category=DisputeCategory.OUT_OF_SCOPE_QUERY,
                escalation_action=EscalationAction.HUMAN_COMPLIANCE_ESCALATION,
                max_customer_liability_inr=0.0,
                provisional_credit_due_days=10,
                compensation_penalty_per_day_inr=0.0,
                confidence_score=0.40,
                routed_via="GUARDRAIL_FALLBACK",
                citations=[],
                rationale="Insufficient regulatory context retrieved; escalated to human compliance officer.",
                latency_ms=latency_ms,
                estimated_cost_inr=0.0,
            )
            self.sql_store.log_verdict(req, verdict)
            return verdict

        # Step 3: Fast-routed failed debit vs. Hybrid RAG agent synthesis
        if fast_cat == DisputeCategory.FAILED_DEBIT_NOT_CREDITED:
            latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            verdict = TriageVerdict(
                transaction_id=req.transaction_id,
                dispute_category=DisputeCategory.FAILED_DEBIT_NOT_CREDITED,
                escalation_action=EscalationAction.AUTO_RESOLVE_T_PLUS_1,
                max_customer_liability_inr=0.0,
                provisional_credit_due_days=1,
                compensation_penalty_per_day_inr=100.0,
                confidence_score=0.96,
                routed_via="DETERMINISTIC_ROUTER",
                citations=citations,
                rationale=(
                    f"Deterministically routed {req.payment_rail.value} failed-debit dispute under "
                    f"{citations[0].circular_id} ({citations[0].section_ref}): mandatory T+1 auto-reversal "
                    f"with INR 100/day delay compensation beyond T+1."
                ),
                latency_ms=latency_ms,
                estimated_cost_inr=0.001,
            )
            self.sql_store.log_verdict(req, verdict)
            return verdict

        # Step 4: Unauthorized Fraud & Complex Dispute Evaluation (Hybrid RAG Agent Path)
        customer_shared_credentials = any(
            phrase in narrative_lower for phrase in ("shared otp", "gave otp", "shared pin", "entered upi pin")
        )
        is_unauthorized = any(
            kw in narrative_lower
            for kw in ("unauthorized", "fraud", "hacked", "phishing", "unknown", "without my knowledge", "stolen")
        )

        if is_unauthorized or customer_shared_credentials:
            category = DisputeCategory.UNAUTHORIZED_FRAUD
            if customer_shared_credentials or req.reporting_delay_hours > 168:
                action = EscalationAction.HUMAN_COMPLIANCE_ESCALATION
                max_liability = req.amount_inr if customer_shared_credentials else min(req.amount_inr, 25_000.0)
                rationale = (
                    f"Escalated under {citations[0].circular_id} ({citations[0].section_ref}): "
                    f"{'Customer credential sharing detected' if customer_shared_credentials else 'Reporting delay >7 working days (>168h)'}. "
                    f"Requires human compliance verification and post-reporting loss freeze."
                )
            elif req.reporting_delay_hours <= 72:
                action = EscalationAction.ZERO_LIABILITY_SHADOW_CREDIT
                max_liability = 0.0
                rationale = (
                    f"Zero Customer Liability qualified under {citations[0].circular_id} ({citations[0].section_ref}): "
                    f"reported within {req.reporting_delay_hours}h (<=72h window). Mandatory shadow credit within 10 working days."
                )
            else:
                action = EscalationAction.LIMITED_LIABILITY_REVIEW
                cap = 25_000.0 if req.payment_rail == PaymentRail.CREDIT_CARD else 10_000.0
                max_liability = min(req.amount_inr, cap)
                rationale = (
                    f"Limited Customer Liability applied under {citations[0].circular_id} ({citations[0].section_ref}): "
                    f"reported within {req.reporting_delay_hours}h (4–7 working day window). "
                    f"Customer liability capped at INR {max_liability:,.2f}; shadow reversal due in 10 working days."
                )
            credit_days = 10
            penalty_per_day = 0.0
        else:
            category = DisputeCategory.FAILED_DEBIT_NOT_CREDITED
            action = EscalationAction.AUTO_RESOLVE_T_PLUS_1
            max_liability = 0.0
            credit_days = 1
            penalty_per_day = 100.0
            rationale = (
                f"Classified under {citations[0].circular_id} ({citations[0].section_ref}) via Hybrid RAG: "
                f"T+1 reversal rule applies with INR 100/day penalty for delayed settlement."
            )

        latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        verdict = TriageVerdict(
            transaction_id=req.transaction_id,
            dispute_category=category,
            escalation_action=action,
            max_customer_liability_inr=max_liability,
            provisional_credit_due_days=credit_days,
            compensation_penalty_per_day_inr=penalty_per_day,
            confidence_score=citations[0].relevance_score,
            routed_via="HYBRID_RAG_AGENT",
            citations=citations,
            rationale=rationale,
            latency_ms=latency_ms,
            estimated_cost_inr=0.028,
        )
        self.sql_store.log_verdict(req, verdict)
        return verdict