"""
schemas.py — Strict Pydantic Schemas & Validators for FinGuard AI
=================================================================
Defines request/response contracts, enums, currency coercion, and mandatory
RBI regulatory citation grounding checks (compatible with Pydantic v1 & v2).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

try:
    from pydantic import field_validator, model_validator  # type: ignore

    PYDANTIC_V2 = True
except ImportError:
    from pydantic import root_validator, validator  # type: ignore

    PYDANTIC_V2 = False


class PaymentRail(str, Enum):
    UPI = "UPI"
    IMPS = "IMPS"
    NEFT = "NEFT"
    RTGS = "RTGS"
    CREDIT_CARD = "CREDIT_CARD"


class DisputeCategory(str, Enum):
    UNAUTHORIZED_FRAUD = "UNAUTHORIZED_FRAUD"
    FAILED_DEBIT_NOT_CREDITED = "FAILED_DEBIT_NOT_CREDITED"
    DUPLICATE_CHARGE = "DUPLICATE_CHARGE"
    MERCHANT_NON_DELIVERY = "MERCHANT_NON_DELIVERY"
    OUT_OF_SCOPE_QUERY = "OUT_OF_SCOPE_QUERY"


class EscalationAction(str, Enum):
    AUTO_RESOLVE_T_PLUS_1 = "AUTO_RESOLVE_T_PLUS_1"
    ZERO_LIABILITY_SHADOW_CREDIT = "ZERO_LIABILITY_SHADOW_CREDIT"
    LIMITED_LIABILITY_REVIEW = "LIMITED_LIABILITY_REVIEW"
    HUMAN_COMPLIANCE_ESCALATION = "HUMAN_COMPLIANCE_ESCALATION"
    REJECT_OUT_OF_SCOPE = "REJECT_OUT_OF_SCOPE"


class DisputeIntakeRequest(BaseModel):
    """Validated inbound dispute payload from web UI or webhook."""

    customer_id: str = Field(..., min_length=4, max_length=32, description="Anonymized customer identifier")
    transaction_id: str = Field(..., min_length=6, max_length=64, description="UTR or RRN reference")
    payment_rail: PaymentRail = Field(..., description="Indian payment rail (UPI, IMPS, NEFT, RTGS, CREDIT_CARD)")
    amount_inr: float = Field(..., gt=0, le=10_000_000, description="Disputed amount in INR")
    reporting_delay_hours: int = Field(
        ..., ge=0, le=8760, description="Hours elapsed between transaction alert and customer report"
    )
    customer_narrative: str = Field(
        ..., min_length=10, max_length=2000, description="Free-text customer complaint narrative"
    )

    if PYDANTIC_V2:
        @field_validator("amount_inr", mode="before")  # type: ignore[misc]
        @classmethod
        def _coerce_amount_v2(cls, v: Any) -> float:
            if isinstance(v, str):
                return float(re.sub(r"[₹,\s]", "", v))
            return float(v)
    else:
        @validator("amount_inr", pre=True)  # type: ignore[misc]
        def _coerce_amount_v1(cls, v: Any) -> float:
            if isinstance(v, str):
                return float(re.sub(r"[₹,\s]", "", v))
            return float(v)


class RBIComplianceCitation(BaseModel):
    """Verifiable regulatory source chunk attached to every compliance decision."""

    circular_id: str = Field(..., description="Official RBI circular or direction identifier")
    title: str = Field(..., description="Title of the regulatory clause")
    section_ref: str = Field(..., description="Exact section or paragraph reference")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Hybrid retrieval + reranker score")


class TriageVerdict(BaseModel):
    """Structured output returned by the FinGuard AI pipeline."""

    transaction_id: str
    dispute_category: DisputeCategory
    escalation_action: EscalationAction
    max_customer_liability_inr: float = Field(..., ge=0.0)
    provisional_credit_due_days: int = Field(..., ge=0, le=90)
    compensation_penalty_per_day_inr: float = Field(..., ge=0.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    routed_via: Literal["DETERMINISTIC_ROUTER", "HYBRID_RAG_AGENT", "GUARDRAIL_FALLBACK"]
    citations: List[RBIComplianceCitation] = Field(default_factory=list)
    rationale: str = Field(..., min_length=15)
    latency_ms: float = Field(..., ge=0.0)
    estimated_cost_inr: float = Field(..., ge=0.0)

    if PYDANTIC_V2:
        @model_validator(mode="after")  # type: ignore[misc]
        def _check_citations_v2(self) -> "TriageVerdict":
            if (
                self.dispute_category != DisputeCategory.OUT_OF_SCOPE_QUERY
                and self.routed_via != "GUARDRAIL_FALLBACK"
                and len(self.citations) == 0
            ):
                raise ValueError("Compliance verdicts must include at least one verified RBI citation.")
            return self
    else:
        @root_validator  # type: ignore[misc]
        def _check_citations_v1(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            cat = values.get("dispute_category")
            route = values.get("routed_via")
            cits = values.get("citations") or []
            if cat != DisputeCategory.OUT_OF_SCOPE_QUERY and route != "GUARDRAIL_FALLBACK" and len(cits) == 0:
                raise ValueError("Compliance verdicts must include at least one verified RBI citation.")
            return values