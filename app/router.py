"""
router.py — Sub-15ms Deterministic Intent Router & Domain Guardrail
===================================================================
Intercepts clear-cut UPI/IMPS failed-debit TAT claims and out-of-scope
queries before invoking an external LLM, cutting token costs by 50–65%.
"""

from __future__ import annotations

import re
from typing import Optional

from schemas import DisputeCategory, DisputeIntakeRequest, PaymentRail


class DeterministicFinTechRouter:
    """
    Routes clear-cut failed-debit TAT cases and out-of-scope queries in <15ms
    without calling an external LLM API.
    """

    OUT_OF_SCOPE_PATTERNS = (
        r"\b(stock tip|crypto|bitcoin|weather|recipe|movie|cricket score)\b",
    )
    FAILED_DEBIT_PATTERNS = (
        r"(debited|deducted).*(not credited|failed|pending|stuck)",
        r"(failed|pending) (upi|imps) transaction.*(debited|deducted)",
    )

    @classmethod
    def classify_fast(cls, req: DisputeIntakeRequest) -> Optional[DisputeCategory]:
        """Return a category for a recognized narrative, or None for full triage.

        Out-of-scope patterns take precedence. Failed-debit matches require UPI
        or IMPS, an amount at most INR 50,000, and neither 'fraud' nor
        'unauthorized' in the narrative. Matching is case-insensitive.
        """
        text = req.customer_narrative.lower()
        for pat in cls.OUT_OF_SCOPE_PATTERNS:
            if re.search(pat, text):
                return DisputeCategory.OUT_OF_SCOPE_QUERY
        if req.payment_rail in (PaymentRail.UPI, PaymentRail.IMPS) and req.amount_inr <= 50_000:
            for pat in cls.FAILED_DEBIT_PATTERNS:
                if re.search(pat, text) and "fraud" not in text and "unauthorized" not in text:
                    return DisputeCategory.FAILED_DEBIT_NOT_CREDITED
        return None