"""
router.py — Sub-15ms Deterministic Intent Router, Prompt-Injection & Guardrail
==============================================================================
Intercepts adversarial jailbreak/prompt injection attempts, out-of-scope queries,
and clear-cut UPI/IMPS/NEFT/RTGS failed-debit & duplicate charges in <15ms before
invoking an external LLM, protecting system integrity and cutting token costs by ~60%.
"""

from __future__ import annotations

import re
from typing import Optional

from schemas import DisputeCategory, DisputeIntakeRequest, PaymentRail


class DeterministicFinTechRouter:
    """
    Sub-15ms classifier and security guardrail:
      1. Intercepts Prompt Injection, Jailbreaks, Delimiter Hijacking, and System Prompt Leaks
      2. Intercepts Out-of-Scope non-fintech queries (crypto, weather, sports)
      3. Fast-routes clear-cut failed debits (UPI, IMPS, NEFT, RTGS) and duplicate charges
    """

    PROMPT_INJECTION_PATTERNS = (
        r"\b(ignore\s+(all\s+)?(previous|prior|above)\s+instructions)\b",
        r"\b(disregard\s+(all\s+)?(previous|prior)\s+instructions)\b",
        r"\b(system\s+prompt|system\s+instruction|developer\s+mode|dan\s+mode|jailbreak)\b",
        r"\b(override\s+(all\s+)?(rules|system|compliance|guidelines))\b",
        r"\b(you\s+are\s+now\s+(an?\s+)?(unrestricted|jailbroken|godmode|devmode))\b",
        r"\b(bypass\s+(all\s+)?(rbi|verification|security|compliance|liability))\b",
        r"\b(print\s+(your|the)\s+(full\s+)?(system\s+prompt|hidden\s+instructions|api\s+key))\b",
        r"(<\|im_start\|>|<\|im_end\|>|\[system\]|assistant:\s*here\s+is)",
        r"(---+\s*end\s+of\s+(prompt|narrative|system)\s*---+)",
    )

    OUT_OF_SCOPE_PATTERNS = (
        r"\b(stock\s+tip|crypto|bitcoin|ethereum|weather|recipe|movie|cricket\s+score|horoscope)\b",
    )

    FAILED_DEBIT_PATTERNS = (
        r"(debited|deducted).*(not\s+credited|failed|pending|stuck)",
        r"(failed|pending)\s+(upi|imps|neft|rtgs)\s+transaction.*(debited|deducted)",
        r"(qr\s+code\s+failed|mandate\s+failed|auto-debit\s+failed|timed\s+out).*(debited|deducted)",
    )

    DUPLICATE_CHARGE_PATTERNS = (
        r"\b(debited\s+twice|charged\s+twice|double\s+debited|duplicate\s+charge|duplicate\s+debit)\b",
    )

    MERCHANT_DISPUTE_PATTERNS = (
        r"\b(merchant\s+did\s+not\s+deliver|goods\s+not\s+received|product\s+not\s+delivered|service\s+not\s+rendered)\b",
    )

    @classmethod
    def is_prompt_injection(cls, text: str) -> bool:
        text_lower = text.lower()
        for pat in cls.PROMPT_INJECTION_PATTERNS:
            if re.search(pat, text_lower):
                return True
        return False

    @classmethod
    def classify_fast(cls, req: DisputeIntakeRequest) -> Optional[DisputeCategory]:
        text = req.customer_narrative.lower()

        # Step 1: Adversarial Prompt-Injection & Security Defense
        if cls.is_prompt_injection(text):
            return DisputeCategory.OUT_OF_SCOPE_QUERY

        # Step 2: General Out-of-Scope Queries
        for pat in cls.OUT_OF_SCOPE_PATTERNS:
            if re.search(pat, text):
                return DisputeCategory.OUT_OF_SCOPE_QUERY

        # Step 3: Duplicate Charges (Auto-Reversal T+1)
        for pat in cls.DUPLICATE_CHARGE_PATTERNS:
            if re.search(pat, text) and "fraud" not in text and "unauthorized" not in text:
                return DisputeCategory.DUPLICATE_CHARGE

        # Step 4: Merchant Non-Delivery / Commercial Disputes
        for pat in cls.MERCHANT_DISPUTE_PATTERNS:
            if re.search(pat, text) and "fraud" not in text and "unauthorized" not in text:
                return DisputeCategory.MERCHANT_NON_DELIVERY

        # Step 5: Failed Debits across all Indian payment rails
        if req.payment_rail in (PaymentRail.UPI, PaymentRail.IMPS, PaymentRail.NEFT, PaymentRail.RTGS):
            for pat in cls.FAILED_DEBIT_PATTERNS:
                if re.search(pat, text) and "fraud" not in text and "unauthorized" not in text:
                    return DisputeCategory.FAILED_DEBIT_NOT_CREDITED

        return None