"""
retriever.py — RBI Regulatory Corpus & Hybrid BM25 + Vector Retriever
=====================================================================
Combines lexical keyword scoring, dense n-gram cosine similarity, and
cross-encoder domain feature reranking over RBI Master Directions.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from schemas import PaymentRail, RBIComplianceCitation


@dataclass(frozen=True)
class RegulatoryDocumentChunk:
    circular_id: str
    title: str
    section_ref: str
    content: str
    keywords: Tuple[str, ...]


RBI_REGULATORY_CORPUS: List[RegulatoryDocumentChunk] = [
    RegulatoryDocumentChunk(
        circular_id="RBI/2019-20/67-DPSS.CO.PD.No.629",
        title="Harmonisation of Turn Around Time (TAT) and Customer Compensation for Failed Transactions",
        section_ref="Section 2(a) — UPI Account Debited but Beneficiary Not Credited",
        content=(
            "For Unified Payments Interface (UPI) transactions where the customer account is debited "
            "but the beneficiary account is not credited (failed transaction), the bank must auto-reverse "
            "the transaction within T+1 calendar days. Failing auto-reversal within T+1 days, the issuer "
            "bank shall pay a penalty compensation of INR 100 per day of delay to the customer unconditionally."
        ),
        keywords=("upi", "failed", "debited", "not credited", "t+1", "100", "penalty", "auto-reverse", "tat"),
    ),
    RegulatoryDocumentChunk(
        circular_id="RBI/2019-20/67-DPSS.CO.PD.No.629",
        title="Harmonisation of Turn Around Time (TAT) and Customer Compensation for Failed Transactions",
        section_ref="Section 2(b) — IMPS & Card-to-Card Failed Transfer Reversal",
        content=(
            "For Immediate Payment Service (IMPS) transactions where the remitter account is debited "
            "and the beneficiary account is not credited, the harmonization turnaround time is T+1 days "
            "with a mandatory financial compensation of INR 100 per day beyond T+1."
        ),
        keywords=("imps", "failed", "debited", "beneficiary", "t+1", "100", "compensation"),
    ),
    RegulatoryDocumentChunk(
        circular_id="RBI/2017-18/15-DBR.No.Leg.BC.78",
        title="Customer Protection — Limiting Liability of Customers in Unauthorised Electronic Banking Transactions",
        section_ref="Paragraph 6 — Zero Liability of a Customer",
        content=(
            "A customer shall have Zero Liability (INR 0) in an unauthorized electronic banking transaction "
            "if the deficiency lies in the banking system, or in third-party breaches where neither the bank "
            "nor the customer is at fault and the customer notifies the bank within 3 working days (72 hours) "
            "of receiving the transaction communication. The bank must credit (shadow reversal) the amount "
            "involved in the unauthorized transaction within 10 working days of notification."
        ),
        keywords=(
            "unauthorized", "fraud", "zero liability", "3 working days", "72 hours",
            "10 working days", "shadow reversal", "third-party breach", "phishing", "hacked"
        ),
    ),
    RegulatoryDocumentChunk(
        circular_id="RBI/2017-18/15-DBR.No.Leg.BC.78",
        title="Customer Protection — Limiting Liability of Customers in Unauthorised Electronic Banking Transactions",
        section_ref="Paragraph 7 & Table 1 — Limited Liability of a Customer (4 to 7 Working Days Delay)",
        content=(
            "Where an unauthorized electronic banking transaction occurs due to a third-party breach and "
            "the customer notifies the bank within 4 to 7 working days (73 to 168 hours), per-transaction "
            "customer liability is capped at a maximum of INR 10,000 for basic savings/regular accounts "
            "and INR 25,000 for high-tier accounts and credit cards. Shadow credit must still be completed "
            "within 10 working days."
        ),
        keywords=(
            "unauthorized", "fraud", "limited liability", "4 to 7", "10000", "25000",
            "delay", "late report", "10 working days"
        ),
    ),
    RegulatoryDocumentChunk(
        circular_id="RBI/2017-18/15-DBR.No.Leg.BC.78",
        title="Customer Protection — Limiting Liability of Customers in Unauthorised Electronic Banking Transactions",
        section_ref="Paragraph 7(ii) — Customer Negligence (Sharing OTP/PIN) & Beyond 7 Days Delay",
        content=(
            "In cases where the loss is due to negligence by a customer, such as where they have shared "
            "the payment credentials (UPI PIN or OTP), the customer will bear the entire loss until they "
            "report the unauthorized transaction to the bank. Any loss occurring after the reporting of "
            "the unauthorized transaction shall be borne by the bank. If reported beyond 7 working days "
            "(>168 hours), liability is determined per the bank Board-approved policy and requires "
            "human compliance escalation."
        ),
        keywords=("negligence", "shared otp", "shared pin", "beyond 7 days", "board policy", "escalation"),
    ),
]


class HybridComplianceRetriever:
    """
    Production-pattern Hybrid Retriever combining:
      1. Sparse lexical BM25-style keyword scoring
      2. Dense hashed n-gram vector cosine similarity
      3. Cross-Encoder domain feature reranking with minimum relevance threshold
    """

    def __init__(self, corpus: Sequence[RegulatoryDocumentChunk] = RBI_REGULATORY_CORPUS, min_threshold: float = 0.22) -> None:
        """Copy corpus chunks and prepare their vectors for retrieval.

        ``min_threshold`` is the inclusive cutoff for rounded relevance scores.
        """
        self.corpus = list(corpus)
        self.min_threshold = min_threshold
        self._doc_vectors = [self._embed_text(f"{c.title} {c.section_ref} {c.content}") for c in self.corpus]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Return lowercase runs of ASCII letters, digits, and plus signs."""
        return re.findall(r"[a-z0-9\+]+", text.lower())

    @classmethod
    def _embed_text(cls, text: str, dim: int = 128) -> List[float]:
        """Return a normalized hashed token/bigram vector with ``dim`` components.

        A zero vector stays zero. A token-bearing input with ``dim=0`` raises
        ZeroDivisionError; negative dimensions raise IndexError for such input.
        """
        vec = [0.0] * dim
        tokens = cls._tokenize(text)
        features = list(tokens)
        for i in range(len(tokens) - 1):
            features.append(f"{tokens[i]}_{tokens[i+1]}")
        for feat in features:
            digest = hashlib.md5(feat.encode("utf-8")).digest()
            idx = digest[0] % dim
            sign = 1.0 if (digest[1] % 2 == 0) else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        """Return the dot product clipped at zero.

        This equals clipped cosine similarity for unit vectors. Ignore any
        unpaired trailing components.
        """
        return max(0.0, sum(x * y for x, y in zip(a, b)))

    def retrieve_and_rerank(
        self, query: str, payment_rail: PaymentRail, delay_hours: int, top_k: int = 2
    ) -> List[RBIComplianceCitation]:
        """Return citations in descending relevance order, or [] if none qualify.

        Match the narrative using lexical and vector scores, with payment-rail
        and reporting-delay boosts. ``delay_hours`` is elapsed time from alert
        to report; fraud matches use <=72, 73-168, and >168 hour bands.
        Scores are capped at 0.99, rounded to three decimals, and filtered by
        the inclusive minimum threshold. Ties retain corpus order.

        ``top_k`` uses slice semantics: zero returns no citations and a negative
        value omits that many qualifying citations from the end. Citation model
        validation errors propagate.
        """
        query_lower = query.lower()
        query_tokens = set(self._tokenize(query))
        q_vec = self._embed_text(query)

        scored: List[Tuple[float, RegulatoryDocumentChunk]] = []
        for chunk, d_vec in zip(self.corpus, self._doc_vectors):
            doc_tokens = set(self._tokenize(chunk.content))
            lexical_hits = len(query_tokens & doc_tokens)
            keyword_hits = sum(1 for kw in chunk.keywords if kw in query_lower)
            sparse_score = min(1.0, (lexical_hits * 0.06) + (keyword_hits * 0.18))

            dense_score = self._cosine(q_vec, d_vec)

            rerank_boost = 0.0
            if payment_rail.value.lower() in chunk.keywords:
                rerank_boost += 0.18
            if ("unauthorized" in query_lower or "fraud" in query_lower or "hacked" in query_lower):
                if delay_hours <= 72 and "72 hours" in chunk.keywords:
                    rerank_boost += 0.25
                elif 72 < delay_hours <= 168 and "4 to 7" in chunk.keywords:
                    rerank_boost += 0.25
                elif delay_hours > 168 and "beyond 7 days" in chunk.keywords:
                    rerank_boost += 0.25
            if ("shared otp" in query_lower or "shared pin" in query_lower) and "negligence" in chunk.keywords:
                rerank_boost += 0.30

            final_score = round(min(0.99, (0.45 * sparse_score) + (0.30 * dense_score) + rerank_boost), 3)
            if final_score >= self.min_threshold:
                scored.append((final_score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RBIComplianceCitation(
                circular_id=chunk.circular_id,
                title=chunk.title,
                section_ref=chunk.section_ref,
                relevance_score=score,
            )
            for score, chunk in scored[:top_k]
        ]