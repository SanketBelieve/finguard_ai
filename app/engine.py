"""
engine.py — Day 3-4 Live LLM Structured Synthesis with Deterministic Fallback & Guardrails
==========================================================================================
Wires the `HYBRID_RAG_AGENT` branch in `FinGuardTriageEngine.process_dispute()`
to call OpenAI (`gpt-4o-mini`), Anthropic (`claude-3-5-haiku-latest`), or a
local Ollama model using `.with_structured_output(LLMSynthesisDraft)`, with prompt-injection
defense, duplicate debit auto-reversal, commercial dispute handling, and automatic
deterministic fallback when API timeouts or rate limits occur.
"""

from __future__ import annotations

import os
import time
from typing import Any, List, Optional, Protocol, Sequence, Tuple

from database import ComplianceSQLStore
from retriever import HybridComplianceRetriever
from router import DeterministicFinTechRouter
from schemas import (
    DisputeCategory,
    DisputeIntakeRequest,
    EscalationAction,
    LLMSynthesisDraft,
    PaymentRail,
    RBIComplianceCitation,
    TriageVerdict,
)


class StructuredRunnable(Protocol):
    """Protocol matching LangChain's `.with_structured_output(...)` runnable."""

    def invoke(self, prompt: str) -> LLMSynthesisDraft | Dict[str, Any]:
        ...


class StructuredLLMSynthesizer:
    """
    Production LLM synthesis wrapper supporting:
      1. Custom injected `llm` (supporting `.with_structured_output(LLMSynthesisDraft)`)
      2. OpenAI (`gpt-4o-mini`) via `langchain_openai.ChatOpenAI`
      3. Anthropic (`claude-3-5-haiku-latest`) via `langchain_anthropic.ChatAnthropic`
      4. Local Ollama (`llama3.2`) via `langchain_ollama.ChatOllama`
    """

    SYSTEM_PROMPT_TEMPLATE = (
        "You are an RBI Banking Compliance Officer AI. Analyze the customer dispute "
        "strictly using the retrieved RBI Master Direction chunks below.\n"
        "Rules:\n"
        "1. Do NOT speculate beyond retrieved context.\n"
        "2. Cite the exact RBI Circular_ID, Source PDF filename, and Page Number.\n"
        "3. Apply RBI statutory caps: Zero Liability (INR 0) if unauthorized 3rd-party "
        "breach reported <=72h; Limited Liability (max INR 10,000 savings / INR 25,000 "
        "credit card) if reported 73-168h; Human Compliance Escalation if customer shared "
        "OTP/PIN or reported >168h; T+1 auto-reversal + INR 100/day penalty for failed UPI/IMPS/NEFT/RTGS debits "
        "and duplicate charges.\n\n"
        "Retrieved RBI Context:\n{context}\n\n"
        "Dispute Details:\n"
        "- Transaction ID: {transaction_id}\n"
        "- Payment Rail: {payment_rail}\n"
        "- Amount (INR): {amount_inr}\n"
        "- Reporting Delay (Hours): {reporting_delay_hours}\n"
        "- Customer Narrative: {customer_narrative}\n"
    )

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        provider_name: Optional[str] = None,
        timeout_seconds: float = 2.5,
        max_retries: int = 2,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.structured_runnable: Optional[Any] = None
        self.provider_name: str = provider_name or "unconfigured"

        if llm_client is not None:
            self.provider_name = provider_name or getattr(llm_client, "model_name", "injected_llm")
            if hasattr(llm_client, "with_structured_output"):
                self.structured_runnable = llm_client.with_structured_output(LLMSynthesisDraft)
            else:
                self.structured_runnable = llm_client
        else:
            self._auto_init_from_env()

    def _auto_init_from_env(self) -> None:
        if os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI  # type: ignore

                model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                llm = ChatOpenAI(model=model, temperature=0.0, timeout=self.timeout_seconds)
                self.structured_runnable = llm.with_structured_output(LLMSynthesisDraft)
                self.provider_name = f"openai/{model}"
                return
            except ImportError:
                pass

        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                from langchain_anthropic import ChatAnthropic  # type: ignore

                model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
                llm = ChatAnthropic(model=model, temperature=0.0, timeout=self.timeout_seconds)
                self.structured_runnable = llm.with_structured_output(LLMSynthesisDraft)
                self.provider_name = f"anthropic/{model}"
                return
            except ImportError:
                pass

        if os.getenv("OLLAMA_MODEL"):
            try:
                from langchain_ollama import ChatOllama  # type: ignore

                model = os.getenv("OLLAMA_MODEL", "llama3.2")
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                llm = ChatOllama(model=model, base_url=base_url, temperature=0.0)
                self.structured_runnable = llm.with_structured_output(LLMSynthesisDraft)
                self.provider_name = f"ollama/{model}"
                return
            except ImportError:
                pass

    @property
    def is_configured(self) -> bool:
        return self.structured_runnable is not None

    def build_prompt(self, req: DisputeIntakeRequest, citations: Sequence[RBIComplianceCitation]) -> str:
        context_lines = [
            f"[{i+1}] Circular: {c.circular_id} | Section: {c.section_ref} | "
            f"PDF: {c.source_pdf} (Page {c.page_number}) | Relevance: {c.relevance_score}"
            for i, c in enumerate(citations)
        ]
        return self.SYSTEM_PROMPT_TEMPLATE.format(
            context="\n".join(context_lines),
            transaction_id=req.transaction_id,
            payment_rail=req.payment_rail.value,
            amount_inr=req.amount_inr,
            reporting_delay_hours=req.reporting_delay_hours,
            customer_narrative=req.customer_narrative,
        )

    def synthesize(
        self, req: DisputeIntakeRequest, citations: Sequence[RBIComplianceCitation]
    ) -> Tuple[LLMSynthesisDraft, str]:
        if not self.structured_runnable:
            raise RuntimeError("No live LLM provider configured; triggering deterministic policy fallback.")

        prompt = self.build_prompt(req, citations)
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                raw_result = self.structured_runnable.invoke(prompt)
                if isinstance(raw_result, LLMSynthesisDraft):
                    draft = raw_result
                elif isinstance(raw_result, dict):
                    draft = LLMSynthesisDraft(**raw_result)
                else:
                    raise ValueError(f"Unexpected structured output type: {type(raw_result)}")
                return draft, self.provider_name
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(0.01 * (2 ** (attempt - 1)))

        raise TimeoutError(f"LLM synthesis failed after {self.max_retries} attempts: {last_exc}") from last_exc


class FinGuardTriageEngine:
    """
    Coordinates:
      Step 1 -> Deterministic Intent & Prompt Injection Security Router
      Step 2 -> Fast Route for duplicate charges & merchant non-delivery
      Step 3 -> Hybrid RAG + Cross-Encoder Reranking over ingested RBI PDF circulars
      Step 4 -> Live LLM Synthesis via `.with_structured_output(LLMSynthesisDraft)`
      Step 5 -> Zero-Downtime Deterministic Fallback on LLM timeout / rate limit
      Step 6 -> Parameterized SQL audit logging
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        pdf_dir: str = "rbi_circulars",
        vector_backend: str = "sqlite_vec",
        llm_client: Optional[Any] = None,
        llm_provider_name: Optional[str] = None,
    ) -> None:
        self.retriever = HybridComplianceRetriever(
            pdf_dir=pdf_dir,
            vector_backend=vector_backend,
        )
        self.sql_store = ComplianceSQLStore(db_path=db_path)
        self.llm_synthesizer = StructuredLLMSynthesizer(
            llm_client=llm_client,
            provider_name=llm_provider_name,
        )

    @staticmethod
    def compute_deterministic_policy_fallback(
        req: DisputeIntakeRequest, citations: List[RBIComplianceCitation]
    ) -> LLMSynthesisDraft:
        narrative_lower = req.customer_narrative.lower()
        top_cit = citations[0]
        pdf_loc = f"{top_cit.source_pdf} p.{top_cit.page_number}"

        customer_shared_credentials = any(
            phrase in narrative_lower for phrase in ("shared otp", "gave otp", "shared pin", "entered upi pin", "shared cvv")
        )
        is_unauthorized = any(
            kw in narrative_lower
            for kw in ("unauthorized", "fraud", "hacked", "phishing", "unknown", "without my knowledge", "stolen", "breach")
        )

        if is_unauthorized or customer_shared_credentials:
            category = DisputeCategory.UNAUTHORIZED_FRAUD
            if customer_shared_credentials or req.reporting_delay_hours > 168:
                action = EscalationAction.HUMAN_COMPLIANCE_ESCALATION
                max_liability = req.amount_inr if customer_shared_credentials else min(
                    req.amount_inr, 25_000.0 if req.payment_rail == PaymentRail.CREDIT_CARD else 10_000.0
                )
                rationale = (
                    f"Escalated under {top_cit.circular_id} ({top_cit.section_ref}, {pdf_loc}): "
                    f"{'Customer credential sharing detected' if customer_shared_credentials else 'Reporting delay >7 working days (>168h)'}. "
                    f"Requires human compliance verification and post-reporting loss freeze."
                )
            elif req.reporting_delay_hours <= 72:
                action = EscalationAction.ZERO_LIABILITY_SHADOW_CREDIT
                max_liability = 0.0
                rationale = (
                    f"Zero Customer Liability qualified under {top_cit.circular_id} ({top_cit.section_ref}, {pdf_loc}): "
                    f"reported within {req.reporting_delay_hours}h (<=72h window). Mandatory shadow credit within 10 working days."
                )
            else:
                action = EscalationAction.LIMITED_LIABILITY_REVIEW
                cap = 25_000.0 if req.payment_rail == PaymentRail.CREDIT_CARD else 10_000.0
                max_liability = min(req.amount_inr, cap)
                rationale = (
                    f"Limited Customer Liability applied under {top_cit.circular_id} ({top_cit.section_ref}, {pdf_loc}): "
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
                f"Classified under {top_cit.circular_id} ({top_cit.section_ref}, {pdf_loc}) via Hybrid RAG: "
                f"T+1 reversal rule applies with INR 100/day penalty for delayed settlement."
            )

        return LLMSynthesisDraft(
            dispute_category=category,
            escalation_action=action,
            max_customer_liability_inr=max_liability,
            provisional_credit_due_days=credit_days,
            compensation_penalty_per_day_inr=penalty_per_day,
            rationale=rationale,
        )

    def process_dispute(self, req: DisputeIntakeRequest) -> TriageVerdict:
        """Return a triage verdict and commit its audit record before returning.

        Reject out-of-scope matches without retrieval; missing citations instead
        produce a human-escalation fallback. After fast routing, fraud matches
        use reporting-delay thresholds of <=72, 73-168, and >168 elapsed hours;
        credential-sharing matches select human escalation. Remaining cases use
        failed-debit reversal terms. The verdict's latency excludes audit logging.

        Retrieval, Pydantic validation, and SQLite audit errors propagate rather
        than being converted to fallback verdicts.
        """
        t0 = time.perf_counter()

        # Step 1: Sub-15ms Intent & Prompt-Injection Security Router
        fast_cat = DeterministicFinTechRouter.classify_fast(req)
        if fast_cat == DisputeCategory.OUT_OF_SCOPE_QUERY:
            latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            is_injection = DeterministicFinTechRouter.is_prompt_injection(req.customer_narrative)
            rationale = (
                "Security Guardrail: Adversarial prompt injection or instruction override attempt intercepted and blocked. "
                "FinGuard AI enforces strict RBI statutory compliance."
                if is_injection
                else "Out-of-scope query blocked by deterministic domain guardrail. "
                "FinGuard AI only processes regulated Indian payment disputes and RBI compliance rules."
            )
            verdict = TriageVerdict(
                transaction_id=req.transaction_id,
                dispute_category=DisputeCategory.OUT_OF_SCOPE_QUERY,
                escalation_action=EscalationAction.REJECT_OUT_OF_SCOPE,
                max_customer_liability_inr=0.0,
                provisional_credit_due_days=0,
                compensation_penalty_per_day_inr=0.0,
                confidence_score=0.99,
                routed_via="GUARDRAIL_FALLBACK",
                llm_provider_used="security_guardrail" if is_injection else "deterministic_guardrail",
                fallback_triggered=False,
                citations=[],
                rationale=rationale,
                latency_ms=latency_ms,
                estimated_cost_inr=0.0,
            )
            self.sql_store.log_verdict(req, verdict)
            return verdict

        # Step 2: Retrieve & rerank RBI compliance circulars from PDF vector store
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
                llm_provider_used="deterministic_guardrail",
                fallback_triggered=False,
                citations=[],
                rationale="Insufficient regulatory context retrieved; escalated to human compliance officer.",
                latency_ms=latency_ms,
                estimated_cost_inr=0.0,
            )
            self.sql_store.log_verdict(req, verdict)
            return verdict

        top_cit = citations[0]
        pdf_loc = f"{top_cit.source_pdf} p.{top_cit.page_number}"

        # Step 3: Fast-routed Duplicate Debits & Failed Debits
        if fast_cat == DisputeCategory.DUPLICATE_CHARGE:
            latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            verdict = TriageVerdict(
                transaction_id=req.transaction_id,
                dispute_category=DisputeCategory.DUPLICATE_CHARGE,
                escalation_action=EscalationAction.AUTO_RESOLVE_T_PLUS_1,
                max_customer_liability_inr=0.0,
                provisional_credit_due_days=1,
                compensation_penalty_per_day_inr=100.0,
                confidence_score=0.98,
                routed_via="DETERMINISTIC_ROUTER",
                llm_provider_used="deterministic_router_bypass",
                fallback_triggered=False,
                citations=citations,
                rationale=(
                    f"Deterministically routed duplicate debit dispute under {top_cit.circular_id} "
                    f"({top_cit.section_ref}, {pdf_loc}): duplicate charge auto-reversal mandatory within T+1 days "
                    f"with INR 100/day delay compensation."
                ),
                latency_ms=latency_ms,
                estimated_cost_inr=0.001,
            )
            self.sql_store.log_verdict(req, verdict)
            return verdict

        if fast_cat == DisputeCategory.MERCHANT_NON_DELIVERY:
            latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            verdict = TriageVerdict(
                transaction_id=req.transaction_id,
                dispute_category=DisputeCategory.MERCHANT_NON_DELIVERY,
                escalation_action=EscalationAction.LIMITED_LIABILITY_REVIEW,
                max_customer_liability_inr=0.0,
                provisional_credit_due_days=30,
                compensation_penalty_per_day_inr=0.0,
                confidence_score=0.95,
                routed_via="DETERMINISTIC_ROUTER",
                llm_provider_used="deterministic_router_bypass",
                fallback_triggered=False,
                citations=citations,
                rationale=(
                    f"Commercial merchant dispute processed under {top_cit.circular_id} "
                    f"({top_cit.section_ref}, {pdf_loc}): customer liability paused, chargeback dispute "
                    f"initiated with 30-day merchant rebuttal period."
                ),
                latency_ms=latency_ms,
                estimated_cost_inr=0.001,
            )
            self.sql_store.log_verdict(req, verdict)
            return verdict

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
                llm_provider_used="deterministic_router_bypass",
                fallback_triggered=False,
                citations=citations,
                rationale=(
                    f"Deterministically routed {req.payment_rail.value} failed-debit dispute under "
                    f"{top_cit.circular_id} ({top_cit.section_ref}, {pdf_loc}): mandatory T+1 auto-reversal "
                    f"with INR 100/day delay compensation beyond T+1."
                ),
                latency_ms=latency_ms,
                estimated_cost_inr=0.001,
            )
            self.sql_store.log_verdict(req, verdict)
            return verdict

        # Step 4: Complex Disputes & Unauthorized Fraud — Live LLM Synthesis with Policy Fallback
        fallback_triggered = False
        if self.llm_synthesizer.is_configured:
            try:
                draft, provider_used = self.llm_synthesizer.synthesize(req, citations)
                est_cost_inr = 0.028
            except Exception:
                draft = self.compute_deterministic_policy_fallback(req, citations)
                provider_used = "deterministic_policy_fallback"
                fallback_triggered = True
                est_cost_inr = 0.002
        else:
            draft = self.compute_deterministic_policy_fallback(req, citations)
            provider_used = "deterministic_policy_fallback"
            fallback_triggered = False
            est_cost_inr = 0.028

        latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        verdict = TriageVerdict(
            transaction_id=req.transaction_id,
            dispute_category=draft.dispute_category,
            escalation_action=draft.escalation_action,
            max_customer_liability_inr=draft.max_customer_liability_inr,
            provisional_credit_due_days=draft.provisional_credit_due_days,
            compensation_penalty_per_day_inr=draft.compensation_penalty_per_day_inr,
            confidence_score=top_cit.relevance_score,
            routed_via="HYBRID_RAG_AGENT",
            llm_provider_used=provider_used,
            fallback_triggered=fallback_triggered,
            citations=citations,
            rationale=draft.rationale,
            latency_ms=latency_ms,
            estimated_cost_inr=est_cost_inr,
        )
        self.sql_store.log_verdict(req, verdict)
        return verdict