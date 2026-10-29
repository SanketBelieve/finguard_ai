"""
ci_eval_gate.py — Day 5-6 GitHub Actions CI/CD Quality Gate & PR Scorecard Runner
=================================================================================
Enforces automated regression thresholds on every `git push` and `pull_request`
for FinGuard AI:
  1. Executes all 9 unit, PDF (`pdfplumber`), LLM structured output, timeout
     fallback, prompt-injection security, sliding-window rate limiter, Docker
     manifest, and SQL guardrail tests.
  2. Runs the 36-case Golden Dataset Evaluation Suite and checks strict
     production thresholds:
       - Verdict Accuracy          >= 98.0% (Baseline naive RAG: 76.0%)
       - PDF Citation Faithfulness >= 98.0% (Baseline naive RAG: 72.0%)
       - Prompt Injection Blocking == 100.0%
       - Hallucination Rate        <= 2.0%  (Baseline naive RAG: 23.0%)
       - p95 Latency               <= 50.0 ms
  3. Generates a Markdown Pull Request Scorecard (`eval_report.md` and
     `$GITHUB_STEP_SUMMARY`) plus `.github/workflows/ci-eval.yml`,
     `requirements.txt`, `Dockerfile`, and `docker-compose.yml`.

Usage:
    python3 ci_eval_gate.py
    python3 ci_eval_gate.py --generate-workflow
    python3 ci_eval_gate.py --min-accuracy 100 --max-hallucination 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
_here = Path(__file__).resolve().parent
_app_dir = _here if _here.name == "app" else _here / "app"
if _app_dir.is_dir() and str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from main import generate_container_manifests
from test_evals import (
    run_evaluation_suite,
    test_adversarial_prompt_injection_is_blocked_without_leak,
    test_day3_llm_timeout_triggers_deterministic_fallback,
    test_day3_structured_llm_synthesis_happy_path,
    test_day6_docker_and_compose_manifests_valid,
    test_day6_sliding_window_rate_limiter_enforces_quota,
    test_eval_suite_passes_quality_gates,
    test_pdfplumber_ingests_and_indexes_real_pdfs,
    test_pydantic_coerces_formatted_inr_strings,
    test_sql_guardrail_blocks_destructive_queries,
)


GITHUB_ACTIONS_WORKFLOW_YAML = """name: FinGuard AI — CI/CD RAG & Security Quality Gate

on:
  push:
    branches: [ "main", "develop" ]
  pull_request:
    branches: [ "main", "develop" ]
  workflow_dispatch:

jobs:
  rag-and-security-quality-gate:
    name: Run Unit Tests, SQL Guardrails & 36-Case Golden Eval Gate
    runs-on: ubuntu-latest
    timeout-minutes: 5

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Pytest Unit, PDF, LLM Fallback, Rate-Limit & Security Suite
        run: |
          PYTHONPATH=. pytest test_evals.py -v

      - name: Enforce 36-Case Golden Dataset Quality Gate & Publish PR Summary
        run: |
          PYTHONPATH=. python3 ci_eval_gate.py \\
            --min-accuracy 98.0 \\
            --min-faithfulness 98.0 \\
            --min-injection-block 100.0 \\
            --max-hallucination 2.0 \\
            --max-p95-ms 50.0 \\
            --export-markdown eval_report.md \\
            --export-json eval_report.json

      - name: Upload Evaluation Scorecard Artifact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: finguard-ai-eval-scorecard
          path: |
            eval_report.md
            eval_report.json
          retention-days: 30
"""

REQUIREMENTS_TXT = """fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=1.10.0,<3.0.0
pdfplumber>=0.11.0
reportlab>=4.0.0
numpy>=1.24.0
pytest>=8.0.0
# Optional production vector, rate-limiting & LLM integrations:
# slowapi>=0.1.9
# chromadb>=0.5.0
# psycopg2-binary>=2.9.9
# langchain-openai>=0.1.0
# langchain-anthropic>=0.1.0
# langchain-ollama>=0.1.0
# deepeval>=1.0.0
"""


def generate_ci_files(base_dir: str | Path = ".") -> Tuple[Path, Path]:
    """Writes `.github/workflows/ci-eval.yml`, `requirements.txt`, and Docker manifests."""
    root = Path(base_dir)
    workflow_dir = root / ".github" / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)

    workflow_path = workflow_dir / "ci-eval.yml"
    workflow_path.write_text(GITHUB_ACTIONS_WORKFLOW_YAML, encoding="utf-8")

    req_path = root / "requirements.txt"
    req_path.write_text(REQUIREMENTS_TXT, encoding="utf-8")

    generate_container_manifests(root)
    return workflow_path, req_path


def run_unit_and_security_tests() -> List[Tuple[str, bool, str]]:
    """Executes all 9 deterministic unit, PDF, LLM, rate-limiter, Docker, and security tests."""
    checks = [
        ("PDF Extraction & Vector Indexing (`pdfplumber`)", test_pdfplumber_ingests_and_indexes_real_pdfs),
        ("Structured LLM Output Happy Path (`.with_structured_output`)", test_day3_structured_llm_synthesis_happy_path),
        ("LLM Timeout -> Deterministic Policy Fallback", test_day3_llm_timeout_triggers_deterministic_fallback),
        ("Adversarial Prompt-Injection & Jailbreak Defense", test_adversarial_prompt_injection_is_blocked_without_leak),
        ("Sliding-Window API Rate Limiter (HTTP 429 Quota)", test_day6_sliding_window_rate_limiter_enforces_quota),
        ("Multi-Stage Dockerfile & Compose Security Manifests", test_day6_docker_and_compose_manifests_valid),
        ("Pydantic Currency Coercion & Schema Validation", test_pydantic_coerces_formatted_inr_strings),
        ("Read-Only SQL Sanitizer (`DROP TABLE` Blocked)", test_sql_guardrail_blocks_destructive_queries),
        ("36-Case Golden Dataset Quality Gate Assertions", test_eval_suite_passes_quality_gates),
    ]
    results: List[Tuple[str, bool, str]] = []
    for label, fn in checks:
        try:
            fn()
            results.append((label, True, "PASSED"))
        except Exception as exc:
            results.append((label, False, f"FAILED: {exc}"))
    return results


def evaluate_quality_gates(
    summary: Dict[str, Any],
    min_accuracy: float = 98.0,
    min_faithfulness: float = 98.0,
    min_injection_block: float = 100.0,
    max_hallucination: float = 2.0,
    max_p95_ms: float = 50.0,
) -> List[Dict[str, Any]]:
    gates = [
        {
            "metric": "Verdict Accuracy (%)",
            "baseline_naive_rag": "76.0%",
            "actual": f"{summary['verdict_accuracy_pct']}%",
            "threshold": f">= {min_accuracy}%",
            "passed": summary["verdict_accuracy_pct"] >= min_accuracy,
        },
        {
            "metric": "PDF + Page Citation Faithfulness (%)",
            "baseline_naive_rag": "72.0%",
            "actual": f"{summary['citation_faithfulness_pct']}%",
            "threshold": f">= {min_faithfulness}%",
            "passed": summary["citation_faithfulness_pct"] >= min_faithfulness,
        },
        {
            "metric": "Prompt-Injection Block Rate (%)",
            "baseline_naive_rag": "0.0% (Unprotected)",
            "actual": f"{summary['prompt_injection_block_rate_pct']}%",
            "threshold": f">= {min_injection_block}%",
            "passed": summary["prompt_injection_block_rate_pct"] >= min_injection_block,
        },
        {
            "metric": "Hallucination Rate (%)",
            "baseline_naive_rag": "23.0%",
            "actual": f"{summary['hallucination_rate_pct']}%",
            "threshold": f"<= {max_hallucination}%",
            "passed": summary["hallucination_rate_pct"] <= max_hallucination,
        },
        {
            "metric": "p95 Triage Latency (ms)",
            "baseline_naive_rag": "3,800.0 ms",
            "actual": f"{summary['p95_latency_ms']} ms",
            "threshold": f"<= {max_p95_ms} ms",
            "passed": summary["p95_latency_ms"] <= max_p95_ms,
        },
    ]
    return gates


def build_markdown_scorecard(
    unit_results: List[Tuple[str, bool, str]],
    gates: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> str:
    all_units_ok = all(ok for _, ok, _ in unit_results)
    all_gates_ok = all(g["passed"] for g in gates)
    overall_badge = "PASS — READY TO MERGE" if (all_units_ok and all_gates_ok) else "FAIL — MERGE BLOCKED"

    op = summary["sql_operational_metrics"]
    lines = [
        f"# FinGuard AI — CI/CD Evaluation & Security Scorecard ({overall_badge})",
        "",
        f"- **Timestamp (UTC)**: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`",
        f"- **Golden Dataset Size**: `{summary['total_cases']} cases` across `{summary['indexed_pdf_chunks']} RBI PDF chunks`",
        f"- **Estimated Cost / 1,000 Queries**: `INR {op.get('cost_per_1000_queries_inr', 0.0)}` "
        f"(Deterministic Router Bypass: `{op.get('deterministic_router_bypass_pct', 0.0)}%`)",
        "",
        "## 1. Production Quality Gates (Before vs. After Tuning)",
        "",
        "| Quality Gate Metric | Baseline (Naive RAG) | Current FinGuard AI | CI Threshold | Gate Status |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for g in gates:
        status_str = "PASS" if g["passed"] else "FAIL"
        lines.append(
            f"| **{g['metric']}** | {g['baseline_naive_rag']} | **{g['actual']}** | `{g['threshold']}` | **{status_str}** |"
        )

    lines.extend(
        [
            "",
            "## 2. Automated Unit, PDF, LLM Fallback, Rate-Limit & Docker Checks",
            "",
            "| Check Name | Status |",
            "| :--- | :--- |",
        ]
    )
    for label, ok, msg in unit_results:
        lines.append(f"| {label} | **{'PASS' if ok else msg}** |")

    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FinGuard AI CI/CD Quality Gate Runner")
    parser.add_argument("--generate-workflow", action="store_true", help="Generate CI & Docker files")
    parser.add_argument("--min-accuracy", type=float, default=98.0)
    parser.add_argument("--min-faithfulness", type=float, default=98.0)
    parser.add_argument("--min-injection-block", type=float, default=100.0)
    parser.add_argument("--max-hallucination", type=float, default=2.0)
    parser.add_argument("--max-p95-ms", type=float, default=50.0)
    parser.add_argument("--export-markdown", type=str, default="eval_report.md")
    parser.add_argument("--export-json", type=str, default="eval_report.json")
    args = parser.parse_args(argv)

    if args.generate_workflow:
        wf_path, req_path = generate_ci_files(".")
        print(f"[CI Generator] Wrote GitHub Actions workflow : {wf_path}")
        print(f"[CI Generator] Wrote Python requirements     : {req_path}")
        print("[CI Generator] Wrote Dockerfile, docker-compose.yml, .env.example, and .dockerignore")

    unit_results = run_unit_and_security_tests()
    summary = run_evaluation_suite()
    gates = evaluate_quality_gates(
        summary=summary,
        min_accuracy=args.min_accuracy,
        min_faithfulness=args.min_faithfulness,
        min_injection_block=args.min_injection_block,
        max_hallucination=args.max_hallucination,
        max_p95_ms=args.max_p95_ms,
    )

    md_report = build_markdown_scorecard(unit_results, gates, summary)

    Path(args.export_markdown).write_text(md_report, encoding="utf-8")
    github_summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if github_summary_path:
        with open(github_summary_path, "a", encoding="utf-8") as f:
            f.write(md_report)

    json_payload = {
        "unit_tests_passed": all(ok for _, ok, _ in unit_results),
        "quality_gates_passed": all(g["passed"] for g in gates),
        "quality_gates": gates,
        "summary": summary,
    }
    Path(args.export_json).write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    print("=" * 100)
    print("DAY 6 FINGUARD AI — CI/CD QUALITY GATE, RATE LIMITER & DOCKER VERIFICATION")
    print("=" * 100)
    for g in gates:
        badge = "PASS" if g["passed"] else "FAIL"
        print(
            f"[{badge}] {g['metric']:<38} | Naive Baseline: {g['baseline_naive_rag']:<18} "
            f"| Actual: {g['actual']:<10} | Gate: {g['threshold']}"
        )
    print("-" * 100)
    print(f"Markdown PR Scorecard exported to : {args.export_markdown}")
    print(f"JSON CI Artifact exported to      : {args.export_json}")
    print("=" * 100)

    if not json_payload["unit_tests_passed"] or not json_payload["quality_gates_passed"]:
        print("CI/CD QUALITY GATE FAILED: Blocking Pull Request Merge.", file=sys.stderr)
        return 1

    print("CI/CD QUALITY GATE PASSED: All 9 test suites & 5 regression gates green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())