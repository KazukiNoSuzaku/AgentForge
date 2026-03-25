"""
AgentForge Benchmark Runner

Executes research queries through the AgentForge pipeline and captures
detailed performance metrics: wall time, token usage, source counts,
quality scores, and estimated cost.

Usage:
    python benchmarks/run_benchmark.py "Your research question"
    python benchmarks/run_benchmark.py "Your question" --compare
    python benchmarks/run_benchmark.py --suite
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from src.config import get_config
from src.graph import build_graph
from src.state import create_initial_state

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"

BENCHMARK_SUITE = [
    "Compare AI regulatory frameworks in the EU, US, and China",
    "What are the key technical approaches to LLM alignment and safety?",
    "Explain the causes and global impact of the 2023 semiconductor shortage",
]

# Anthropic pricing (per 1M tokens) — Claude Sonnet
COST_INPUT_PER_1M = 3.00
COST_OUTPUT_PER_1M = 15.00


# ---------------------------------------------------------------------------
# Token Counting
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English text."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Metrics Extraction
# ---------------------------------------------------------------------------


def extract_metrics(final_state: dict, wall_time: float) -> Dict[str, Any]:
    """
    Extract structured metrics from the pipeline's final state.

    Args:
        final_state: The terminal state dict from graph.ainvoke().
        wall_time: Elapsed seconds for the full pipeline run.

    Returns:
        Dict of pipeline metrics for the benchmark report.
    """
    findings = final_state.get("research_findings", [])
    analysis = final_state.get("analysis")
    feedback = final_state.get("critic_feedback")
    report = final_state.get("final_report", "")
    sub_questions = final_state.get("sub_questions", [])

    # Source breakdown
    sources_by_type: Dict[str, int] = {}
    unique_urls: set = set()
    for f in findings:
        source = f.source_type if hasattr(f, "source_type") else "unknown"
        source_key = source.value if hasattr(source, "value") else str(source)
        sources_by_type[source_key] = sources_by_type.get(source_key, 0) + 1
        if hasattr(f, "url"):
            unique_urls.add(f.url)

    # Claims
    claims = analysis.claims if analysis else []
    high_conf = sum(1 for c in claims if c.confidence >= 0.8)

    # Token estimation from report + findings
    report_tokens = estimate_tokens(report)
    findings_text = " ".join(f.content for f in findings if hasattr(f, "content"))
    input_tokens_est = estimate_tokens(findings_text) + report_tokens
    output_tokens_est = report_tokens + estimate_tokens(" ".join(c.statement for c in claims))

    # Cost
    input_cost = (input_tokens_est / 1_000_000) * COST_INPUT_PER_1M
    output_cost = (output_tokens_est / 1_000_000) * COST_OUTPUT_PER_1M

    # Citation count (inline [N] references)
    import re

    citation_numbers = set(int(n) for n in re.findall(r"\[(\d+)\]", report))

    return {
        "wall_time_seconds": round(wall_time, 1),
        "sub_questions_generated": len(sub_questions),
        "sources_retrieved": len(findings),
        "unique_sources": len(unique_urls),
        "sources_by_type": sources_by_type,
        "claims_extracted": len(claims),
        "high_confidence_claims": high_conf,
        "patterns_identified": len(analysis.patterns) if analysis else 0,
        "contradictions_found": len(analysis.contradictions) if analysis else 0,
        "information_gaps": len(analysis.gaps) if analysis else 0,
        "revision_count": final_state.get("revision_count", 0),
        "critic_quality_score": feedback.quality_score if feedback else None,
        "report_word_count": len(report.split()),
        "report_char_count": len(report),
        "citations_in_report": len(citation_numbers),
        "token_estimate": {
            "input": input_tokens_est,
            "output": output_tokens_est,
            "total": input_tokens_est + output_tokens_est,
        },
        "cost_estimate_usd": {
            "input_cost": round(input_cost, 4),
            "output_cost": round(output_cost, 4),
            "total": round(input_cost + output_cost, 4),
        },
    }


def extract_claims(final_state: dict) -> List[Dict[str, Any]]:
    """Extract claims with metadata for downstream evaluation."""
    analysis = final_state.get("analysis")
    if not analysis:
        return []

    return [
        {
            "id": claim.id,
            "statement": claim.statement,
            "confidence": claim.confidence,
            "supporting_sources": claim.supporting_sources,
            "sub_question_id": claim.sub_question_id,
        }
        for claim in analysis.claims
    ]


# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------


async def run_single_benchmark(query: str) -> Dict[str, Any]:
    """
    Run a single query through the AgentForge pipeline and capture metrics.

    Args:
        query: The research question to benchmark.

    Returns:
        Complete benchmark result dict.
    """
    config = get_config()
    graph = build_graph()
    initial_state = create_initial_state(query)

    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK: {query}")
    print(f"{'=' * 70}")
    print(f"  Model: {config.anthropic_model}")
    print(f"  Max revisions: {config.max_revision_loops}")
    print(f"  Quality threshold: {config.quality_threshold}")
    print(f"{'=' * 70}\n")

    start = time.monotonic()

    # Stream to show progress
    final_state = {}
    for update in graph.stream(initial_state, config={"recursion_limit": 20}):
        for node_name, node_output in update.items():
            status = node_output.get("agent_statuses", {}).get(node_name, "?")
            icon = "+" if status == "done" else "!"
            print(f"  [{icon}] {node_name:<12} {status}")
            final_state.update(node_output)

    wall_time = time.monotonic() - start

    metrics = extract_metrics(final_state, wall_time)
    claims = extract_claims(final_state)

    result = {
        "benchmark_id": f"bench_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}",
        "query": query,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "agentforge_version": "1.0.0",
        "model": config.anthropic_model,
        "config": {
            "max_sub_questions": config.max_sub_questions,
            "max_revision_loops": config.max_revision_loops,
            "quality_threshold": config.quality_threshold,
            "max_search_results": config.max_search_results,
            "temperature": config.llm_temperature,
        },
        "pipeline_metrics": metrics,
        "claims": claims,
        "final_report_preview": final_state.get("final_report", "")[:500],
        "errors": final_state.get("errors", []),
    }

    # Print summary
    print(f"\n{'─' * 50}")
    print(f"  Wall time:    {metrics['wall_time_seconds']}s")
    print(f"  Sources:      {metrics['sources_retrieved']} ({metrics['sources_by_type']})")
    print(
        f"  Claims:       {metrics['claims_extracted']} ({metrics['high_confidence_claims']} high-conf)"
    )
    print(f"  Quality:      {metrics['critic_quality_score']}")
    print(f"  Revisions:    {metrics['revision_count']}")
    print(f"  Report:       {metrics['report_word_count']} words")
    print(f"  Est. cost:    ${metrics['cost_estimate_usd']['total']:.3f}")
    print(f"{'─' * 50}\n")

    return result


def save_result(result: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
    """Save benchmark result to a JSON file."""
    if output_path is None:
        slug = result["query"][:50].lower().replace(" ", "_").replace(",", "")
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        filename = f"{slug}_{result['benchmark_id']}.json"
        output_path = RESULTS_DIR / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"  Results saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Comparison Mode
# ---------------------------------------------------------------------------


def print_comparison(result: Dict[str, Any]) -> None:
    """Print a comparison table against known baselines."""
    m = result["pipeline_metrics"]

    baselines = {
        "Perplexity Pro": {"sources": "8–15", "accuracy": "~79", "time": "15–30s", "cost": "$0.67"},
        "ChatGPT (4o)": {"sources": "0", "accuracy": "~71", "time": "10–20s", "cost": "$0.67"},
        "Manual Research": {
            "sources": "10–30+",
            "accuracy": "~89",
            "time": "2–6hr",
            "cost": "$50–200",
        },
    }

    print(f"\n{'=' * 70}")
    print("  COMPARISON vs BASELINES")
    print(f"{'=' * 70}")
    print(f"  {'Tool':<20} {'Sources':<12} {'Accuracy':<12} {'Time':<12} {'Cost':<10}")
    print(f"  {'─' * 66}")
    print(
        f"  {'AgentForge':<20} {m['sources_retrieved']:<12} "
        f"{'(evaluate)':<12} {m['wall_time_seconds']:.0f}s{'':<8} "
        f"${m['cost_estimate_usd']['total']:.2f}"
    )
    for name, b in baselines.items():
        print(
            f"  {name:<20} {b['sources']:<12} {b['accuracy']:<12} {b['time']:<12} {b['cost']:<10}"
        )
    print(f"{'=' * 70}")
    print("  Run `python benchmarks/evaluate.py <result.json>` for factual accuracy scoring.\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgentForge Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Research question to benchmark",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run all 3 benchmark queries",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print comparison table against baselines",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for the results JSON",
    )

    args = parser.parse_args()

    if args.suite:
        queries = BENCHMARK_SUITE
    elif args.query:
        queries = [args.query]
    else:
        parser.error("Provide a query or use --suite to run all benchmarks")

    for query in queries:
        result = asyncio.run(run_single_benchmark(query))
        save_result(result, args.output)

        if args.compare:
            print_comparison(result)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
