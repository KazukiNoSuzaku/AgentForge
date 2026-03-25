"""
AgentForge Factual Accuracy Evaluator

Uses GPT-4 as an impartial judge (G-Eval framework) to score the factual
accuracy of claims in a benchmark result against Wikipedia ground truth.

Usage:
    python benchmarks/evaluate.py benchmarks/results/ai_regulation_benchmark.json
    python benchmarks/evaluate.py benchmarks/results/*.json --summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Ground Truth Retrieval (Wikipedia)
# ---------------------------------------------------------------------------


async def fetch_wikipedia_context(claim: str, num_results: int = 2) -> str:
    """
    Retrieve Wikipedia context relevant to a claim for ground truth comparison.

    Uses the Wikipedia MCP server to search and fetch article summaries.

    Args:
        claim: The factual claim to find ground truth for.
        num_results: Number of Wikipedia articles to fetch.

    Returns:
        Concatenated Wikipedia summaries as ground truth context.
    """
    import wikipedia

    # Extract key terms from the claim for search
    try:
        search_results = wikipedia.search(claim, results=num_results)
    except wikipedia.exceptions.WikipediaException:
        return "(No Wikipedia results found)"

    context_parts = []
    for title in search_results[:num_results]:
        try:
            page = wikipedia.page(title, auto_suggest=False)
            context_parts.append(f"[{page.title}]: {page.summary}")
        except (
            wikipedia.exceptions.DisambiguationError,
            wikipedia.exceptions.PageError,
        ):
            continue

    return "\n\n".join(context_parts) if context_parts else "(No Wikipedia context available)"


# ---------------------------------------------------------------------------
# GPT-4 Judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an impartial factual accuracy judge. You evaluate whether claims from a research report are factually correct by comparing them against provided ground truth from Wikipedia.

## Scoring Rubric (0-10):

- **10**: Fully supported by ground truth. Correct in all details and nuance.
- **8-9**: Substantively correct. Core claim is accurate; minor details may be simplified or slightly imprecise.
- **6-7**: Mostly correct. The main idea is right but some important details are wrong, outdated, or missing context.
- **4-5**: Partially correct. Contains a mix of accurate and inaccurate information.
- **2-3**: Mostly incorrect. The core claim is wrong or seriously misleading, though minor elements may be true.
- **0-1**: Fabricated or directly contradicts ground truth.

## Rules:
- Score based ONLY on the provided ground truth, not your own knowledge.
- If the ground truth doesn't cover the claim's topic, score 5 (insufficient evidence) and note this.
- Be strict about factual details (dates, names, specific provisions).
- Be lenient about reasonable simplifications of complex topics.

## Output Format:
Return ONLY valid JSON:
{
  "score": <int 0-10>,
  "reasoning": "<1-2 sentence explanation>",
  "verdict": "supported" | "partially_supported" | "unsupported" | "insufficient_evidence"
}"""


async def judge_claim(
    claim: str,
    ground_truth: str,
    openai_api_key: str,
    model: str = "gpt-4-turbo-2024-04-09",
) -> Dict[str, Any]:
    """
    Use GPT-4 to score a single claim against ground truth.

    Args:
        claim: The factual claim to evaluate.
        ground_truth: Wikipedia context as reference.
        openai_api_key: OpenAI API key for the judge model.
        model: GPT-4 model variant to use.

    Returns:
        Dict with score (0-10), reasoning, and verdict.
    """
    import openai

    client = openai.AsyncOpenAI(api_key=openai_api_key)

    user_prompt = (
        f"## Claim to Evaluate\n{claim}\n\n"
        f"## Ground Truth (Wikipedia)\n{ground_truth}\n\n"
        "Score this claim's factual accuracy (0-10) based on the ground truth above."
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=256,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        return {
            "score": int(result.get("score", 5)),
            "reasoning": result.get("reasoning", ""),
            "verdict": result.get("verdict", "unknown"),
        }
    except Exception as e:
        return {
            "score": 5,
            "reasoning": f"Judge error: {e}",
            "verdict": "error",
        }


# ---------------------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------------------


async def evaluate_benchmark(
    benchmark_path: Path,
    openai_api_key: str,
    judge_model: str = "gpt-4-turbo-2024-04-09",
) -> Dict[str, Any]:
    """
    Evaluate all claims in a benchmark result file.

    Args:
        benchmark_path: Path to the benchmark JSON file.
        openai_api_key: OpenAI API key for the judge.
        judge_model: GPT-4 model to use as judge.

    Returns:
        Complete evaluation result dict.
    """
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    claims = benchmark.get("claims", [])

    if not claims:
        print("  No claims found in benchmark file.")
        return {"error": "No claims to evaluate"}

    print(f"\n  Evaluating {len(claims)} claims from: {benchmark['query']}")
    print(f"  Judge model: {judge_model}\n")

    evaluations = []
    for i, claim_data in enumerate(claims, 1):
        statement = claim_data["statement"]
        print(f"  [{i}/{len(claims)}] {statement[:80]}...")

        # Fetch ground truth from Wikipedia
        ground_truth = await fetch_wikipedia_context(statement)

        # Judge the claim
        judgment = await judge_claim(
            claim=statement,
            ground_truth=ground_truth,
            openai_api_key=openai_api_key,
            model=judge_model,
        )

        evaluation = {
            "claim_id": claim_data.get("id", f"claim_{i}"),
            "statement": statement,
            "confidence": claim_data.get("confidence", 0.0),
            "ground_truth_excerpt": ground_truth[:500],
            **judgment,
        }
        evaluations.append(evaluation)

        verdict_icon = {
            "supported": "+",
            "partially_supported": "~",
            "unsupported": "!",
            "insufficient_evidence": "?",
        }.get(judgment["verdict"], "?")

        print(f"           [{verdict_icon}] Score: {judgment['score']}/10 — {judgment['verdict']}")

        # Rate limiting — be respectful to the API
        time.sleep(0.5)

    # Aggregate scores
    scores = [e["score"] for e in evaluations]
    mean_score = sum(scores) / len(scores) if scores else 0
    factual_accuracy = round(mean_score * 10, 1)

    result = {
        "benchmark_id": benchmark.get("benchmark_id", "unknown"),
        "query": benchmark["query"],
        "evaluation_timestamp": datetime.now(tz=UTC).isoformat(),
        "judge_model": judge_model,
        "num_claims": len(claims),
        "factual_accuracy_score": factual_accuracy,
        "mean_claim_score": round(mean_score, 2),
        "claims_above_7": sum(1 for s in scores if s >= 7),
        "claims_below_4": sum(1 for s in scores if s < 4),
        "score_distribution": {
            "9-10 (excellent)": sum(1 for s in scores if s >= 9),
            "7-8 (good)": sum(1 for s in scores if 7 <= s < 9),
            "5-6 (partial)": sum(1 for s in scores if 5 <= s < 7),
            "3-4 (poor)": sum(1 for s in scores if 3 <= s < 5),
            "0-2 (wrong)": sum(1 for s in scores if s < 3),
        },
        "claim_evaluations": evaluations,
    }

    # Print summary
    print(f"\n  {'=' * 50}")
    print("  EVALUATION SUMMARY")
    print(f"  {'=' * 50}")
    print(f"  Factual Accuracy Score: {factual_accuracy}/100")
    print(f"  Mean Claim Score:       {mean_score:.1f}/10")
    print(f"  Claims ≥ 7 (good+):    {result['claims_above_7']}/{len(claims)}")
    print(f"  Claims < 4 (poor):     {result['claims_below_4']}/{len(claims)}")
    print(f"  {'=' * 50}\n")

    return result


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print an aggregate summary across multiple evaluations."""
    print(f"\n  {'=' * 60}")
    print(f"  AGGREGATE SUMMARY ({len(results)} benchmarks)")
    print(f"  {'=' * 60}")
    print(f"  {'Query':<50} {'Score':<10}")
    print(f"  {'─' * 60}")

    scores = []
    for r in results:
        if "error" in r:
            continue
        score = r["factual_accuracy_score"]
        scores.append(score)
        query_short = r["query"][:48]
        print(f"  {query_short:<50} {score}/100")

    if scores:
        avg = sum(scores) / len(scores)
        print(f"  {'─' * 60}")
        print(f"  {'AVERAGE':<50} {avg:.1f}/100")
    print(f"  {'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgentForge Factual Accuracy Evaluator (GPT-4 as Judge)",
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Benchmark result JSON file(s) to evaluate",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4-turbo-2024-04-09",
        help="OpenAI model to use as judge (default: gpt-4-turbo)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for evaluation results (default: alongside input with _eval suffix)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print aggregate summary across all input files",
    )

    args = parser.parse_args()

    # Check for OpenAI API key
    import os

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print(
            "\n  [ERROR] OPENAI_API_KEY is required for GPT-4 judge evaluation.\n"
            "  Set it in your .env file or environment.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    import asyncio

    all_results = []
    for filepath in args.files:
        if not filepath.exists():
            print(f"  [SKIP] File not found: {filepath}")
            continue

        result = asyncio.run(evaluate_benchmark(filepath, openai_api_key, args.judge_model))
        all_results.append(result)

        # Save evaluation result
        if args.output:
            output_path = args.output
        else:
            output_path = filepath.with_name(filepath.stem + "_eval.json")

        output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"  Evaluation saved to: {output_path}")

    if args.summary and len(all_results) > 1:
        print_summary(all_results)


if __name__ == "__main__":
    main()
