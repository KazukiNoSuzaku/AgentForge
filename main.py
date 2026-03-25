"""
AgentForge CLI Entry Point

Run a research query from the command line and stream agent progress
to the terminal. Supports saving the final report to a file.

Usage:
    python main.py "Compare AI regulatory frameworks in the EU, US, and China"
    python main.py "What are the key drivers of inflation in 2024?" --output report.md
    python main.py "Explain quantum computing's impact on cryptography" --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Load .env before importing anything that reads config
load_dotenv()

from src.config import get_config
from src.graph import build_graph
from src.state import create_initial_state

console = Console()

AGENT_COLORS = {
    "planner": "blue",
    "researcher": "green",
    "analyst": "yellow",
    "writer": "magenta",
    "critic": "red",
}


def setup_logging(log_level: str) -> None:
    """Configure logging for the CLI run."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def print_agent_update(node_name: str, node_output: dict, verbose: bool) -> None:
    """
    Print a formatted update when an agent completes its work.

    Args:
        node_name: Name of the agent node that just ran.
        node_output: The state delta returned by the node.
        verbose: Whether to show thinking steps.
    """
    color = AGENT_COLORS.get(node_name, "white")
    status = node_output.get("agent_statuses", {}).get(node_name, "done")
    icon = "✓" if status == "done" else "✗"

    console.print(f"\n[{color}]{icon} [{node_name.upper()}][/{color}]", end=" ")

    # Print a brief summary per agent
    if node_name == "planner":
        sub_qs = node_output.get("sub_questions", [])
        console.print(f"Decomposed into {len(sub_qs)} sub-questions")
        for sq in sub_qs:
            console.print(f"  [dim]{sq.id}:[/dim] {sq.question}")

    elif node_name == "researcher":
        findings = node_output.get("research_findings", [])
        console.print(f"Collected {len(findings)} research findings")
        by_source = {}
        for f in findings:
            by_source[f.source_type] = by_source.get(f.source_type, 0) + 1
        for source, count in by_source.items():
            console.print(f"  [dim]{source}:[/dim] {count} findings")

    elif node_name == "analyst":
        analysis = node_output.get("analysis")
        if analysis:
            console.print(
                f"Identified {len(analysis.claims)} claims, "
                f"{len(analysis.patterns)} patterns, "
                f"{len(analysis.contradictions)} contradictions"
            )
        else:
            console.print("[red]Analysis failed[/red]")

    elif node_name == "writer":
        draft = node_output.get("draft_report", "")
        rev = node_output.get("revision_count", 0)
        console.print(f"{'Revised' if rev > 0 else 'Drafted'} report ({len(draft):,} chars)")

    elif node_name == "critic":
        feedback = node_output.get("critic_feedback")
        if feedback:
            color_score = (
                "green"
                if feedback.quality_score >= 0.75
                else "yellow"
                if feedback.quality_score >= 0.5
                else "red"
            )
            console.print(
                f"Quality score: [{color_score}]{feedback.quality_score:.2f}[/{color_score}] — "
                + ("Revision requested" if feedback.requires_revision else "Approved ✓")
            )
        else:
            console.print("Critic skipped or failed")

    # Show errors
    for error in node_output.get("errors", []):
        console.print(f"  [red]! {error}[/red]")

    # Show thinking steps if verbose
    if verbose:
        for step in node_output.get("thinking_steps", []):
            console.print(f"  [dim]→ {step.action}: {step.content}[/dim]")


async def run_research(
    query: str,
    verbose: bool = False,
) -> Optional[str]:
    """
    Execute the full research pipeline and stream progress to the console.

    Args:
        query: The research question to investigate.
        verbose: Whether to print thinking steps for each agent.

    Returns:
        The final markdown report, or None if the pipeline failed.
    """
    console.print(
        Panel(
            f"[bold]{query}[/bold]",
            title="[cyan]AgentForge Research Engine[/cyan]",
            subtitle="[dim]Multi-Agent Research Pipeline[/dim]",
        )
    )

    graph = build_graph()
    initial_state = create_initial_state(query)
    final_report = None

    try:
        # stream() yields {node_name: state_delta} for each completed node
        for update in graph.stream(
            initial_state,
            config={"recursion_limit": 20},
        ):
            for node_name, node_output in update.items():
                print_agent_update(node_name, node_output, verbose)

                # Capture final report whenever it's set
                if node_output.get("final_report"):
                    final_report = node_output["final_report"]

    except Exception as exc:
        console.print(f"\n[red bold]Pipeline error:[/red bold] {exc}")
        if verbose:
            console.print_exception()
        return None

    return final_report


def save_report(report: str, output_path: Path) -> None:
    """
    Save the research report to a markdown file.

    Args:
        report: Markdown report content.
        output_path: Destination file path.
    """
    output_path.write_text(report, encoding="utf-8")
    console.print(f"\n[green]Report saved to:[/green] {output_path.resolve()}")


def main() -> None:
    """CLI entry point for the AgentForge research engine."""
    parser = argparse.ArgumentParser(
        description="AgentForge: Multi-Agent Research Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Compare AI regulatory frameworks in the EU, US, and China"
  python main.py "What caused the 2008 financial crisis?" --output report.md
  python main.py "Quantum computing threats to cryptography" --verbose
        """,
    )
    parser.add_argument(
        "query",
        help="Research question to investigate",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to save the markdown report (default: print to stdout)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show agent thinking steps",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (default: from .env or INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    config = get_config()
    log_level = args.log_level or config.log_level
    setup_logging(log_level)

    # Run the async pipeline in a synchronous context
    report = asyncio.run(run_research(args.query, verbose=args.verbose))

    if report is None:
        console.print("\n[red bold]Research failed — check errors above.[/red bold]")
        sys.exit(1)

    console.print("\n")
    console.print(Panel("[green bold]Research Complete[/green bold]"))

    if args.output:
        save_report(report, args.output)
    else:
        # Print the report to stdout
        console.print("\n")
        console.print(Markdown(report))


if __name__ == "__main__":
    main()
