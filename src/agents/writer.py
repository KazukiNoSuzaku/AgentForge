"""
Writer Agent

Generates a well-structured markdown research report from the analysis and
raw findings. Supports revision cycles driven by Critic feedback — when
revising, the agent receives structured critique and must address each point.

Report structure:
  - Executive Summary
  - Introduction
  - Thematic sections (one per major analytical theme)
  - Conclusion
  - Inline citations [1], [2], ...
  - Bibliography (appended by citations.py)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import AppConfig, get_config
from src.models.schemas import (
    AnalysisResult,
    CriticFeedback,
    ResearchFinding,
    SubQuestion,
    ThinkingStep,
)
from src.state import ResearchState
from src.utils.citations import build_citation_map, inject_bibliography
from src.utils.llm import LLMManager, get_llm_manager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

WRITER_SYSTEM_PROMPT = """You are the Writer agent for AgentForge, an advanced research system.

Your role is to produce a comprehensive, well-cited markdown research report.

## Report Structure (Required)

1. **Executive Summary** (3-5 sentences): Key findings, main conclusions, significance
2. **Introduction**: Context, scope, and why this question matters
3. **[Thematic Sections]**: 3-5 sections, each addressing a major finding or comparison
   - Section titles should be specific and informative (not generic like "Section 1")
   - Each section addresses one or more sub-questions
4. **Conclusion**: Synthesis, implications, and open questions
5. **Bibliography**: Will be appended automatically — do NOT write it yourself

## Citation Rules

- EVERY factual claim must have an inline citation: [1], [2], etc.
- Citation numbers correspond to the bibliography provided
- Multiple citations for one claim: [1][3] (no spaces)
- Do not invent citations — only use numbers from the provided list
- If a claim lacks a citation, mark it clearly: [UNCITED - needs source]

## Writing Style

- Professional but accessible — written for a knowledgeable non-specialist
- Active voice, clear topic sentences, logical paragraph flow
- Specific claims over vague generalizations
- Acknowledge uncertainty: "According to [1]..." or "Evidence suggests..."
- Target length: 1200-2000 words (excluding bibliography)

## Formatting

Use standard markdown:
- # for main title
- ## for major sections
- ### for subsections
- **bold** for key terms on first use
- Bullet lists for comparisons or feature lists

IMPORTANT: Output only the markdown report content. Do not include preamble like "Here is the report:".
"""

WRITER_REVISION_PROMPT = """You are revising a research report based on feedback from the Critic agent.

## Revision Instructions

Address EVERY point raised in the Critic's feedback:
1. **Unsupported claims**: Add citations or qualify the claim with appropriate uncertainty language
2. **Missing perspectives**: Add a paragraph or subsection covering the missing viewpoint
3. **Logical gaps**: Fix the reasoning or add transitional content to bridge the gap
4. **Citation issues**: Correct any misattributed or wrong citations

## Key Rules

- Preserve all strengths identified by the Critic
- Do not remove content unless it is genuinely unsupported
- Every change should directly address a specific critique point
- Maintain the same overall structure unless the critique requires restructuring
- Target the same length range: 1200-2000 words

Output the complete revised report in markdown. Do not include meta-commentary about changes made.
"""


# ---------------------------------------------------------------------------
# Helper: Build Context for Writer Prompt
# ---------------------------------------------------------------------------


def _build_citation_reference(findings: List[ResearchFinding]) -> str:
    """
    Build a concise citation reference list for the Writer's prompt.

    Returns a numbered list of sources the Writer can cite by number.
    """
    citation_map = build_citation_map(findings)
    if not citation_map:
        return "No sources available."

    sorted_entries = sorted(citation_map.values(), key=lambda e: e.number)
    lines = ["Available citations (use [N] inline):"]
    for entry in sorted_entries:
        authors_str = (
            f"{entry.authors[0]} et al." if len(entry.authors) > 1
            else entry.authors[0] if entry.authors
            else ""
        )
        date_str = f" ({entry.published_date})" if entry.published_date else ""
        lines.append(
            f"[{entry.number}] {entry.title}{date_str} — {entry.source_type} — {entry.url}"
        )
        if authors_str:
            lines.append(f"      Authors: {authors_str}")
    return "\n".join(lines)


def _format_analysis_for_writer(analysis: AnalysisResult) -> str:
    """Render the AnalysisResult as readable text for the writer prompt."""
    lines = ["## Analysis Summary\n"]

    lines.append("### Patterns Identified")
    for p in analysis.patterns:
        lines.append(f"- {p}")
    lines.append("")

    lines.append("### Contradictions")
    for c in analysis.contradictions:
        lines.append(f"- {c}")
    lines.append("")

    lines.append("### Information Gaps")
    for g in analysis.gaps:
        lines.append(f"- {g}")
    lines.append("")

    lines.append("### Claims (with confidence scores)")
    for claim in analysis.claims:
        conf_label = (
            "HIGH" if claim.confidence >= 0.8
            else "MEDIUM" if claim.confidence >= 0.6
            else "LOW"
        )
        lines.append(
            f"- [{conf_label} {claim.confidence:.0%}] {claim.statement}"
        )
        if claim.supporting_sources:
            lines.append(f"  Supporting: {', '.join(claim.supporting_sources[:3])}")
        if claim.contradicting_sources:
            lines.append(f"  Contradicting: {', '.join(claim.contradicting_sources[:2])}")
    lines.append("")

    lines.append(f"### Confidence Summary\n{analysis.confidence_summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent Class
# ---------------------------------------------------------------------------


class WriterAgent:
    """
    Produces markdown research reports with inline citations.

    Supports both initial drafts (no prior feedback) and revision cycles
    (with structured Critic feedback to address specific issues).
    """

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self._llm = llm_manager or get_llm_manager()
        self._config = config or get_config()

    async def write(
        self,
        query: str,
        sub_questions: List[SubQuestion],
        findings: List[ResearchFinding],
        analysis: AnalysisResult,
        critic_feedback: Optional[CriticFeedback] = None,
        current_draft: Optional[str] = None,
        revision_number: int = 0,
    ) -> str:
        """
        Generate or revise a research report.

        Args:
            query: Original user research question.
            sub_questions: The decomposed sub-questions from the Planner.
            findings: All research findings (for citation extraction).
            analysis: Structured analysis from the Analyst.
            critic_feedback: Optional feedback for revision mode.
            current_draft: Current draft to revise (revision mode only).
            revision_number: Which revision cycle this is (0 = initial draft).

        Returns:
            Markdown-formatted research report with inline citations.
        """
        is_revision = critic_feedback is not None and current_draft

        system_prompt = WRITER_REVISION_PROMPT if is_revision else WRITER_SYSTEM_PROMPT
        citation_ref = _build_citation_reference(findings)
        analysis_text = _format_analysis_for_writer(analysis)

        sq_text = "\n".join(
            f"{i}. {sq.question}" for i, sq in enumerate(sub_questions, 1)
        )

        if is_revision:
            user_content = (
                f"Research Question: {query}\n\n"
                f"Revision Number: {revision_number}\n\n"
                f"## Critic Feedback to Address\n"
                f"Quality Score: {critic_feedback.quality_score:.2f}\n"
                f"Feedback Summary: {critic_feedback.feedback_summary}\n\n"
                f"**Unsupported Claims:**\n"
                + "\n".join(f"- {c}" for c in critic_feedback.unsupported_claims)
                + "\n\n**Missing Perspectives:**\n"
                + "\n".join(f"- {p}" for p in critic_feedback.missing_perspectives)
                + "\n\n**Logical Gaps:**\n"
                + "\n".join(f"- {g}" for g in critic_feedback.logical_gaps)
                + "\n\n**Citation Issues:**\n"
                + "\n".join(f"- {i}" for i in critic_feedback.citation_issues)
                + "\n\n**Strengths to Preserve:**\n"
                + "\n".join(f"- {s}" for s in critic_feedback.strengths)
                + f"\n\n## Current Draft\n\n{current_draft}\n\n"
                f"## Available Citations\n{citation_ref}\n\n"
                f"## Analysis\n{analysis_text}"
            )
        else:
            user_content = (
                f"Research Question: {query}\n\n"
                f"Sub-questions Addressed:\n{sq_text}\n\n"
                f"## Available Citations\n{citation_ref}\n\n"
                f"{analysis_text}"
            )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        logger.info(
            "Writer: generating %s (revision=%d)",
            "revision" if is_revision else "initial draft",
            revision_number,
        )

        draft = await self._llm.ainvoke(messages)

        # Append bibliography
        citation_map = build_citation_map(findings)
        report_with_bibliography = inject_bibliography(draft, citation_map)

        logger.info("Writer: draft complete (%d chars)", len(report_with_bibliography))
        return report_with_bibliography


# ---------------------------------------------------------------------------
# LangGraph Node
# ---------------------------------------------------------------------------


async def writer_node(state: ResearchState) -> dict:
    """
    LangGraph node: generate or revise the research report.

    Args:
        state: Current ResearchState.

    Returns:
        State delta with draft_report, revision_count, agent_statuses,
        thinking_steps, and errors.
    """
    query = state.get("query", "")
    sub_questions = state.get("sub_questions", [])
    findings = state.get("research_findings", [])
    analysis = state.get("analysis")
    critic_feedback = state.get("critic_feedback")
    current_draft = state.get("draft_report", "")
    revision_count = state.get("revision_count", 0)

    is_revision = bool(critic_feedback and current_draft and revision_count > 0)

    thinking_step_start = ThinkingStep(
        agent="writer",
        action="writing_report" if not is_revision else "revising_report",
        content=(
            f"{'Revising' if is_revision else 'Writing'} research report "
            f"(revision {revision_count})" if is_revision
            else "Writing initial research report draft"
        ),
        timestamp=datetime.now(),
    )

    if analysis is None:
        return {
            "draft_report": "",
            "agent_statuses": {**state.get("agent_statuses", {}), "writer": "error"},
            "thinking_steps": [thinking_step_start],
            "errors": ["Writer: cannot write report — analysis is missing"],
        }

    agent = WriterAgent()

    try:
        report = await agent.write(
            query=query,
            sub_questions=sub_questions,
            findings=findings,
            analysis=analysis,
            critic_feedback=critic_feedback if is_revision else None,
            current_draft=current_draft if is_revision else None,
            revision_number=revision_count,
        )

        new_revision_count = revision_count + 1 if is_revision else revision_count

        thinking_step_done = ThinkingStep(
            agent="writer",
            action="report_ready",
            content=f"Report {'revised' if is_revision else 'drafted'}: {len(report)} chars",
            timestamp=datetime.now(),
        )

        return {
            "draft_report": report,
            "revision_count": new_revision_count,
            "agent_statuses": {**state.get("agent_statuses", {}), "writer": "done"},
            "thinking_steps": [thinking_step_start, thinking_step_done],
            "errors": [],
        }

    except Exception as exc:
        error_msg = f"Writer failed: {exc}"
        logger.exception("Writer node error")
        return {
            "draft_report": current_draft,  # Preserve existing draft
            "agent_statuses": {**state.get("agent_statuses", {}), "writer": "error"},
            "thinking_steps": [thinking_step_start],
            "errors": [error_msg],
        }
