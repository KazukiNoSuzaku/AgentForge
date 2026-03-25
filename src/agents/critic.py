"""
Critic Agent

Reviews the Writer's report for quality across five dimensions:
  1. Citation accuracy (unsupported claims / hallucination check)
  2. Completeness (all sub-questions addressed)
  3. Logical consistency (contradictions, non-sequiturs)
  4. Balanced perspective (missing stakeholders or viewpoints)
  5. Factual grounding (every claim traceable to provided sources)

If quality_score < threshold AND revision_count < max_loops, it triggers
a revision cycle. Otherwise, it approves the report as final.
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
from src.utils.citations import build_citation_map, validate_citations
from src.utils.llm import LLMManager, get_llm_manager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

CRITIC_SYSTEM_PROMPT = """You are the Critic agent for AgentForge, an advanced multi-agent research system.

Your role is to rigorously evaluate research reports before they are delivered to users.
You are the final quality gate — your evaluation directly determines whether the report
is approved or sent back for revision.

## Evaluation Criteria

### 1. Citation Accuracy (weight: 30%)
- Every factual claim must be supported by an inline citation [N]
- Citations must correspond to real sources provided in the research findings
- Invented URLs or misattributed citations are critical failures

### 2. Completeness (weight: 25%)
- All sub-questions identified in the research plan must be addressed
- No significant aspect of the main question should be entirely omitted
- Gaps acknowledged in the text are acceptable; unacknowledged gaps are not

### 3. Logical Consistency (weight: 20%)
- Arguments must flow logically
- Contradictions within the report (not between sources) must be resolved
- Conclusions must follow from the evidence presented

### 4. Balanced Perspective (weight: 15%)
- Multiple stakeholder viewpoints should be represented where relevant
- Avoid reporting only one side of genuinely contested issues
- Acknowledge dissenting views even when they are in the minority

### 5. Factual Grounding / Hallucination Detection (weight: 10%)
- Cross-check key claims against the provided research findings
- Flag any claim that cannot be traced to the provided sources
- Distinguish between claims that are in the findings (good) vs. invented (bad)

## Quality Score Rubric

- **0.90–1.00**: Publish-ready. No significant issues.
- **0.75–0.89**: Minor issues only. Approve with feedback for context.
- **0.50–0.74**: Significant issues requiring revision. Trigger revision.
- **0.00–0.49**: Major problems. Trigger revision with detailed remediation.

## Revision Threshold

Trigger revision if: quality_score < threshold AND revision_count < max_revisions.

## Output Format

Return valid JSON matching CriticFeedback exactly:
{
  "quality_score": 0.0-1.0,
  "unsupported_claims": ["Claim text that lacks citation or is not in sources", ...],
  "missing_perspectives": ["Stakeholder/viewpoint not covered", ...],
  "logical_gaps": ["Specific reasoning gap description", ...],
  "citation_issues": ["Description of citation problem", ...],
  "requires_revision": true/false,
  "feedback_summary": "2-3 sentence overall assessment and primary revision directive",
  "strengths": ["What the report does well and should be preserved", ...]
}

## Critical Rules

- Be specific and actionable — vague feedback ("improve clarity") is useless
- Identify exact claims or passages with issues
- If the report is good, say so — don't invent problems
- "strengths" must include at least 2 positive observations
"""


# ---------------------------------------------------------------------------
# Agent Class
# ---------------------------------------------------------------------------


class CriticAgent:
    """
    Evaluates a research report and decides whether it needs revision.

    Uses the LLM to assess the report across multiple quality dimensions,
    then applies the configured quality threshold to determine if a
    revision loop should be triggered.
    """

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self._llm = llm_manager or get_llm_manager()
        self._config = config or get_config()

    async def critique(
        self,
        report: str,
        findings: List[ResearchFinding],
        sub_questions: List[SubQuestion],
        analysis: Optional[AnalysisResult],
        revision_count: int,
    ) -> CriticFeedback:
        """
        Evaluate a research report and produce structured feedback.

        Args:
            report: The markdown report to evaluate.
            findings: Research findings (used for citation cross-checking).
            sub_questions: The original sub-questions the report should address.
            analysis: The Analyst's structured output (for claim verification).
            revision_count: How many revision cycles have already occurred.

        Returns:
            A CriticFeedback with quality_score and revision decision.
        """
        logger.info(
            "Critic: evaluating report (revision %d, threshold %.2f)",
            revision_count,
            self._config.quality_threshold,
        )

        # Build citation map for validation
        citation_map = build_citation_map(findings)
        citation_issues_auto = validate_citations(report, citation_map)

        # Format sub-questions for prompt
        sq_list = "\n".join(
            f"  {i}. [{sq.id}] {sq.question}" for i, sq in enumerate(sub_questions, 1)
        )

        # Format top findings summary for cross-referencing
        findings_summary = "\n".join(
            f"- [{f.source_type}] {f.title}: {f.content[:200]}..."
            if len(f.content) > 200
            else f"- [{f.source_type}] {f.title}: {f.content}"
            for f in findings[:20]
        )

        # Include any automatically detected citation issues
        auto_issues_text = (
            "\n**Auto-detected citation issues:**\n"
            + "\n".join(f"- {issue}" for issue in citation_issues_auto)
            if citation_issues_auto
            else ""
        )

        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Quality threshold for revision: {self._config.quality_threshold}\n"
                    f"Current revision count: {revision_count} "
                    f"(max: {self._config.max_revision_loops})\n\n"
                    f"Sub-questions the report must address:\n{sq_list}\n\n"
                    f"Research findings (for cross-referencing):\n{findings_summary}\n"
                    f"{auto_issues_text}\n\n"
                    f"## Report to Evaluate\n\n{report}"
                )
            ),
        ]

        feedback = await self._llm.ainvoke_structured(messages, CriticFeedback)

        # Add any automatically detected citation issues the LLM might have missed
        for issue in citation_issues_auto:
            if issue not in feedback.citation_issues:
                feedback.citation_issues.append(issue)

        # Enforce revision logic: no more revisions after max loops
        if revision_count >= self._config.max_revision_loops:
            if feedback.requires_revision:
                logger.info(
                    "Critic: would revise but max loops (%d) reached. Approving.",
                    self._config.max_revision_loops,
                )
            feedback = CriticFeedback(
                quality_score=feedback.quality_score,
                unsupported_claims=feedback.unsupported_claims,
                missing_perspectives=feedback.missing_perspectives,
                logical_gaps=feedback.logical_gaps,
                citation_issues=feedback.citation_issues,
                requires_revision=False,  # Force approval
                feedback_summary=feedback.feedback_summary,
                strengths=feedback.strengths,
            )

        logger.info(
            "Critic: quality_score=%.2f, requires_revision=%s",
            feedback.quality_score,
            feedback.requires_revision,
        )
        return feedback


# ---------------------------------------------------------------------------
# LangGraph Node
# ---------------------------------------------------------------------------


async def critic_node(state: ResearchState) -> dict:
    """
    LangGraph node: critique the current draft and decide on revision.

    This node always runs after the Writer. It either:
    - Sets requires_revision=True and sends the draft back to the Writer, or
    - Sets requires_revision=False and marks the report as final.

    Args:
        state: Current ResearchState (must contain draft_report).

    Returns:
        State delta with critic_feedback, final_report (if approved),
        agent_statuses, thinking_steps, and errors.
    """
    draft = state.get("draft_report", "")
    findings = state.get("research_findings", [])
    sub_questions = state.get("sub_questions", [])
    analysis = state.get("analysis")
    revision_count = state.get("revision_count", 0)

    thinking_step_start = ThinkingStep(
        agent="critic",
        action="reviewing_report",
        content=(
            f"Reviewing report quality (revision {revision_count}). "
            f"Checking citations, completeness, logic, and hallucinations."
        ),
        timestamp=datetime.now(),
    )

    if not draft:
        return {
            "critic_feedback": None,
            "agent_statuses": {**state.get("agent_statuses", {}), "critic": "error"},
            "thinking_steps": [thinking_step_start],
            "errors": ["Critic: no draft report to evaluate"],
        }

    agent = CriticAgent()

    try:
        feedback = await agent.critique(
            report=draft,
            findings=findings,
            sub_questions=sub_questions,
            analysis=analysis,
            revision_count=revision_count,
        )

        # If approved, mark as final
        final_report = draft if not feedback.requires_revision else ""

        thinking_step_done = ThinkingStep(
            agent="critic",
            action="review_complete",
            content=(
                f"Quality score: {feedback.quality_score:.2f}. "
                + ("Sending for revision." if feedback.requires_revision else "Report approved.")
            ),
            timestamp=datetime.now(),
        )

        return {
            "critic_feedback": feedback,
            "final_report": final_report,
            "agent_statuses": {**state.get("agent_statuses", {}), "critic": "done"},
            "thinking_steps": [thinking_step_start, thinking_step_done],
            "errors": [],
        }

    except Exception as exc:
        error_msg = f"Critic failed: {exc}"
        logger.exception("Critic node error")
        # On Critic failure, approve the draft to avoid blocking the pipeline
        return {
            "critic_feedback": None,
            "final_report": draft,
            "agent_statuses": {**state.get("agent_statuses", {}), "critic": "error"},
            "thinking_steps": [thinking_step_start],
            "errors": [error_msg],
        }
