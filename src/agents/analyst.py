"""
Analyst Agent

Synthesizes all research findings into structured insights: cross-source
patterns, contradictions, information gaps, and discrete claims with
confidence scores based on source corroboration.

The Analyst's output is the key input for the Writer — it transforms raw
evidence into a structured analytical scaffold for the report.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import AppConfig, get_config
from src.models.schemas import AnalysisResult, Claim, ResearchFinding, ThinkingStep
from src.state import ResearchState
from src.utils.llm import LLMManager, get_llm_manager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

ANALYST_SYSTEM_PROMPT = """You are the Analyst agent for AgentForge, an advanced research system.

Your role is to critically synthesize research findings into structured analytical insights.

## Input

You receive:
- Research findings with content, relevance scores, key insights, and source URLs
- Multiple sub-questions with their findings grouped together

## Task

Produce a rigorous, evidence-based analysis:

### 1. Patterns
Identify 3-6 recurring themes or consistent findings across multiple sources.
Format: "Pattern: [claim]. Evidence: [source URLs]"

### 2. Contradictions
Identify 1-4 areas where sources disagree or provide conflicting information.
Format: "Contradiction: [source A says X] vs [source B says Y]. URLs: [...]"

### 3. Information Gaps
Identify 2-4 aspects of the question that the research did NOT adequately cover.
These gaps are important for the Critic to assess report completeness.

### 4. Claims
Extract 5-10 discrete, citable claims. For each claim:
- **Statement**: Clear, specific, one-sentence claim
- **Confidence** (0.0–1.0):
  - 0.9–1.0: Multiple high-quality sources agree (peer-reviewed or authoritative)
  - 0.7–0.9: 2+ sources agree, or one very authoritative source
  - 0.5–0.7: Single source or sources with caveats
  - < 0.5: Speculative or single weak source
- **Supporting sources**: URLs of sources that support this claim
- **Contradicting sources**: URLs of sources that contradict it

### 5. Confidence Summary
A 2-3 sentence narrative assessing the overall strength and reliability of the research base.

## Output Format

Return valid JSON matching AnalysisResult exactly:
{
  "patterns": ["Pattern: ...", ...],
  "contradictions": ["Contradiction: ...", ...],
  "gaps": ["Gap: ...", ...],
  "claims": [
    {
      "id": "claim_1",
      "statement": "...",
      "confidence": 0.85,
      "supporting_sources": ["https://..."],
      "contradicting_sources": [],
      "sub_question_id": "sq_1"
    },
    ...
  ],
  "confidence_summary": "..."
}

## Critical Rules

- Ground EVERY claim in provided sources — do not extrapolate beyond the evidence
- Confidence scores must reflect actual source corroboration, not topic familiarity
- Contradictions are valuable — do not smooth over genuine disagreements
- Be specific: vague claims like "AI is complex" are not useful
"""


# ---------------------------------------------------------------------------
# Helper: Format Findings for Prompt
# ---------------------------------------------------------------------------


def _format_findings_for_prompt(findings: List[ResearchFinding]) -> str:
    """
    Render findings as a condensed text block for the Analyst's prompt.

    Truncates each finding to stay within LLM context limits while
    preserving the essential information (content, URL, insights).
    """
    lines = []
    for i, f in enumerate(findings, 1):
        lines.append(f"[Finding {i}]")
        lines.append(f"Sub-question ID: {f.sub_question_id}")
        lines.append(f"Sub-question: {f.sub_question}")
        lines.append(f"Source: {f.source_type} | Relevance: {f.relevance_score:.2f}")
        lines.append(f"Title: {f.title}")
        lines.append(f"URL: {f.url}")
        lines.append(f"Content: {f.content[:600]}")
        if f.key_insights:
            lines.append("Key Insights:")
            for insight in f.key_insights[:4]:
                lines.append(f"  - {insight}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent Class
# ---------------------------------------------------------------------------


class AnalystAgent:
    """
    Analyzes aggregated research findings to produce structured insights.

    Identifies patterns, contradictions, gaps, and discrete claims with
    confidence scores. This agent is the analytical backbone of the pipeline.
    """

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self._llm = llm_manager or get_llm_manager()
        self._config = config or get_config()

    async def analyze(
        self,
        findings: List[ResearchFinding],
        query: str,
    ) -> AnalysisResult:
        """
        Synthesize research findings into structured analytical insights.

        Args:
            findings: All research findings from the Researcher agent.
            query: The original user research question (for context).

        Returns:
            A validated AnalysisResult with patterns, claims, etc.

        Raises:
            RuntimeError: If analysis fails and LLM fallback is unavailable.
        """
        logger.info("Analyst: analyzing %d findings", len(findings))

        # Sort by relevance score, highest first
        sorted_findings = sorted(findings, key=lambda f: f.relevance_score, reverse=True)
        # Cap at 25 findings to avoid huge prompts
        top_findings = sorted_findings[:25]

        findings_text = _format_findings_for_prompt(top_findings)

        messages = [
            SystemMessage(content=ANALYST_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Original Research Question: {query}\n\n"
                    f"Total Findings: {len(findings)} "
                    f"(showing top {len(top_findings)} by relevance)\n\n"
                    f"{findings_text}\n\n"
                    "Produce a rigorous AnalysisResult JSON."
                )
            ),
        ]

        analysis = await self._llm.ainvoke_structured(messages, AnalysisResult)

        # Assign IDs to claims that don't have them
        for i, claim in enumerate(analysis.claims, 1):
            if not claim.id or claim.id == "string":
                analysis.claims[i - 1] = Claim(
                    id=f"claim_{i}",
                    statement=claim.statement,
                    confidence=claim.confidence,
                    supporting_sources=claim.supporting_sources,
                    contradicting_sources=claim.contradicting_sources,
                    sub_question_id=claim.sub_question_id,
                )

        logger.info(
            "Analyst: %d patterns, %d contradictions, %d gaps, %d claims",
            len(analysis.patterns),
            len(analysis.contradictions),
            len(analysis.gaps),
            len(analysis.claims),
        )
        return analysis


# ---------------------------------------------------------------------------
# LangGraph Node
# ---------------------------------------------------------------------------


async def analyst_node(state: ResearchState) -> dict:
    """
    LangGraph node: run the Analyst agent on all collected research findings.

    Args:
        state: Current ResearchState (must contain research_findings).

    Returns:
        State delta with analysis, agent_statuses, thinking_steps, and errors.
    """
    findings = state.get("research_findings", [])
    query = state.get("query", "")

    thinking_step_start = ThinkingStep(
        agent="analyst",
        action="starting_analysis",
        content=f"Synthesizing {len(findings)} research findings into structured insights",
        timestamp=datetime.now(),
    )

    if not findings:
        return {
            "analysis": None,
            "agent_statuses": {**state.get("agent_statuses", {}), "analyst": "error"},
            "thinking_steps": [thinking_step_start],
            "errors": ["Analyst: no research findings to analyze"],
        }

    agent = AnalystAgent()

    try:
        analysis = await agent.analyze(findings, query)

        high_conf = sum(1 for c in analysis.claims if c.confidence >= 0.8)
        thinking_step_done = ThinkingStep(
            agent="analyst",
            action="analysis_complete",
            content=(
                f"Identified {len(analysis.patterns)} patterns, "
                f"{len(analysis.contradictions)} contradictions, "
                f"{len(analysis.gaps)} gaps, and "
                f"{len(analysis.claims)} claims "
                f"({high_conf} high-confidence)"
            ),
            timestamp=datetime.now(),
        )

        return {
            "analysis": analysis,
            "agent_statuses": {**state.get("agent_statuses", {}), "analyst": "done"},
            "thinking_steps": [thinking_step_start, thinking_step_done],
            "errors": [],
        }

    except Exception as exc:
        error_msg = f"Analyst failed: {exc}"
        logger.exception("Analyst node error")
        return {
            "analysis": None,
            "agent_statuses": {**state.get("agent_statuses", {}), "analyst": "error"},
            "thinking_steps": [thinking_step_start],
            "errors": [error_msg],
        }
