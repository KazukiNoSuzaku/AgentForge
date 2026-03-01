"""
LangGraph state schema for the AgentForge research pipeline.

The ResearchState TypedDict is the single shared state that flows through
every node in the graph. Fields with Annotated reducers are accumulated
across node updates; plain fields are overwritten by the latest node output.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional

from typing_extensions import TypedDict

from src.models.schemas import (
    AnalysisResult,
    CriticFeedback,
    ResearchFinding,
    SubQuestion,
    ThinkingStep,
)


def _replace(left: Any, right: Any) -> Any:
    """Reducer that always replaces with the new value (default behaviour)."""
    return right


class ResearchState(TypedDict, total=False):
    """
    Shared state that flows through every node of the research graph.

    Fields annotated with operator.add are *accumulated* across updates
    (i.e. lists grow as each agent appends to them).  All other fields
    are overwritten by the returning node.

    total=False means all fields are optional when constructing the initial
    state — nodes are responsible for populating their outputs.
    """

    # ------------------------------------------------------------------
    # Input — set once by the entry point, never modified
    # ------------------------------------------------------------------
    query: str

    # ------------------------------------------------------------------
    # Planner output
    # ------------------------------------------------------------------
    sub_questions: List[SubQuestion]
    research_plan: str  # Human-readable plan narrative

    # ------------------------------------------------------------------
    # Researcher output — accumulated across parallel sub-question runs
    # ------------------------------------------------------------------
    research_findings: Annotated[List[ResearchFinding], operator.add]

    # ------------------------------------------------------------------
    # Analyst output
    # ------------------------------------------------------------------
    analysis: Optional[AnalysisResult]

    # ------------------------------------------------------------------
    # Writer output
    # ------------------------------------------------------------------
    draft_report: str  # Raw markdown content of the current draft
    revision_count: int  # How many Writer→Critic loops have run

    # ------------------------------------------------------------------
    # Critic output
    # ------------------------------------------------------------------
    critic_feedback: Optional[CriticFeedback]

    # ------------------------------------------------------------------
    # Final output — set by Critic when quality threshold is met
    # ------------------------------------------------------------------
    final_report: str

    # ------------------------------------------------------------------
    # Pipeline metadata — accumulated across all agents
    # ------------------------------------------------------------------
    agent_statuses: Dict[str, str]  # agent_name → AgentStatus value
    thinking_steps: Annotated[List[ThinkingStep], operator.add]
    errors: Annotated[List[str], operator.add]


def create_initial_state(query: str) -> ResearchState:
    """
    Build the initial state for a new research run.

    Args:
        query: The user's research question.

    Returns:
        A fully initialized ResearchState with sensible defaults.
    """
    return ResearchState(
        query=query,
        sub_questions=[],
        research_plan="",
        research_findings=[],
        analysis=None,
        draft_report="",
        revision_count=0,
        critic_feedback=None,
        final_report="",
        agent_statuses={
            "planner": "idle",
            "researcher": "idle",
            "analyst": "idle",
            "writer": "idle",
            "critic": "idle",
        },
        thinking_steps=[],
        errors=[],
    )
