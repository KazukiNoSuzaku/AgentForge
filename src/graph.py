"""
LangGraph State Machine: AgentForge Research Pipeline

Defines the directed graph of agents and the conditional routing logic
that determines whether the Critic triggers a revision or approves the report.

Graph topology:
    START → Planner → Researcher → Analyst → Writer → Critic → (Writer | END)

The Critic→Writer back-edge creates the revision loop. The loop is bounded
by max_revision_loops in the application config (default: 2).
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agents.analyst import analyst_node
from src.agents.critic import critic_node
from src.agents.planner import planner_node
from src.agents.researcher import researcher_node
from src.agents.writer import writer_node
from src.state import ResearchState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing Logic
# ---------------------------------------------------------------------------


def should_revise(state: ResearchState) -> Literal["revise", "finish"]:
    """
    Conditional edge: decide whether the Critic wants a revision.

    Reads the CriticFeedback from state. If requires_revision is True,
    routes back to the Writer. Otherwise routes to END.

    Note: The CriticAgent already enforces the max_revision_loops cap by
    setting requires_revision=False when the limit is reached, so this
    function only needs to check the flag.

    Args:
        state: Current ResearchState after the Critic node has run.

    Returns:
        "revise" to route back to the Writer, "finish" to terminate.
    """
    feedback = state.get("critic_feedback")

    if feedback is None:
        # Critic didn't run or failed — end the pipeline
        logger.warning("should_revise: no critic_feedback found; routing to END")
        return "finish"

    if feedback.requires_revision:
        logger.info(
            "should_revise: quality=%.2f < threshold → routing to Writer for revision",
            feedback.quality_score,
        )
        return "revise"

    logger.info(
        "should_revise: quality=%.2f ≥ threshold → routing to END",
        feedback.quality_score,
    )
    return "finish"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """
    Construct and compile the AgentForge LangGraph research pipeline.

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    workflow = StateGraph(ResearchState)

    # ------------------------------------------------------------------ #
    # Register nodes
    # ------------------------------------------------------------------ #

    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)

    # ------------------------------------------------------------------ #
    # Add edges (linear pipeline)
    # ------------------------------------------------------------------ #

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "critic")

    # ------------------------------------------------------------------ #
    # Add conditional edge (revision loop)
    # ------------------------------------------------------------------ #

    workflow.add_conditional_edges(
        "critic",
        should_revise,
        {
            "revise": "writer",
            "finish": END,
        },
    )

    return workflow.compile()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_graph = None


def get_graph():
    """
    Return (or build) the module-level compiled graph singleton.

    The graph is compiled once and reused for all research runs in the
    same process. This avoids redundant compilation overhead.

    Returns:
        Compiled LangGraph StateGraph.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
        logger.info("AgentForge graph compiled successfully")
    return _graph
