"""
Tests for the LangGraph graph structure and routing logic.

Verifies conditional edges, revision loop behaviour, state propagation,
and end-to-end pipeline integration with fully mocked agents.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph import build_graph, get_graph, should_revise
from src.models.schemas import CriticFeedback
from src.state import ResearchState, create_initial_state


# ---------------------------------------------------------------------------
# Unit Tests: should_revise routing function
# ---------------------------------------------------------------------------


class TestShouldRevise:
    """Unit tests for the conditional edge routing function."""

    def test_returns_finish_when_no_feedback(self):
        state = create_initial_state("test query")
        assert should_revise(state) == "finish"

    def test_returns_revise_when_requires_revision_true(self):
        state = create_initial_state("test query")
        state["critic_feedback"] = CriticFeedback(
            quality_score=0.60,
            unsupported_claims=["Some claim"],
            missing_perspectives=[],
            logical_gaps=[],
            citation_issues=[],
            requires_revision=True,
            feedback_summary="Needs work.",
            strengths=["Good structure"],
        )
        assert should_revise(state) == "revise"

    def test_returns_finish_when_requires_revision_false(self):
        state = create_initial_state("test query")
        state["critic_feedback"] = CriticFeedback(
            quality_score=0.88,
            unsupported_claims=[],
            missing_perspectives=[],
            logical_gaps=[],
            citation_issues=[],
            requires_revision=False,
            feedback_summary="Looks good.",
            strengths=["Comprehensive", "Well-cited"],
        )
        assert should_revise(state) == "finish"

    def test_returns_finish_when_high_quality(self):
        state = create_initial_state("test query")
        state["critic_feedback"] = CriticFeedback(
            quality_score=0.95,
            unsupported_claims=[],
            missing_perspectives=[],
            logical_gaps=[],
            citation_issues=[],
            requires_revision=False,
            feedback_summary="Excellent report.",
            strengths=["Outstanding research", "Perfect citations"],
        )
        assert should_revise(state) == "finish"


# ---------------------------------------------------------------------------
# Unit Tests: Graph Structure
# ---------------------------------------------------------------------------


class TestGraphStructure:
    """Tests that verify the graph is constructed correctly."""

    def test_build_graph_returns_compiled_graph(self):
        """build_graph() should return a compiled StateGraph without errors."""
        graph = build_graph()
        assert graph is not None

    def test_get_graph_returns_singleton(self):
        """get_graph() should return the same instance on repeated calls."""
        graph1 = get_graph()
        graph2 = get_graph()
        assert graph1 is graph2

    def test_graph_has_all_expected_nodes(self):
        """The compiled graph should expose the expected node names."""
        graph = build_graph()
        # LangGraph compiled graphs expose nodes via .nodes dict
        node_names = set(graph.nodes.keys())
        expected = {"planner", "researcher", "analyst", "writer", "critic"}
        assert expected.issubset(node_names), (
            f"Missing nodes: {expected - node_names}"
        )


# ---------------------------------------------------------------------------
# Integration Tests: End-to-End Pipeline (all agents mocked)
# ---------------------------------------------------------------------------


class TestGraphEndToEnd:
    """
    Integration tests that run the full pipeline with all agent nodes mocked.

    These tests verify state propagation and routing without real LLM or API calls.
    """

    def _make_node_mock(self, return_value: dict):
        """Create an async mock function that returns the given state delta."""
        async def mock_node(state: ResearchState) -> dict:
            return return_value
        return mock_node

    @pytest.mark.asyncio
    async def test_pipeline_runs_to_completion_on_first_pass(
        self,
        sample_query,
        sample_sub_questions,
        sample_findings,
        sample_analysis,
        sample_draft_report,
        sample_critic_feedback_pass,
    ):
        """A high-quality report should flow through all 5 agents exactly once."""
        call_order = []

        async def mock_planner(state):
            call_order.append("planner")
            return {
                "sub_questions": sample_sub_questions,
                "research_plan": "Research plan",
                "agent_statuses": {"planner": "done"},
                "thinking_steps": [],
                "errors": [],
            }

        async def mock_researcher(state):
            call_order.append("researcher")
            return {
                "research_findings": sample_findings,
                "agent_statuses": {"researcher": "done"},
                "thinking_steps": [],
                "errors": [],
            }

        async def mock_analyst(state):
            call_order.append("analyst")
            return {
                "analysis": sample_analysis,
                "agent_statuses": {"analyst": "done"},
                "thinking_steps": [],
                "errors": [],
            }

        async def mock_writer(state):
            call_order.append("writer")
            return {
                "draft_report": sample_draft_report,
                "revision_count": state.get("revision_count", 0),
                "agent_statuses": {"writer": "done"},
                "thinking_steps": [],
                "errors": [],
            }

        async def mock_critic(state):
            call_order.append("critic")
            return {
                "critic_feedback": sample_critic_feedback_pass,
                "final_report": sample_draft_report,  # Approved
                "agent_statuses": {"critic": "done"},
                "thinking_steps": [],
                "errors": [],
            }

        with patch("src.graph.planner_node", mock_planner), \
             patch("src.graph.researcher_node", mock_researcher), \
             patch("src.graph.analyst_node", mock_analyst), \
             patch("src.graph.writer_node", mock_writer), \
             patch("src.graph.critic_node", mock_critic):

            graph = build_graph()
            initial_state = create_initial_state(sample_query)
            final_state = await graph.ainvoke(initial_state, config={"recursion_limit": 15})

        assert "planner" in call_order
        assert "researcher" in call_order
        assert "analyst" in call_order
        assert "writer" in call_order
        assert "critic" in call_order
        assert call_order.count("writer") == 1  # Only one write pass

    @pytest.mark.asyncio
    async def test_pipeline_triggers_one_revision_on_low_score(
        self,
        sample_query,
        sample_sub_questions,
        sample_findings,
        sample_analysis,
        sample_draft_report,
        sample_critic_feedback_fail,
        sample_critic_feedback_pass,
    ):
        """A low-quality initial draft should trigger exactly one revision cycle."""
        call_order = []
        critic_call_count = [0]

        async def mock_planner(state):
            call_order.append("planner")
            return {
                "sub_questions": sample_sub_questions,
                "research_plan": "Research plan",
                "agent_statuses": {"planner": "done"},
                "thinking_steps": [],
                "errors": [],
            }

        async def mock_researcher(state):
            call_order.append("researcher")
            return {
                "research_findings": sample_findings,
                "agent_statuses": {"researcher": "done"},
                "thinking_steps": [],
                "errors": [],
            }

        async def mock_analyst(state):
            call_order.append("analyst")
            return {
                "analysis": sample_analysis,
                "agent_statuses": {"analyst": "done"},
                "thinking_steps": [],
                "errors": [],
            }

        async def mock_writer(state):
            call_order.append("writer")
            return {
                "draft_report": sample_draft_report,
                "revision_count": state.get("revision_count", 0) + (1 if state.get("critic_feedback") else 0),
                "agent_statuses": {"writer": "done"},
                "thinking_steps": [],
                "errors": [],
            }

        async def mock_critic(state):
            call_order.append("critic")
            critic_call_count[0] += 1

            if critic_call_count[0] == 1:
                # First pass: fail → trigger revision
                feedback = sample_critic_feedback_fail
                final = ""
            else:
                # Second pass: pass → approve
                feedback = sample_critic_feedback_pass
                final = sample_draft_report

            return {
                "critic_feedback": feedback,
                "final_report": final,
                "agent_statuses": {"critic": "done"},
                "thinking_steps": [],
                "errors": [],
            }

        with patch("src.graph.planner_node", mock_planner), \
             patch("src.graph.researcher_node", mock_researcher), \
             patch("src.graph.analyst_node", mock_analyst), \
             patch("src.graph.writer_node", mock_writer), \
             patch("src.graph.critic_node", mock_critic):

            graph = build_graph()
            initial_state = create_initial_state(sample_query)
            final_state = await graph.ainvoke(initial_state, config={"recursion_limit": 20})

        assert call_order.count("writer") == 2, "Expected writer to run twice (1 draft + 1 revision)"
        assert call_order.count("critic") == 2, "Expected critic to run twice"

    @pytest.mark.asyncio
    async def test_pipeline_final_state_contains_report(
        self,
        sample_query,
        sample_sub_questions,
        sample_findings,
        sample_analysis,
        sample_draft_report,
        sample_critic_feedback_pass,
    ):
        """The final state should contain a non-empty final_report."""
        async def mock_planner(state):
            return {"sub_questions": sample_sub_questions, "research_plan": "plan",
                    "agent_statuses": {"planner": "done"}, "thinking_steps": [], "errors": []}

        async def mock_researcher(state):
            return {"research_findings": sample_findings,
                    "agent_statuses": {"researcher": "done"}, "thinking_steps": [], "errors": []}

        async def mock_analyst(state):
            return {"analysis": sample_analysis,
                    "agent_statuses": {"analyst": "done"}, "thinking_steps": [], "errors": []}

        async def mock_writer(state):
            return {"draft_report": sample_draft_report, "revision_count": 0,
                    "agent_statuses": {"writer": "done"}, "thinking_steps": [], "errors": []}

        async def mock_critic(state):
            return {"critic_feedback": sample_critic_feedback_pass,
                    "final_report": sample_draft_report,
                    "agent_statuses": {"critic": "done"}, "thinking_steps": [], "errors": []}

        with patch("src.graph.planner_node", mock_planner), \
             patch("src.graph.researcher_node", mock_researcher), \
             patch("src.graph.analyst_node", mock_analyst), \
             patch("src.graph.writer_node", mock_writer), \
             patch("src.graph.critic_node", mock_critic):

            graph = build_graph()
            initial_state = create_initial_state(sample_query)
            final_state = await graph.ainvoke(initial_state, config={"recursion_limit": 15})

        assert final_state.get("final_report"), "final_report should be non-empty"
        assert "## Bibliography" in final_state["final_report"] or "#" in final_state["final_report"]
