"""
Tests for the Planner agent.

Verifies query decomposition, sub-question structure, config enforcement,
and the LangGraph node interface — all without real LLM calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.planner import PlannerAgent, planner_node
from src.models.schemas import ResearchPlan, SubQuestion
from src.state import create_initial_state


class TestPlannerAgent:
    """Unit tests for PlannerAgent.plan()."""

    @pytest.mark.asyncio
    async def test_plan_returns_research_plan(
        self, mock_llm_manager, sample_research_plan
    ):
        """plan() should return a ResearchPlan with the LLM's structured output."""
        mock_llm_manager.ainvoke_structured.return_value = sample_research_plan

        agent = PlannerAgent(llm_manager=mock_llm_manager)
        result = await agent.plan("Compare AI regulatory frameworks in EU, US, and China")

        assert isinstance(result, ResearchPlan)
        assert len(result.sub_questions) == 3
        mock_llm_manager.ainvoke_structured.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_plan_enforces_max_sub_questions(
        self, mock_llm_manager, sample_research_plan
    ):
        """plan() must cap sub-questions at max_sub_questions config value."""
        # Return plan with 3 sub-questions, but config limits to 2
        mock_llm_manager.ainvoke_structured.return_value = sample_research_plan

        config = MagicMock()
        config.max_sub_questions = 2
        config.anthropic_model = "claude-sonnet-4-5"
        config.openai_model = "gpt-4o"

        agent = PlannerAgent(llm_manager=mock_llm_manager, config=config)
        result = await agent.plan("Any query")

        assert len(result.sub_questions) <= 2

    @pytest.mark.asyncio
    async def test_plan_includes_search_terms(
        self, mock_llm_manager, sample_research_plan
    ):
        """Each sub-question should include at least one search term."""
        mock_llm_manager.ainvoke_structured.return_value = sample_research_plan

        agent = PlannerAgent(llm_manager=mock_llm_manager)
        result = await agent.plan("Compare AI regulatory frameworks")

        for sq in result.sub_questions:
            assert isinstance(sq.search_terms, list)
            # Production plans should have search terms; fixture guarantees this
            assert len(sq.search_terms) > 0, f"Sub-question {sq.id} has no search terms"

    @pytest.mark.asyncio
    async def test_plan_sub_questions_have_unique_ids(
        self, mock_llm_manager, sample_research_plan
    ):
        """All sub-question IDs must be unique."""
        mock_llm_manager.ainvoke_structured.return_value = sample_research_plan

        agent = PlannerAgent(llm_manager=mock_llm_manager)
        result = await agent.plan("Any query")

        ids = [sq.id for sq in result.sub_questions]
        assert len(ids) == len(set(ids)), "Duplicate sub-question IDs found"

    @pytest.mark.asyncio
    async def test_plan_propagates_llm_error(self, mock_llm_manager):
        """plan() should propagate RuntimeError when the LLM fails."""
        mock_llm_manager.ainvoke_structured.side_effect = RuntimeError("LLM unavailable")

        agent = PlannerAgent(llm_manager=mock_llm_manager)

        with pytest.raises(RuntimeError, match="LLM unavailable"):
            await agent.plan("Any query")

    @pytest.mark.asyncio
    async def test_plan_sends_correct_message_structure(
        self, mock_llm_manager, sample_research_plan
    ):
        """plan() should invoke the LLM with SystemMessage + HumanMessage."""
        from langchain_core.messages import HumanMessage, SystemMessage

        mock_llm_manager.ainvoke_structured.return_value = sample_research_plan

        agent = PlannerAgent(llm_manager=mock_llm_manager)
        await agent.plan("Test query")

        call_args = mock_llm_manager.ainvoke_structured.await_args
        messages = call_args[0][0]  # First positional arg
        schema = call_args[0][1]    # Second positional arg

        assert any(isinstance(m, SystemMessage) for m in messages)
        assert any(isinstance(m, HumanMessage) for m in messages)
        assert schema is ResearchPlan


class TestPlannerNode:
    """Integration tests for the planner_node LangGraph interface."""

    @pytest.mark.asyncio
    async def test_planner_node_returns_state_delta(
        self, sample_query, sample_research_plan
    ):
        """planner_node should return a valid state delta dict."""
        state = create_initial_state(sample_query)

        with patch("src.agents.planner.PlannerAgent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.plan = AsyncMock(return_value=sample_research_plan)
            MockAgent.return_value = mock_agent_instance

            result = await planner_node(state)

        assert "sub_questions" in result
        assert "research_plan" in result
        assert "agent_statuses" in result
        assert "thinking_steps" in result
        assert "errors" in result

    @pytest.mark.asyncio
    async def test_planner_node_marks_status_done_on_success(
        self, sample_query, sample_research_plan
    ):
        """On success, agent_statuses should mark planner as 'done'."""
        state = create_initial_state(sample_query)

        with patch("src.agents.planner.PlannerAgent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.plan = AsyncMock(return_value=sample_research_plan)
            MockAgent.return_value = mock_agent_instance

            result = await planner_node(state)

        assert result["agent_statuses"].get("planner") == "done"
        assert len(result["sub_questions"]) == 3

    @pytest.mark.asyncio
    async def test_planner_node_marks_status_error_on_failure(self, sample_query):
        """On failure, agent_statuses should mark planner as 'error' and record errors."""
        state = create_initial_state(sample_query)

        with patch("src.agents.planner.PlannerAgent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.plan = AsyncMock(side_effect=RuntimeError("LLM error"))
            MockAgent.return_value = mock_agent_instance

            result = await planner_node(state)

        assert result["agent_statuses"].get("planner") == "error"
        assert len(result["errors"]) > 0
        assert "Planner failed" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_planner_node_populates_thinking_steps(
        self, sample_query, sample_research_plan
    ):
        """planner_node should produce at least two thinking steps."""
        state = create_initial_state(sample_query)

        with patch("src.agents.planner.PlannerAgent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.plan = AsyncMock(return_value=sample_research_plan)
            MockAgent.return_value = mock_agent_instance

            result = await planner_node(state)

        assert len(result["thinking_steps"]) >= 2
        for step in result["thinking_steps"]:
            assert step.agent == "planner"
