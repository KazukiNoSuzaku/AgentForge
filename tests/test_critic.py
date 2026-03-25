"""
Tests for the Critic agent.

Covers quality scoring, revision threshold logic, max-loop enforcement,
auto citation validation, and the LangGraph node interface.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.critic import CriticAgent, critic_node
from src.models.schemas import CriticFeedback
from src.state import create_initial_state


class TestCriticAgent:
    """Unit tests for CriticAgent.critique()."""

    @pytest.mark.asyncio
    async def test_critique_returns_feedback(
        self,
        mock_llm_manager,
        sample_draft_report,
        sample_findings,
        sample_sub_questions,
        sample_analysis,
        sample_critic_feedback_pass,
    ):
        """critique() should return a CriticFeedback instance."""
        mock_llm_manager.ainvoke_structured.return_value = sample_critic_feedback_pass

        agent = CriticAgent(llm_manager=mock_llm_manager)
        result = await agent.critique(
            report=sample_draft_report,
            findings=sample_findings,
            sub_questions=sample_sub_questions,
            analysis=sample_analysis,
            revision_count=0,
        )

        assert isinstance(result, CriticFeedback)
        assert 0.0 <= result.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_critique_forces_approval_at_max_loops(
        self,
        mock_llm_manager,
        sample_draft_report,
        sample_findings,
        sample_sub_questions,
        sample_analysis,
        sample_critic_feedback_fail,
    ):
        """When revision_count >= max_revision_loops, requires_revision must be False."""
        mock_llm_manager.ainvoke_structured.return_value = sample_critic_feedback_fail

        config = MagicMock()
        config.quality_threshold = 0.75
        config.max_revision_loops = 2

        agent = CriticAgent(llm_manager=mock_llm_manager, config=config)

        # At max loops, even a failing score should not trigger revision
        result = await agent.critique(
            report=sample_draft_report,
            findings=sample_findings,
            sub_questions=sample_sub_questions,
            analysis=sample_analysis,
            revision_count=2,  # At max
        )

        assert result.requires_revision is False

    @pytest.mark.asyncio
    async def test_critique_allows_revision_below_max_loops(
        self,
        mock_llm_manager,
        sample_draft_report,
        sample_findings,
        sample_sub_questions,
        sample_analysis,
        sample_critic_feedback_fail,
    ):
        """When revision_count < max_revision_loops, requires_revision should follow LLM output."""
        mock_llm_manager.ainvoke_structured.return_value = sample_critic_feedback_fail

        config = MagicMock()
        config.quality_threshold = 0.75
        config.max_revision_loops = 2

        agent = CriticAgent(llm_manager=mock_llm_manager, config=config)
        result = await agent.critique(
            report=sample_draft_report,
            findings=sample_findings,
            sub_questions=sample_sub_questions,
            analysis=sample_analysis,
            revision_count=0,  # Below max
        )

        # The LLM said requires_revision=True, and we're below the cap
        assert result.requires_revision is True

    @pytest.mark.asyncio
    async def test_critique_appends_auto_detected_citation_issues(
        self,
        mock_llm_manager,
        sample_findings,
        sample_sub_questions,
        sample_analysis,
        sample_critic_feedback_pass,
    ):
        """Auto-detected citation issues not caught by LLM should be added to feedback."""
        # Report references [99] which doesn't exist in findings
        bad_report = "This claim is supported [99]. Another claim [1]."
        mock_llm_manager.ainvoke_structured.return_value = sample_critic_feedback_pass

        config = MagicMock()
        config.quality_threshold = 0.75
        config.max_revision_loops = 2

        agent = CriticAgent(llm_manager=mock_llm_manager, config=config)
        result = await agent.critique(
            report=bad_report,
            findings=sample_findings,
            sub_questions=sample_sub_questions,
            analysis=sample_analysis,
            revision_count=0,
        )

        # Citation [99] should be flagged
        assert any("99" in issue for issue in result.citation_issues)

    @pytest.mark.asyncio
    async def test_critique_includes_strengths(
        self,
        mock_llm_manager,
        sample_draft_report,
        sample_findings,
        sample_sub_questions,
        sample_analysis,
        sample_critic_feedback_pass,
    ):
        """Feedback should always contain at least one strength."""
        mock_llm_manager.ainvoke_structured.return_value = sample_critic_feedback_pass

        config = MagicMock()
        config.quality_threshold = 0.75
        config.max_revision_loops = 2

        agent = CriticAgent(llm_manager=mock_llm_manager, config=config)
        result = await agent.critique(
            report=sample_draft_report,
            findings=sample_findings,
            sub_questions=sample_sub_questions,
            analysis=sample_analysis,
            revision_count=0,
        )

        assert len(result.strengths) >= 1


class TestCriticNode:
    """Integration tests for critic_node."""

    @pytest.mark.asyncio
    async def test_critic_node_approves_quality_report(
        self, sample_state, sample_critic_feedback_pass
    ):
        """On approval, final_report should be populated and requires_revision=False."""
        with patch("src.agents.critic.CriticAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.critique = AsyncMock(return_value=sample_critic_feedback_pass)
            MockAgent.return_value = mock_instance

            result = await critic_node(sample_state)

        assert result["final_report"] == sample_state["draft_report"]
        assert result["critic_feedback"].requires_revision is False
        assert result["agent_statuses"]["critic"] == "done"

    @pytest.mark.asyncio
    async def test_critic_node_triggers_revision(self, sample_state, sample_critic_feedback_fail):
        """When requires_revision=True, final_report should be empty."""
        with patch("src.agents.critic.CriticAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.critique = AsyncMock(return_value=sample_critic_feedback_fail)
            MockAgent.return_value = mock_instance

            result = await critic_node(sample_state)

        assert result["critic_feedback"].requires_revision is True
        assert result["final_report"] == ""  # Not approved yet

    @pytest.mark.asyncio
    async def test_critic_node_approves_on_failure(self, sample_state):
        """If the Critic itself fails, the draft is approved to avoid blocking."""
        with patch("src.agents.critic.CriticAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.critique = AsyncMock(side_effect=RuntimeError("API timeout"))
            MockAgent.return_value = mock_instance

            result = await critic_node(sample_state)

        # Draft should be set as final (fail-open to prevent pipeline deadlock)
        assert result["final_report"] == sample_state["draft_report"]
        assert result["agent_statuses"]["critic"] == "error"
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_critic_node_handles_missing_draft(self, sample_query):
        """critic_node should return error state when draft_report is empty."""
        state = create_initial_state(sample_query)
        state["draft_report"] = ""

        result = await critic_node(state)

        assert result["agent_statuses"]["critic"] == "error"
        assert any("no draft" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_critic_node_thinking_steps_mention_quality_score(
        self, sample_state, sample_critic_feedback_pass
    ):
        """The completion thinking step should include the quality score."""
        with patch("src.agents.critic.CriticAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.critique = AsyncMock(return_value=sample_critic_feedback_pass)
            MockAgent.return_value = mock_instance

            result = await critic_node(sample_state)

        done_step = next(
            (s for s in result["thinking_steps"] if s.action == "review_complete"), None
        )
        assert done_step is not None
        assert "0.88" in done_step.content or "score" in done_step.content.lower()
