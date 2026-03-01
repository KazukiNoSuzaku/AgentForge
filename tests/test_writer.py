"""
Tests for the Writer agent.

Covers initial draft generation, revision mode, citation injection,
bibliography formatting, and the LangGraph node interface.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.writer import WriterAgent, _build_citation_reference, writer_node
from src.state import create_initial_state


class TestBuildCitationReference:
    """Unit tests for the citation reference builder."""

    def test_returns_numbered_list(self, sample_findings):
        text = _build_citation_reference(sample_findings)
        assert "[1]" in text
        assert "[2]" in text

    def test_returns_placeholder_when_no_findings(self):
        text = _build_citation_reference([])
        assert "No sources" in text

    def test_includes_source_type(self, sample_findings):
        text = _build_citation_reference(sample_findings)
        # Should mention at least one source type
        assert any(t in text for t in ["brave", "arxiv", "wikipedia"])

    def test_deduplicates_same_url(self, sample_findings):
        # Add a duplicate
        dup = sample_findings[0].model_copy()
        findings_with_dup = sample_findings + [dup]
        text = _build_citation_reference(findings_with_dup)
        # Counting [N] references: should not have [4] since there are only 3 unique URLs
        assert "[4]" not in text


class TestWriterAgent:
    """Unit tests for WriterAgent.write()."""

    @pytest.mark.asyncio
    async def test_write_initial_draft(
        self,
        mock_llm_manager,
        sample_query,
        sample_sub_questions,
        sample_findings,
        sample_analysis,
    ):
        """write() without feedback should return a non-empty markdown string."""
        mock_llm_manager.ainvoke.return_value = "# Research Report\n\nSome content [1][2]."

        agent = WriterAgent(llm_manager=mock_llm_manager)
        result = await agent.write(
            query=sample_query,
            sub_questions=sample_sub_questions,
            findings=sample_findings,
            analysis=sample_analysis,
        )

        assert isinstance(result, str)
        assert len(result) > 0
        assert "## Bibliography" in result  # Should be injected

    @pytest.mark.asyncio
    async def test_write_revision_includes_feedback_in_prompt(
        self,
        mock_llm_manager,
        sample_query,
        sample_sub_questions,
        sample_findings,
        sample_analysis,
        sample_draft_report,
        sample_critic_feedback_fail,
    ):
        """In revision mode, the LLM prompt must contain the Critic's feedback."""
        mock_llm_manager.ainvoke.return_value = "# Revised Report\n\nRevised content [1]."

        agent = WriterAgent(llm_manager=mock_llm_manager)
        await agent.write(
            query=sample_query,
            sub_questions=sample_sub_questions,
            findings=sample_findings,
            analysis=sample_analysis,
            critic_feedback=sample_critic_feedback_fail,
            current_draft=sample_draft_report,
            revision_number=1,
        )

        # Check the HumanMessage contains revision feedback
        call_args = mock_llm_manager.ainvoke.await_args
        messages = call_args[0][0]
        human_content = next(
            m.content for m in messages if hasattr(m, "content") and "Critic Feedback" in m.content
        )
        assert "unsupported_claims" in human_content.lower() or "Unsupported Claims" in human_content

    @pytest.mark.asyncio
    async def test_write_always_injects_bibliography(
        self,
        mock_llm_manager,
        sample_query,
        sample_sub_questions,
        sample_findings,
        sample_analysis,
    ):
        """write() must always append a bibliography section."""
        mock_llm_manager.ainvoke.return_value = "# Report Without Bibliography\n\nContent [1]."

        agent = WriterAgent(llm_manager=mock_llm_manager)
        result = await agent.write(
            query=sample_query,
            sub_questions=sample_sub_questions,
            findings=sample_findings,
            analysis=sample_analysis,
        )

        assert "## Bibliography" in result

    @pytest.mark.asyncio
    async def test_write_uses_revision_system_prompt_when_feedback_provided(
        self,
        mock_llm_manager,
        sample_query,
        sample_sub_questions,
        sample_findings,
        sample_analysis,
        sample_draft_report,
        sample_critic_feedback_fail,
    ):
        """Revision mode should use WRITER_REVISION_PROMPT, not WRITER_SYSTEM_PROMPT."""
        from src.agents.writer import WRITER_REVISION_PROMPT, WRITER_SYSTEM_PROMPT
        from langchain_core.messages import SystemMessage

        mock_llm_manager.ainvoke.return_value = "# Revised Report\n\nContent."

        agent = WriterAgent(llm_manager=mock_llm_manager)
        await agent.write(
            query=sample_query,
            sub_questions=sample_sub_questions,
            findings=sample_findings,
            analysis=sample_analysis,
            critic_feedback=sample_critic_feedback_fail,
            current_draft=sample_draft_report,
            revision_number=1,
        )

        messages = mock_llm_manager.ainvoke.await_args[0][0]
        system_msg = next(m for m in messages if isinstance(m, SystemMessage))
        assert WRITER_REVISION_PROMPT in system_msg.content


class TestWriterNode:
    """Integration tests for writer_node."""

    @pytest.mark.asyncio
    async def test_writer_node_success(self, sample_state, sample_draft_report):
        """writer_node should return draft_report and status 'done' on success."""
        with patch("src.agents.writer.WriterAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.write = AsyncMock(return_value=sample_draft_report)
            MockAgent.return_value = mock_instance

            result = await writer_node(sample_state)

        assert result["draft_report"] == sample_draft_report
        assert result["agent_statuses"]["writer"] == "done"
        assert result["errors"] == []

    @pytest.mark.asyncio
    async def test_writer_node_increments_revision_count_on_revision(
        self, sample_state, sample_draft_report, sample_critic_feedback_fail
    ):
        """revision_count should increment when running a revision."""
        sample_state["critic_feedback"] = sample_critic_feedback_fail
        sample_state["revision_count"] = 1

        with patch("src.agents.writer.WriterAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.write = AsyncMock(return_value=sample_draft_report)
            MockAgent.return_value = mock_instance

            result = await writer_node(sample_state)

        assert result["revision_count"] == 2

    @pytest.mark.asyncio
    async def test_writer_node_error_when_analysis_missing(self, sample_query):
        """writer_node should return error state when analysis is None."""
        state = create_initial_state(sample_query)
        state["analysis"] = None

        result = await writer_node(state)

        assert result["agent_statuses"]["writer"] == "error"
        assert any("analysis is missing" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_writer_node_preserves_existing_draft_on_failure(
        self, sample_state, sample_draft_report
    ):
        """On failure, writer_node should preserve the existing draft."""
        sample_state["draft_report"] = sample_draft_report

        with patch("src.agents.writer.WriterAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.write = AsyncMock(side_effect=RuntimeError("LLM error"))
            MockAgent.return_value = mock_instance

            result = await writer_node(sample_state)

        # Existing draft should be preserved
        assert result["draft_report"] == sample_draft_report
        assert result["agent_statuses"]["writer"] == "error"
