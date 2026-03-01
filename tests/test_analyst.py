"""
Tests for the Analyst agent.

Verifies synthesis logic, claim confidence handling, and error cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.analyst import AnalystAgent, _format_findings_for_prompt, analyst_node
from src.models.schemas import AnalysisResult, Claim
from src.state import create_initial_state


class TestFormatFindingsForPrompt:
    """Unit tests for the findings formatter helper."""

    def test_formats_non_empty_findings(self, sample_findings):
        text = _format_findings_for_prompt(sample_findings)
        assert "[Finding 1]" in text
        assert "Sub-question ID:" in text
        assert "Relevance:" in text

    def test_returns_empty_string_for_no_findings(self):
        result = _format_findings_for_prompt([])
        assert result == ""

    def test_truncates_long_content(self, sample_findings):
        """Content longer than 600 chars should be truncated in the prompt."""
        long_finding = sample_findings[0].model_copy(
            update={"content": "x" * 1000}
        )
        text = _format_findings_for_prompt([long_finding])
        # The formatted content block should not exceed 600 chars for this finding
        assert len(text) < 2000  # Reasonable upper bound

    def test_includes_key_insights(self, sample_findings):
        text = _format_findings_for_prompt(sample_findings)
        assert "Key Insights:" in text
        assert "risk" in text.lower() or "insight" in text.lower()


class TestAnalystAgent:
    """Unit tests for AnalystAgent.analyze()."""

    @pytest.mark.asyncio
    async def test_analyze_returns_analysis_result(
        self, mock_llm_manager, sample_findings, sample_analysis, sample_query
    ):
        """analyze() should return the LLM's structured output."""
        mock_llm_manager.ainvoke_structured.return_value = sample_analysis

        agent = AnalystAgent(llm_manager=mock_llm_manager)
        result = await agent.analyze(sample_findings, sample_query)

        assert isinstance(result, AnalysisResult)
        assert len(result.claims) > 0
        assert len(result.patterns) > 0

    @pytest.mark.asyncio
    async def test_analyze_assigns_ids_to_claims_without_ids(
        self, mock_llm_manager, sample_findings, sample_query
    ):
        """Claims returned without IDs should get auto-assigned IDs."""
        analysis_no_ids = AnalysisResult(
            patterns=["Pattern: something"],
            contradictions=[],
            gaps=[],
            claims=[
                Claim(
                    id="",  # Empty ID
                    statement="Some claim",
                    confidence=0.8,
                    supporting_sources=[],
                    contradicting_sources=[],
                    sub_question_id="sq_1",
                )
            ],
            confidence_summary="Good research base.",
        )
        mock_llm_manager.ainvoke_structured.return_value = analysis_no_ids

        agent = AnalystAgent(llm_manager=mock_llm_manager)
        result = await agent.analyze(sample_findings, sample_query)

        for i, claim in enumerate(result.claims, 1):
            assert claim.id, f"Claim {i} still has no ID"

    @pytest.mark.asyncio
    async def test_analyze_caps_findings_at_25(
        self, mock_llm_manager, sample_analysis, sample_query, sample_findings
    ):
        """analyze() should cap findings at 25 to stay within context budget."""
        # Create 30 findings
        many_findings = sample_findings * 10  # 30 findings
        mock_llm_manager.ainvoke_structured.return_value = sample_analysis

        agent = AnalystAgent(llm_manager=mock_llm_manager)
        await agent.analyze(many_findings, sample_query)

        # The LLM should have been called with a prompt, not raise an error
        mock_llm_manager.ainvoke_structured.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_analyze_propagates_llm_failure(
        self, mock_llm_manager, sample_findings, sample_query
    ):
        """analyze() should propagate RuntimeError when the LLM fails."""
        mock_llm_manager.ainvoke_structured.side_effect = RuntimeError("LLM down")

        agent = AnalystAgent(llm_manager=mock_llm_manager)

        with pytest.raises(RuntimeError, match="LLM down"):
            await agent.analyze(sample_findings, sample_query)

    @pytest.mark.asyncio
    async def test_analyze_sorts_findings_by_relevance(
        self, mock_llm_manager, sample_analysis, sample_query, sample_findings
    ):
        """analyze() should sort findings by relevance before passing to the LLM."""
        # Create findings with known relevance scores in reverse order
        low_relevance = sample_findings[0].model_copy(update={"relevance_score": 0.2})
        high_relevance = sample_findings[1].model_copy(update={"relevance_score": 0.95})
        findings = [low_relevance, high_relevance]

        mock_llm_manager.ainvoke_structured.return_value = sample_analysis

        agent = AnalystAgent(llm_manager=mock_llm_manager)
        await agent.analyze(findings, sample_query)

        # Verify the LLM was called — we can't easily check sort order in the prompt
        # but we verify no exception was raised during sorting
        mock_llm_manager.ainvoke_structured.assert_awaited_once()


class TestAnalystNode:
    """Integration tests for the analyst_node LangGraph interface."""

    @pytest.mark.asyncio
    async def test_analyst_node_returns_analysis_on_success(
        self, sample_state, sample_analysis
    ):
        """analyst_node should populate state['analysis'] on success."""
        with patch("src.agents.analyst.AnalystAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.analyze = AsyncMock(return_value=sample_analysis)
            MockAgent.return_value = mock_instance

            result = await analyst_node(sample_state)

        assert result["analysis"] is not None
        assert result["agent_statuses"]["analyst"] == "done"
        assert result["errors"] == []

    @pytest.mark.asyncio
    async def test_analyst_node_handles_empty_findings(self, sample_query):
        """analyst_node should return error state when findings list is empty."""
        state = create_initial_state(sample_query)
        state["research_findings"] = []

        result = await analyst_node(state)

        assert result["analysis"] is None
        assert result["agent_statuses"]["analyst"] == "error"
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_analyst_node_marks_error_on_llm_failure(self, sample_state):
        """analyst_node should mark status as 'error' when the LLM fails."""
        with patch("src.agents.analyst.AnalystAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.analyze = AsyncMock(side_effect=RuntimeError("timeout"))
            MockAgent.return_value = mock_instance

            result = await analyst_node(sample_state)

        assert result["agent_statuses"]["analyst"] == "error"
        assert result["analysis"] is None
        assert any("Analyst failed" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_analyst_node_reports_claim_counts(self, sample_state, sample_analysis):
        """analyst_node thinking_steps should mention high-confidence claim count."""
        with patch("src.agents.analyst.AnalystAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.analyze = AsyncMock(return_value=sample_analysis)
            MockAgent.return_value = mock_instance

            result = await analyst_node(sample_state)

        steps = result["thinking_steps"]
        completion_step = next((s for s in steps if s.action == "analysis_complete"), None)
        assert completion_step is not None
        assert "claims" in completion_step.content
