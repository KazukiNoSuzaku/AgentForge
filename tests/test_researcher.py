"""
Tests for the Researcher agent.

Covers MCP tool invocation, parallel search, error tolerance,
result evaluation, and the LangGraph node interface.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.researcher import (
    ResearcherAgent,
    call_mcp_tool,
    evaluate_raw_results,
    parse_json_safely,
    researcher_node,
)
from src.models.schemas import SourceType
from src.state import create_initial_state

# ---------------------------------------------------------------------------
# Unit Tests: parse_json_safely
# ---------------------------------------------------------------------------


class TestParseJsonSafely:
    def test_parses_valid_list(self):
        raw = json.dumps([{"title": "Article 1"}, {"title": "Article 2"}])
        result = parse_json_safely(raw)
        assert len(result) == 2
        assert result[0]["title"] == "Article 1"

    def test_returns_empty_list_on_none(self):
        assert parse_json_safely(None) == []

    def test_returns_empty_list_on_invalid_json(self):
        assert parse_json_safely("not valid json {") == []

    def test_returns_empty_list_when_json_is_not_list(self):
        assert parse_json_safely('{"key": "value"}') == []

    def test_returns_empty_list_on_empty_string(self):
        assert parse_json_safely("") == []


# ---------------------------------------------------------------------------
# Unit Tests: call_mcp_tool
# ---------------------------------------------------------------------------


class TestCallMcpTool:
    @pytest.mark.asyncio
    async def test_returns_none_on_timeout(self, tmp_path):
        """call_mcp_tool should return None on timeout without raising."""
        server_script = tmp_path / "dummy_server.py"
        server_script.write_text("# dummy")

        # Patch stdio_client to raise TimeoutError
        with patch("src.agents.researcher.stdio_client") as mock_ctx:
            mock_ctx.return_value.__aenter__.side_effect = TimeoutError()

            result = await call_mcp_tool(server_script, "any_tool", {}, timeout=0.001)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self, tmp_path):
        """call_mcp_tool should return None (not raise) when the server fails."""
        server_script = tmp_path / "dummy_server.py"
        server_script.write_text("# dummy")

        with patch("src.agents.researcher.stdio_client") as mock_ctx:
            mock_ctx.return_value.__aenter__.side_effect = Exception("Connection refused")

            result = await call_mcp_tool(server_script, "any_tool", {})

        assert result is None


# ---------------------------------------------------------------------------
# Unit Tests: evaluate_raw_results
# ---------------------------------------------------------------------------


class TestEvaluateRawResults:
    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_raw_results(
        self, mock_llm_manager, sample_sub_questions
    ):
        """Should return [] without calling the LLM if there are no results."""
        sq = sample_sub_questions[0]
        result = await evaluate_raw_results(sq, [], mock_llm_manager)

        assert result == []
        mock_llm_manager.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_filters_low_relevance_findings(self, mock_llm_manager, sample_sub_questions):
        """Findings with relevance_score < 0.3 should be filtered out."""
        sq = sample_sub_questions[0]

        # LLM returns one high-relevance and one low-relevance finding
        mock_response = json.dumps(
            [
                {
                    "sub_question_id": sq.id,
                    "sub_question": sq.question,
                    "source_type": "brave",
                    "title": "Relevant article",
                    "url": "https://example.com/relevant",
                    "content": "Very relevant content",
                    "relevance_score": 0.85,
                    "key_insights": ["Key insight"],
                    "metadata": {},
                },
                {
                    "sub_question_id": sq.id,
                    "sub_question": sq.question,
                    "source_type": "brave",
                    "title": "Irrelevant article",
                    "url": "https://example.com/irrelevant",
                    "content": "Barely related",
                    "relevance_score": 0.15,
                    "key_insights": [],
                    "metadata": {},
                },
            ]
        )
        mock_llm_manager.ainvoke.return_value = mock_response

        raw_results = [
            {
                "_source_type": "brave",
                "title": "Relevant",
                "url": "https://example.com/relevant",
                "description": "content",
            }
        ]

        findings = await evaluate_raw_results(sq, raw_results, mock_llm_manager)

        # Only the high-relevance finding should survive
        assert len(findings) == 1
        assert findings[0].relevance_score == 0.85

    @pytest.mark.asyncio
    async def test_handles_malformed_llm_json_gracefully(
        self, mock_llm_manager, sample_sub_questions
    ):
        """Should return [] when the LLM returns invalid JSON."""
        sq = sample_sub_questions[0]
        mock_llm_manager.ainvoke.return_value = "This is not JSON at all"

        raw_results = [{"_source_type": "brave", "title": "Test", "url": "https://test.com"}]
        findings = await evaluate_raw_results(sq, raw_results, mock_llm_manager)

        assert findings == []

    @pytest.mark.asyncio
    async def test_handles_markdown_code_block_in_response(
        self, mock_llm_manager, sample_sub_questions
    ):
        """Should strip ```json code blocks from the LLM response."""
        sq = sample_sub_questions[0]
        json_content = json.dumps(
            [
                {
                    "sub_question_id": sq.id,
                    "sub_question": sq.question,
                    "source_type": "wikipedia",
                    "title": "Test",
                    "url": "https://en.wikipedia.org/wiki/Test",
                    "content": "Test content",
                    "relevance_score": 0.75,
                    "key_insights": [],
                    "metadata": {},
                }
            ]
        )
        mock_llm_manager.ainvoke.return_value = f"```json\n{json_content}\n```"

        raw_results = [
            {
                "_source_type": "wikipedia",
                "title": "Test",
                "url": "https://en.wikipedia.org/wiki/Test",
            }
        ]
        findings = await evaluate_raw_results(sq, raw_results, mock_llm_manager)

        assert len(findings) == 1
        assert findings[0].source_type == SourceType.WIKIPEDIA


# ---------------------------------------------------------------------------
# Unit Tests: ResearcherAgent
# ---------------------------------------------------------------------------


class TestResearcherAgent:
    @pytest.mark.asyncio
    async def test_research_returns_tuple_of_findings_and_errors(
        self, sample_sub_questions, mock_llm_manager
    ):
        """research() should return (findings, errors) regardless of search failures."""
        sq = sample_sub_questions[0]

        with (
            patch("src.agents.researcher.search_brave", return_value=[]),
            patch("src.agents.researcher.search_arxiv", return_value=[]),
            patch("src.agents.researcher.search_wikipedia", return_value=[]),
            patch("src.agents.researcher.evaluate_raw_results", return_value=[]),
        ):
            agent = ResearcherAgent(llm_manager=mock_llm_manager)
            findings, errors = await agent.research(sq)

        assert isinstance(findings, list)
        assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_research_tolerates_individual_source_failure(
        self, sample_sub_questions, mock_llm_manager, sample_findings
    ):
        """research() continues even if one search source raises an exception."""
        sq = sample_sub_questions[0]

        # Brave fails, arXiv succeeds, Wikipedia succeeds
        with (
            patch("src.agents.researcher.search_brave", side_effect=Exception("API error")),
            patch(
                "src.agents.researcher.search_arxiv",
                return_value=[
                    {"_source_type": "arxiv", "title": "Paper", "url": "http://arxiv.org/1"}
                ],
            ),
            patch("src.agents.researcher.search_wikipedia", return_value=[]),
            patch("src.agents.researcher.evaluate_raw_results", return_value=[sample_findings[1]]),
        ):
            agent = ResearcherAgent(llm_manager=mock_llm_manager)
            findings, errors = await agent.research(sq)

        # Should report Brave error but still return findings from arXiv
        assert any("Brave" in e for e in errors)
        assert len(findings) > 0


# ---------------------------------------------------------------------------
# Integration Tests: researcher_node
# ---------------------------------------------------------------------------


class TestResearcherNode:
    @pytest.mark.asyncio
    async def test_researcher_node_returns_findings_on_success(self, sample_state, sample_findings):
        """researcher_node should accumulate findings from all sub-questions."""
        with patch("src.agents.researcher.ResearcherAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.research = AsyncMock(return_value=(sample_findings[:1], []))
            MockAgent.return_value = mock_instance

            result = await researcher_node(sample_state)

        assert "research_findings" in result
        assert len(result["research_findings"]) > 0
        assert result["agent_statuses"]["researcher"] == "done"

    @pytest.mark.asyncio
    async def test_researcher_node_handles_empty_sub_questions(self, sample_query):
        """researcher_node returns an error state if sub_questions is empty."""
        state = create_initial_state(sample_query)
        state["sub_questions"] = []

        result = await researcher_node(state)

        assert result["agent_statuses"]["researcher"] == "error"
        assert any("no sub-questions" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_researcher_node_runs_parallel_tasks(self, sample_state, sample_findings):
        """researcher_node should call research() once per sub-question."""
        with patch("src.agents.researcher.ResearcherAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.research = AsyncMock(return_value=([], []))
            MockAgent.return_value = mock_instance

            await researcher_node(sample_state)

        # research() should have been called for each sub-question
        assert mock_instance.research.await_count == len(sample_state["sub_questions"])
