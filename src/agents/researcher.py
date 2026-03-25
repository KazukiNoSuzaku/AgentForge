"""
Researcher Agent

For each sub-question, runs parallel searches across Brave Search, arXiv,
and Wikipedia via MCP tool servers. Uses an LLM to evaluate and annotate
the raw results, extracting relevant passages with confidence scores.

Architecture note: MCP servers are invoked as stdio subprocesses. Each call
creates a short-lived connection; this is intentional to keep the process
model simple while remaining MCP-spec compliant.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.config import AppConfig, get_config
from src.models.schemas import (
    ResearchFinding,
    SourceType,
    SubQuestion,
    ThinkingStep,
)
from src.state import ResearchState
from src.utils.llm import LLMManager, get_llm_manager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

RESEARCHER_SYSTEM_PROMPT = """You are the Researcher agent for AgentForge.

Your role is to evaluate raw search results and extract structured, citable research findings.

## Input

You will receive:
- A sub-question to research
- Raw results from Brave Search (web), arXiv (academic), and Wikipedia (reference)

## Task

For each search result, assess:
1. **Relevance** (0.0–1.0): How directly does this source answer the sub-question?
2. **Key Insights**: 2-4 bullet-point insights from this source
3. **Content**: A concise summary (2-4 sentences) of what this source says about the sub-question

## Output

Return a JSON list of ResearchFinding objects. Only include sources with relevance_score >= 0.3.
Limit to the 8 most relevant findings total.

For each finding:
{
  "sub_question_id": "sq_1",
  "sub_question": "The full sub-question text",
  "source_type": "brave" | "arxiv" | "wikipedia",
  "title": "Source title",
  "url": "Source URL",
  "content": "Concise summary of relevant content",
  "relevance_score": 0.0-1.0,
  "key_insights": ["insight 1", "insight 2"],
  "metadata": {"authors": [], "published_date": "YYYY-MM-DD"}
}

## Critical Rules

- Do NOT fabricate content — only summarize what is in the provided results
- Flag any contradictions between sources in the key_insights
- Preserve exact URLs — never modify or invent URLs
- If all results are irrelevant (< 0.3), return an empty list with a note in errors
"""

# ---------------------------------------------------------------------------
# MCP Client Utilities
# ---------------------------------------------------------------------------


async def call_mcp_tool(
    server_script: Path,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout: float = 60.0,
) -> Optional[str]:
    """
    Call a tool on an MCP server running as a stdio subprocess.

    Args:
        server_script: Absolute path to the MCP server Python script.
        tool_name: Name of the tool to invoke.
        arguments: Tool arguments as a dict.
        timeout: Seconds before the call times out.

    Returns:
        The tool's text output, or None on failure.
    """
    env = {**os.environ}  # Pass through all env vars (including API keys)

    server_params = StdioServerParameters(
        command=sys.executable,  # Use the same Python interpreter
        args=[str(server_script)],
        env=env,
    )

    try:
        async with asyncio.timeout(timeout):
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)

                    if result.isError:
                        logger.error(
                            "MCP tool '%s' returned error: %s",
                            tool_name,
                            result.content,
                        )
                        return None

                    if result.content:
                        return result.content[0].text
                    return None

    except TimeoutError:
        logger.warning("MCP tool '%s' timed out after %.0fs", tool_name, timeout)
        return None
    except Exception as exc:
        logger.warning("MCP tool '%s' failed: %s", tool_name, exc)
        return None


def parse_json_safely(raw: Optional[str]) -> List[Dict[str, Any]]:
    """
    Parse a JSON string into a list, returning [] on any error.

    Args:
        raw: JSON string (may be None or malformed).

    Returns:
        Parsed list, or empty list on failure.
    """
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        logger.warning("Failed to parse MCP response as JSON: %.200s", raw)
        return []


# ---------------------------------------------------------------------------
# Source-Specific Search Functions
# ---------------------------------------------------------------------------


async def search_brave(
    query: str,
    search_terms: List[str],
    config: AppConfig,
    mcp_servers_dir: Path,
) -> List[Dict[str, Any]]:
    """Run a Brave web search and return raw results."""
    if not config.has_brave_search:
        logger.info("Brave Search skipped — no API key configured")
        return []

    # Combine query with search terms for a more targeted search
    combined_query = query if not search_terms else f"{query} {' '.join(search_terms[:2])}"
    server_script = mcp_servers_dir / "brave_search.py"

    raw = await call_mcp_tool(
        server_script,
        "search_web",
        {"query": combined_query, "count": config.max_search_results},
    )
    results = parse_json_safely(raw)
    # Tag each result with source info
    for r in results:
        r["_source_type"] = "brave"
    return results


async def search_arxiv(
    query: str,
    search_terms: List[str],
    config: AppConfig,
    mcp_servers_dir: Path,
) -> List[Dict[str, Any]]:
    """Run an arXiv academic search and return raw results."""
    combined_query = query if not search_terms else f"{query} {' '.join(search_terms[:2])}"
    server_script = mcp_servers_dir / "arxiv_search.py"

    raw = await call_mcp_tool(
        server_script,
        "search_papers",
        {"query": combined_query, "max_results": min(config.max_search_results, 10)},
    )
    results = parse_json_safely(raw)
    for r in results:
        r["_source_type"] = "arxiv"
        # Normalise fields to common schema
        if "abstract" in r and "description" not in r:
            r["description"] = r["abstract"]
        if "url" not in r and "entry_id" in r:
            r["url"] = r["entry_id"]
    return results


async def search_wikipedia(
    query: str,
    config: AppConfig,
    mcp_servers_dir: Path,
) -> List[Dict[str, Any]]:
    """Search Wikipedia and fetch article summaries."""
    server_script = mcp_servers_dir / "wikipedia.py"

    # First, search for relevant article titles
    search_raw = await call_mcp_tool(
        server_script,
        "search_wikipedia",
        {"query": query, "num_results": 3},
    )
    search_data = json.loads(search_raw) if search_raw else {}
    titles = search_data.get("results", [])

    if not titles:
        return []

    # Fetch the top 2 articles
    results = []
    for title in titles[:2]:
        summary_raw = await call_mcp_tool(
            server_script,
            "get_article_summary",
            {"title": title, "sentences": 8},
        )
        if summary_raw:
            try:
                article = json.loads(summary_raw)
                if "error" not in article:
                    article["_source_type"] = "wikipedia"
                    article["title"] = article.get("title", title)
                    article["description"] = article.get("summary", "")
                    results.append(article)
            except json.JSONDecodeError:
                pass

    return results


# ---------------------------------------------------------------------------
# LLM Evaluation of Raw Results
# ---------------------------------------------------------------------------


async def evaluate_raw_results(
    sub_question: SubQuestion,
    raw_results: List[Dict[str, Any]],
    llm_manager: LLMManager,
) -> List[ResearchFinding]:
    """
    Use the LLM to evaluate raw search results and produce structured findings.

    Args:
        sub_question: The sub-question being researched.
        raw_results: Combined list of raw results from all sources.
        llm_manager: LLM manager instance.

    Returns:
        List of validated ResearchFinding objects.
    """
    if not raw_results:
        logger.warning("No raw results to evaluate for '%s'", sub_question.question)
        return []

    # Truncate results for the prompt (stay within token budget)
    truncated = []
    for r in raw_results[:15]:
        truncated.append(
            {
                "source_type": r.get("_source_type", "unknown"),
                "title": r.get("title", "Untitled")[:200],
                "url": r.get("url", r.get("entry_id", ""))[:300],
                "description": r.get("description", r.get("abstract", r.get("snippet", "")))[:800],
                "authors": r.get("authors", [])[:3],
                "published": r.get("published", r.get("published_date", "")),
            }
        )

    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Sub-question: {sub_question.question}\n"
                f"Sub-question ID: {sub_question.id}\n\n"
                f"Raw Search Results:\n{json.dumps(truncated, indent=2)}\n\n"
                "Evaluate each result and return a JSON list of ResearchFinding objects."
            )
        ),
    ]

    raw_response = await llm_manager.ainvoke(messages)

    # Extract JSON from the response
    findings = []
    try:
        # Handle markdown code blocks
        json_str = raw_response
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        raw_findings = json.loads(json_str)
        if not isinstance(raw_findings, list):
            raw_findings = []

        for f in raw_findings:
            try:
                finding = ResearchFinding(
                    sub_question_id=f.get("sub_question_id", sub_question.id),
                    sub_question=f.get("sub_question", sub_question.question),
                    source_type=SourceType(f.get("source_type", "brave")),
                    title=f.get("title", ""),
                    url=f.get("url", ""),
                    content=f.get("content", ""),
                    relevance_score=float(f.get("relevance_score", 0.5)),
                    key_insights=f.get("key_insights", []),
                    metadata={
                        "authors": f.get("metadata", {}).get("authors", []),
                        "published_date": f.get("metadata", {}).get("published_date", ""),
                    },
                )
                if finding.url and finding.relevance_score >= 0.3:
                    findings.append(finding)
            except Exception as parse_err:
                logger.debug("Skipped malformed finding: %s", parse_err)

    except json.JSONDecodeError as json_err:
        logger.warning(
            "Could not parse LLM findings JSON for '%s': %s",
            sub_question.question,
            json_err,
        )

    logger.info("Researcher: '%s' → %d findings", sub_question.question, len(findings))
    return findings


# ---------------------------------------------------------------------------
# Agent Class
# ---------------------------------------------------------------------------


class ResearcherAgent:
    """
    Searches all configured sources for a sub-question and evaluates the results.

    Runs Brave Search, arXiv, and Wikipedia searches in parallel, then uses
    an LLM to filter and structure the raw results into ResearchFindings.
    """

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self._llm = llm_manager or get_llm_manager()
        self._config = config or get_config()
        self._mcp_dir = self._config.mcp_servers_dir

    async def research(self, sub_question: SubQuestion) -> Tuple[List[ResearchFinding], List[str]]:
        """
        Research a single sub-question across all available sources.

        Args:
            sub_question: The sub-question to research.

        Returns:
            Tuple of (findings, errors). Errors accumulate without stopping execution.
        """
        errors = []
        query = sub_question.question
        search_terms = sub_question.search_terms

        logger.info("Researcher: starting parallel search for '%s'", query)

        # Run all searches in parallel
        brave_task = search_brave(query, search_terms, self._config, self._mcp_dir)
        arxiv_task = search_arxiv(query, search_terms, self._config, self._mcp_dir)
        wiki_task = search_wikipedia(query, self._config, self._mcp_dir)

        results = await asyncio.gather(brave_task, arxiv_task, wiki_task, return_exceptions=True)

        brave_results, arxiv_results, wiki_results = results
        all_raw: List[Dict[str, Any]] = []

        for name, result in [
            ("Brave", brave_results),
            ("arXiv", arxiv_results),
            ("Wikipedia", wiki_results),
        ]:
            if isinstance(result, Exception):
                err = f"[{name}] search failed for '{query}': {result}"
                logger.warning(err)
                errors.append(err)
            elif isinstance(result, list):
                all_raw.extend(result)

        logger.info("Researcher: '%s' — collected %d raw results", query, len(all_raw))

        findings = await evaluate_raw_results(sub_question, all_raw, self._llm)
        return findings, errors


# ---------------------------------------------------------------------------
# LangGraph Node
# ---------------------------------------------------------------------------


async def researcher_node(state: ResearchState) -> dict:
    """
    LangGraph node: research all sub-questions in parallel.

    Args:
        state: Current ResearchState (must contain sub_questions).

    Returns:
        State delta with accumulated research_findings, agent_statuses,
        thinking_steps, and any errors encountered.
    """
    sub_questions = state.get("sub_questions", [])
    if not sub_questions:
        return {
            "research_findings": [],
            "agent_statuses": {**state.get("agent_statuses", {}), "researcher": "error"},
            "thinking_steps": [],
            "errors": ["Researcher: no sub-questions to research (planner may have failed)"],
        }

    agent = ResearcherAgent()

    thinking_steps = [
        ThinkingStep(
            agent="researcher",
            action="starting_research",
            content=(
                f"Beginning parallel research for {len(sub_questions)} sub-questions "
                f"across Brave Search, arXiv, and Wikipedia"
            ),
            timestamp=datetime.now(),
        )
    ]

    # Research all sub-questions in parallel
    tasks = [agent.research(sq) for sq in sub_questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_findings: List[ResearchFinding] = []
    all_errors: List[str] = []

    for i, result in enumerate(results):
        sq = sub_questions[i]
        if isinstance(result, Exception):
            err = f"Researcher: sub-question '{sq.question}' failed: {result}"
            logger.exception("Research task failed: %s", result)
            all_errors.append(err)
        else:
            findings, errors = result
            all_findings.extend(findings)
            all_errors.extend(errors)

    thinking_steps.append(
        ThinkingStep(
            agent="researcher",
            action="research_complete",
            content=(
                f"Collected {len(all_findings)} relevant findings from "
                f"{len(sub_questions)} sub-questions"
            ),
            timestamp=datetime.now(),
        )
    )

    logger.info("Researcher node: %d total findings", len(all_findings))

    return {
        "research_findings": all_findings,
        "agent_statuses": {**state.get("agent_statuses", {}), "researcher": "done"},
        "thinking_steps": thinking_steps,
        "errors": all_errors,
    }
