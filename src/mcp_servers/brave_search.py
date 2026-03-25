"""
MCP Server: Brave Search

Exposes Brave Search API as MCP tools that any MCP-compatible client
(including the AgentForge Researcher agent) can call via stdio transport.

Run standalone for testing:
    python src/mcp_servers/brave_search.py

Or connect via MCP client:
    mcp = StdioServerParameters(command="python", args=["src/mcp_servers/brave_search.py"])
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load .env so this server can be run as a standalone process
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server Initialisation
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="BraveSearch",
    instructions=(
        "Use search_web to retrieve current web results for any query. "
        "Prefer specific, targeted queries over broad ones for better relevance."
    ),
)

BRAVE_API_BASE = "https://api.search.brave.com/res/v1"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def search_web(
    query: str,
    count: int = 10,
    country: str = "US",
    search_lang: str = "en",
    freshness: Optional[str] = None,
) -> str:
    """
    Search the web using the Brave Search API.

    Returns JSON-encoded list of results, each containing:
    title, url, description, published_date, extra_snippets.

    Args:
        query: Search query string.
        count: Number of results to return (1-20).
        country: Two-letter country code for regional results.
        search_lang: Language for results (e.g. 'en', 'de').
        freshness: Optional date filter — 'pd' (past day), 'pw' (past week),
                   'pm' (past month), 'py' (past year).

    Returns:
        JSON string of search results.
    """
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise ValueError(
            "BRAVE_SEARCH_API_KEY is not set. Get a free key at https://brave.com/search/api/"
        )

    params: Dict[str, Any] = {
        "q": query,
        "count": min(max(count, 1), 20),
        "country": country,
        "search_lang": search_lang,
        "text_decorations": False,
        "spellcheck": True,
    }
    if freshness:
        params["freshness"] = freshness

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{BRAVE_API_BASE}/web/search",
            params=params,
            headers=headers,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return json.dumps({"error": f"Brave Search API returned {exc.response.status_code}"})
        data = response.json()

    results: List[Dict[str, Any]] = []
    web_results = data.get("web", {}).get("results", [])

    for item in web_results:
        result = {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "description": item.get("description", ""),
            "published_date": item.get("age", ""),
            "extra_snippets": item.get("extra_snippets", []),
            "language": item.get("language", ""),
        }
        results.append(result)

    logger.info("Brave Search: '%s' → %d results", query, len(results))
    return json.dumps(results, indent=2, ensure_ascii=False)


@mcp.tool()
async def search_news(
    query: str,
    count: int = 10,
    country: str = "US",
    freshness: str = "pm",
) -> str:
    """
    Search for recent news articles using the Brave News Search API.

    Args:
        query: News search query.
        count: Number of articles to return (1-20).
        country: Two-letter country code.
        freshness: Date filter — 'pd', 'pw', 'pm', 'py'.

    Returns:
        JSON string of news articles.
    """
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise ValueError("BRAVE_SEARCH_API_KEY is not set.")

    params = {
        "q": query,
        "count": min(max(count, 1), 20),
        "country": country,
        "freshness": freshness,
    }
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{BRAVE_API_BASE}/news/search",
            params=params,
            headers=headers,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return json.dumps({"error": f"Brave News API returned {exc.response.status_code}"})
        data = response.json()

    articles: List[Dict[str, Any]] = []
    for item in data.get("results", []):
        articles.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("description", ""),
                "published_date": item.get("age", ""),
                "source": (item.get("meta_url") or {}).get("netloc", ""),
            }
        )

    logger.info("Brave News Search: '%s' → %d articles", query, len(articles))
    return json.dumps(articles, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    mcp.run()
