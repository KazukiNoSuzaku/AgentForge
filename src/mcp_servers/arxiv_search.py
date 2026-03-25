"""
MCP Server: arXiv Academic Paper Search

Exposes arXiv search as MCP tools for retrieving peer-reviewed academic
papers, preprints, and research summaries.

Run standalone for testing:
    python src/mcp_servers/arxiv_search.py
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Optional

import arxiv
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="ArXivSearch",
    instructions=(
        "Use search_papers to find academic research on a topic. "
        "Use get_paper to retrieve full details of a specific paper by ID. "
        "Prefer targeted, domain-specific queries for best results."
    ),
)

# arXiv category codes useful for research routing
CATEGORY_HINTS = {
    "ai": "cs.AI",
    "machine learning": "cs.LG",
    "nlp": "cs.CL",
    "computer vision": "cs.CV",
    "robotics": "cs.RO",
    "economics": "econ",
    "policy": "econ.GN",
    "physics": "physics",
    "biology": "q-bio",
}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def search_papers(
    query: str,
    max_results: int = 5,
    sort_by: str = "relevance",
    category: Optional[str] = None,
) -> str:
    """
    Search arXiv for academic papers matching a query.

    Args:
        query: Search query (supports arXiv query syntax, e.g. 'ti:neural AND abs:regulation').
        max_results: Maximum number of papers to return (1-25).
        sort_by: Sort order — 'relevance', 'recent' (submitted date), or 'updated'.
        category: Optional arXiv category filter (e.g. 'cs.AI', 'econ.GN').

    Returns:
        JSON string with list of papers, each containing:
        arxiv_id, title, authors, abstract, published, updated, url, categories, pdf_url.
    """
    sort_criteria_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "recent": arxiv.SortCriterion.SubmittedDate,
        "updated": arxiv.SortCriterion.LastUpdatedDate,
    }
    sort_criterion = sort_criteria_map.get(sort_by, arxiv.SortCriterion.Relevance)

    # Optionally restrict to a category
    full_query = query
    if category:
        full_query = f"cat:{category} AND ({query})"

    client = arxiv.Client(
        page_size=min(max_results, 25),
        delay_seconds=1.0,
        num_retries=3,
    )
    search = arxiv.Search(
        query=full_query,
        max_results=min(max_results, 25),
        sort_by=sort_criterion,
    )

    results = []
    try:
        for paper in client.results(search):
            results.append(
                {
                    "arxiv_id": paper.entry_id,
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors],
                    "abstract": paper.summary,
                    "published": paper.published.isoformat() if paper.published else None,
                    "updated": paper.updated.isoformat() if paper.updated else None,
                    "url": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                    "categories": paper.categories,
                    "doi": paper.doi,
                    "journal_ref": paper.journal_ref,
                }
            )
    except arxiv.UnexpectedEmptyPageError:
        logger.warning("arXiv returned an empty page for query: '%s'", query)

    logger.info("arXiv Search: '%s' → %d papers", query, len(results))
    return json.dumps(results, indent=2, ensure_ascii=False)


@mcp.tool()
async def get_paper(arxiv_id: str) -> str:
    """
    Retrieve full details for a specific arXiv paper by its ID.

    Args:
        arxiv_id: arXiv paper ID (e.g. '2301.07041' or full URL).

    Returns:
        JSON string with full paper details.
    """
    # Normalise ID — strip URL prefix if present
    paper_id = arxiv_id.split("/")[-1].replace("abs/", "")

    client = arxiv.Client()
    search = arxiv.Search(id_list=[paper_id])

    results = list(client.results(search))
    if not results:
        return json.dumps({"error": f"Paper '{arxiv_id}' not found on arXiv"})

    paper = results[0]
    return json.dumps(
        {
            "arxiv_id": paper.entry_id,
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "abstract": paper.summary,
            "published": paper.published.isoformat() if paper.published else None,
            "updated": paper.updated.isoformat() if paper.updated else None,
            "url": paper.entry_id,
            "pdf_url": paper.pdf_url,
            "categories": paper.categories,
            "doi": paper.doi,
            "journal_ref": paper.journal_ref,
            "comment": paper.comment,
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool()
async def search_papers_by_author(
    author_name: str,
    max_results: int = 5,
) -> str:
    """
    Search arXiv for papers by a specific author.

    Args:
        author_name: Author name to search for.
        max_results: Maximum number of papers to return (1-25).

    Returns:
        JSON string with list of papers by this author.
    """
    query = f"au:{author_name}"
    client = arxiv.Client(page_size=min(max_results, 25))
    search = arxiv.Search(
        query=query,
        max_results=min(max_results, 25),
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    results = []
    for paper in client.results(search):
        results.append(
            {
                "arxiv_id": paper.entry_id,
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "abstract": paper.summary[:500] + "..."
                if len(paper.summary) > 500
                else paper.summary,
                "published": paper.published.isoformat() if paper.published else None,
                "url": paper.entry_id,
            }
        )

    return json.dumps(results, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    mcp.run()
