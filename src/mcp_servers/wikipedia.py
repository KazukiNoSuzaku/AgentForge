"""
MCP Server: Wikipedia

Exposes Wikipedia search and article retrieval as MCP tools.
Useful for background knowledge, definitions, and reference information.

Run standalone for testing:
    python src/mcp_servers/wikipedia.py
"""

from __future__ import annotations

import json
import logging
import sys
from typing import List, Optional

import wikipedia as wiki_api
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="Wikipedia",
    instructions=(
        "Use search_wikipedia to find articles on a topic. "
        "Use get_article_summary for a concise overview. "
        "Use get_article_sections to retrieve a specific section of a long article."
    ),
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def search_wikipedia(
    query: str,
    num_results: int = 5,
    language: str = "en",
) -> str:
    """
    Search Wikipedia for articles matching a query.

    Args:
        query: Search terms.
        num_results: Number of article titles to return (1-20).
        language: Wikipedia language code (e.g. 'en', 'de', 'fr').

    Returns:
        JSON string with a list of matching article titles.
    """
    wiki_api.set_lang(language)
    results = wiki_api.search(query, results=num_results)
    logger.info("Wikipedia search: '%s' → %d titles", query, len(results))
    return json.dumps({"query": query, "results": results}, ensure_ascii=False)


@mcp.tool()
async def get_article_summary(
    title: str,
    sentences: int = 10,
    language: str = "en",
) -> str:
    """
    Retrieve a summary of a Wikipedia article.

    Automatically resolves disambiguation pages by selecting the first option.

    Args:
        title: Exact or approximate Wikipedia article title.
        sentences: Number of sentences to include in the summary (1-20).
        language: Wikipedia language code.

    Returns:
        JSON string with article title, url, summary, and categories.
    """
    wiki_api.set_lang(language)

    try:
        page = wiki_api.page(title, auto_suggest=True, preload=False)
        summary = wiki_api.summary(title, sentences=sentences, auto_suggest=True)

        return json.dumps(
            {
                "title": page.title,
                "url": page.url,
                "summary": summary,
                "categories": page.categories[:15],
                "references": page.references[:10],
            },
            indent=2,
            ensure_ascii=False,
        )

    except wiki_api.DisambiguationError as e:
        # Disambiguation — pick the first option and recurse
        logger.info("Disambiguation for '%s': trying '%s'", title, e.options[0])
        first_option = e.options[0]
        try:
            page = wiki_api.page(first_option, auto_suggest=False)
            summary = wiki_api.summary(first_option, sentences=sentences, auto_suggest=False)
            return json.dumps(
                {
                    "title": page.title,
                    "url": page.url,
                    "summary": summary,
                    "categories": page.categories[:15],
                    "disambiguation_note": (
                        f"'{title}' was ambiguous; retrieved '{first_option}' instead. "
                        f"Other options: {e.options[1:5]}"
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        except Exception as inner_err:
            return json.dumps(
                {"error": f"Could not resolve disambiguation for '{title}': {inner_err}"}
            )

    except wiki_api.PageError:
        return json.dumps(
            {
                "error": f"Wikipedia page '{title}' not found.",
                "suggestion": "Try search_wikipedia first to find the correct title.",
            }
        )

    except Exception as err:
        return json.dumps({"error": f"Wikipedia lookup failed: {err}"})


@mcp.tool()
async def get_article_sections(
    title: str,
    section_title: Optional[str] = None,
    language: str = "en",
) -> str:
    """
    Retrieve the full content of a Wikipedia article or a specific section.

    Args:
        title: Wikipedia article title.
        section_title: Optional specific section to retrieve (e.g. 'History').
                       If None, returns the table of contents with first 2000 chars.
        language: Wikipedia language code.

    Returns:
        JSON string with section content or table of contents.
    """
    wiki_api.set_lang(language)

    try:
        page = wiki_api.page(title, auto_suggest=True, preload=True)

        if section_title is None:
            # Return TOC + introduction
            return json.dumps(
                {
                    "title": page.title,
                    "url": page.url,
                    "sections": page.sections,
                    "introduction": page.content[:2000] + "..." if len(page.content) > 2000 else page.content,
                },
                indent=2,
                ensure_ascii=False,
            )
        else:
            section_content = page.section(section_title)
            if section_content is None:
                return json.dumps(
                    {
                        "error": f"Section '{section_title}' not found in '{title}'",
                        "available_sections": page.sections,
                    }
                )
            return json.dumps(
                {
                    "title": page.title,
                    "section": section_title,
                    "content": section_content,
                    "url": f"{page.url}#{section_title.replace(' ', '_')}",
                },
                indent=2,
                ensure_ascii=False,
            )

    except wiki_api.PageError:
        return json.dumps({"error": f"Wikipedia page '{title}' not found."})
    except Exception as err:
        return json.dumps({"error": f"Failed to retrieve article: {err}"})


@mcp.tool()
async def get_article_references(
    title: str,
    language: str = "en",
) -> str:
    """
    Retrieve the external references/links from a Wikipedia article.

    Useful for discovering primary sources cited by Wikipedia.

    Args:
        title: Wikipedia article title.
        language: Wikipedia language code.

    Returns:
        JSON string with a list of external reference URLs.
    """
    wiki_api.set_lang(language)

    try:
        page = wiki_api.page(title, auto_suggest=True)
        return json.dumps(
            {
                "title": page.title,
                "references": page.references,
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as err:
        return json.dumps({"error": str(err)})


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    mcp.run()
