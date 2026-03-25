"""
Citation formatting utilities for research reports.

Handles the construction of numbered inline citations, bibliography
sections, and citation validation against the provided sources.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.models.schemas import CitationEntry, ResearchFinding, SourceType


def build_citation_map(findings: List[ResearchFinding]) -> Dict[str, CitationEntry]:
    """
    Build a mapping from source URL to CitationEntry for all research findings.

    Deduplicates sources so each unique URL gets exactly one citation number.
    Citations are numbered in order of first appearance across findings.

    Args:
        findings: All research findings from the Researcher agent.

    Returns:
        A dict mapping URL → CitationEntry with sequential numbering.
    """
    seen_urls: Dict[str, CitationEntry] = {}
    counter = 1

    for finding in findings:
        if finding.url not in seen_urls:
            entry = CitationEntry(
                number=counter,
                title=finding.title,
                url=finding.url,
                source_type=finding.source_type,
                authors=finding.metadata.get("authors", []),
                published_date=finding.metadata.get("published_date"),
            )
            seen_urls[finding.url] = entry
            counter += 1

    return seen_urls


def format_bibliography(citation_map: Dict[str, CitationEntry]) -> str:
    """
    Render a formatted bibliography section in markdown.

    Args:
        citation_map: URL → CitationEntry mapping from build_citation_map().

    Returns:
        A markdown-formatted bibliography string, sorted by citation number.
    """
    if not citation_map:
        return ""

    sorted_entries = sorted(citation_map.values(), key=lambda e: e.number)
    lines = ["## Bibliography", ""]

    for entry in sorted_entries:
        source_label = _source_type_label(entry.source_type)
        authors_str = _format_authors(entry.authors)
        date_str = f" ({entry.published_date})" if entry.published_date else ""

        if authors_str:
            line = f"[{entry.number}] {authors_str}. *{entry.title}*{date_str}. {source_label}. <{entry.url}> (accessed {entry.accessed_date})"
        else:
            line = f"[{entry.number}] *{entry.title}*{date_str}. {source_label}. <{entry.url}> (accessed {entry.accessed_date})"

        lines.append(line)
        lines.append("")

    return "\n".join(lines)


def validate_citations(report: str, citation_map: Dict[str, CitationEntry]) -> List[str]:
    """
    Check that every inline citation [N] in the report has an entry in the map.

    Args:
        report: The markdown report text.
        citation_map: URL → CitationEntry mapping.

    Returns:
        A list of issue descriptions (empty list means all citations are valid).
    """
    issues = []
    valid_numbers = {entry.number for entry in citation_map.values()}

    # Find all inline citations like [1], [2], [3]
    inline_refs = re.findall(r"\[(\d+)\]", report)

    referenced_numbers = set()
    for ref_str in inline_refs:
        n = int(ref_str)
        referenced_numbers.add(n)
        if n not in valid_numbers:
            issues.append(f"Citation [{n}] referenced in text but has no bibliography entry")

    # Check for bibliography entries that are never cited
    for number in valid_numbers:
        if number not in referenced_numbers:
            issues.append(f"Bibliography entry [{number}] exists but is never cited in the text")

    return issues


def inject_bibliography(report: str, citation_map: Dict[str, CitationEntry]) -> str:
    """
    Append (or replace) the bibliography section in a report.

    If the report already contains a '## Bibliography' heading, the content
    below it is replaced. Otherwise, the bibliography is appended at the end.

    Args:
        report: The markdown report text.
        citation_map: URL → CitationEntry mapping.

    Returns:
        The report with an up-to-date bibliography section.
    """
    bibliography = format_bibliography(citation_map)

    # Strip any existing bibliography section
    report = re.sub(
        r"\n## Bibliography.*$",
        "",
        report,
        flags=re.DOTALL | re.MULTILINE,
    ).rstrip()

    return f"{report}\n\n{bibliography}"


def extract_citation_numbers(text: str) -> List[int]:
    """
    Extract all citation numbers referenced in a block of text.

    Args:
        text: Any text that may contain [N] style citations.

    Returns:
        Sorted list of unique citation numbers found.
    """
    refs = re.findall(r"\[(\d+)\]", text)
    return sorted(set(int(r) for r in refs))


def url_to_citation_number(url: str, citation_map: Dict[str, CitationEntry]) -> Optional[int]:
    """
    Look up the citation number for a given URL.

    Args:
        url: Source URL to look up.
        citation_map: URL → CitationEntry mapping.

    Returns:
        The citation number, or None if the URL is not in the map.
    """
    entry = citation_map.get(url)
    return entry.number if entry else None


def format_inline_citation(numbers: List[int]) -> str:
    """
    Format a list of citation numbers as an inline citation string.

    Example: [1, 3, 5] → "[1][3][5]"

    Args:
        numbers: Sorted list of citation numbers.

    Returns:
        Formatted inline citation string.
    """
    return "".join(f"[{n}]" for n in sorted(set(numbers)))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _source_type_label(source_type: SourceType) -> str:
    """Return a human-readable label for a source type."""
    labels = {
        SourceType.BRAVE: "Web",
        SourceType.ARXIV: "arXiv",
        SourceType.WIKIPEDIA: "Wikipedia",
    }
    return labels.get(source_type, "Web")


def _format_authors(authors: List[str]) -> str:
    """Format a list of author names into a citation string."""
    if not authors:
        return ""
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    return f"{authors[0]} et al."
