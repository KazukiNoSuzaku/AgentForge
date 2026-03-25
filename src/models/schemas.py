"""
Pydantic schemas for all data structures in AgentForge.

These models enforce type safety across agent boundaries and serve as
the single source of truth for data shapes throughout the pipeline.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AgentStatus(str, Enum):
    """Lifecycle states for each agent in the pipeline."""

    IDLE = "idle"
    ACTIVE = "active"
    DONE = "done"
    ERROR = "error"


class SourceType(str, Enum):
    """Supported research data sources."""

    BRAVE = "brave"
    ARXIV = "arxiv"
    WIKIPEDIA = "wikipedia"


# ---------------------------------------------------------------------------
# Planning Phase
# ---------------------------------------------------------------------------


class SubQuestion(BaseModel):
    """A single decomposed sub-question derived from the user's main query."""

    id: str = Field(description="Unique identifier, e.g. 'sq_1'")
    question: str = Field(description="The sub-question to research")
    priority: int = Field(default=1, ge=1, le=5, description="1 = highest priority")
    search_terms: List[str] = Field(
        default_factory=list,
        description="Suggested search keywords for this sub-question",
    )
    rationale: str = Field(
        default="",
        description="Why this sub-question is important to the overall research",
    )


class ResearchPlan(BaseModel):
    """Structured output from the Planner agent."""

    sub_questions: List[SubQuestion] = Field(
        description="3-5 sub-questions covering all dimensions of the main query"
    )
    research_approach: str = Field(description="High-level description of the research strategy")
    estimated_complexity: Literal["low", "medium", "high"] = Field(
        description="Estimated complexity of the research task"
    )


# ---------------------------------------------------------------------------
# Research Phase
# ---------------------------------------------------------------------------


class RawSearchResult(BaseModel):
    """Raw result from any search tool before LLM evaluation."""

    source_type: SourceType
    title: str
    url: str
    snippet: str = Field(description="Short excerpt or abstract from the source")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific metadata (authors, published date, etc.)",
    )


class ResearchFinding(BaseModel):
    """An evaluated, annotated research finding linked to a sub-question."""

    sub_question_id: str
    sub_question: str
    source_type: SourceType
    title: str
    url: str
    content: str = Field(description="Extracted relevant passage or summary")
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How relevant this finding is to the sub-question (0-1)",
    )
    key_insights: List[str] = Field(
        default_factory=list,
        description="Bullet-point insights extracted from this source",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Analysis Phase
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    """A discrete, citable claim extracted during analysis."""

    id: str = Field(description="Unique identifier, e.g. 'claim_1'")
    statement: str = Field(description="The claim as a clear, concise sentence")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score based on source corroboration (0-1)",
    )
    supporting_sources: List[str] = Field(
        default_factory=list,
        description="URLs of sources that support this claim",
    )
    contradicting_sources: List[str] = Field(
        default_factory=list,
        description="URLs of sources that contradict this claim",
    )
    sub_question_id: str = Field(description="Which sub-question this claim addresses")


class AnalysisResult(BaseModel):
    """Structured output from the Analyst agent."""

    patterns: List[str] = Field(description="Cross-source patterns and recurring themes")
    contradictions: List[str] = Field(description="Identified contradictions between sources")
    gaps: List[str] = Field(description="Missing information not covered by the research")
    claims: List[Claim] = Field(description="Discrete, citable claims with confidence scores")
    confidence_summary: str = Field(description="Narrative summary of overall research confidence")


# ---------------------------------------------------------------------------
# Writing Phase
# ---------------------------------------------------------------------------


class CitationEntry(BaseModel):
    """A bibliography entry for the final report."""

    number: int = Field(description="Citation number as it appears in the text")
    title: str
    url: str
    source_type: SourceType
    authors: List[str] = Field(default_factory=list)
    published_date: Optional[str] = None
    accessed_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))


class ReportSection(BaseModel):
    """A single section of the research report."""

    title: str
    content: str = Field(description="Markdown-formatted section content with inline citations")
    sub_question_ids: List[str] = Field(
        default_factory=list,
        description="Sub-questions this section addresses",
    )


# ---------------------------------------------------------------------------
# Critic Phase
# ---------------------------------------------------------------------------


class CriticFeedback(BaseModel):
    """Structured output from the Critic agent."""

    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall quality score (0-1). Revision triggered if < threshold.",
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Claims made in the report not backed by provided sources",
    )
    missing_perspectives: List[str] = Field(
        default_factory=list,
        description="Important viewpoints or stakeholders not addressed",
    )
    logical_gaps: List[str] = Field(
        default_factory=list,
        description="Reasoning gaps or non-sequiturs in the argument",
    )
    citation_issues: List[str] = Field(
        default_factory=list,
        description="Incorrect, missing, or misattributed citations",
    )
    requires_revision: bool = Field(
        description="Whether the report should be sent back to the Writer"
    )
    feedback_summary: str = Field(description="Concise summary of feedback for the Writer agent")
    strengths: List[str] = Field(
        default_factory=list,
        description="Positive aspects of the report to preserve in revision",
    )


# ---------------------------------------------------------------------------
# Pipeline Metadata
# ---------------------------------------------------------------------------


class ThinkingStep(BaseModel):
    """A single reasoning step recorded during agent execution."""

    agent: str
    action: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class PipelineMetrics(BaseModel):
    """End-to-end metrics for a research run."""

    total_sources_found: int = 0
    sources_by_type: Dict[str, int] = Field(default_factory=dict)
    revision_count: int = 0
    final_quality_score: Optional[float] = None
    total_claims: int = 0
    high_confidence_claims: int = 0  # confidence >= 0.8
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Total pipeline duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
