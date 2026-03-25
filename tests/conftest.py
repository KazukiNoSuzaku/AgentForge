"""
Pytest fixtures shared across all AgentForge test modules.

All LLM calls and MCP tool calls are mocked so tests run without
real API keys or network access. Fixtures are designed to be composable
and follow production data shapes exactly.
"""

from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.schemas import (
    AnalysisResult,
    Claim,
    CriticFeedback,
    ResearchFinding,
    ResearchPlan,
    SourceType,
    SubQuestion,
)
from src.state import ResearchState, create_initial_state

# ---------------------------------------------------------------------------
# pytest-asyncio configuration
# ---------------------------------------------------------------------------

pytest_plugins = ("pytest_asyncio",)


# ---------------------------------------------------------------------------
# Sample Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_query() -> str:
    return "Compare AI regulatory frameworks in the EU, US, and China"


@pytest.fixture
def sample_sub_questions() -> List[SubQuestion]:
    return [
        SubQuestion(
            id="sq_1",
            question="What are the key provisions of the EU AI Act?",
            priority=1,
            search_terms=["EU AI Act", "European AI regulation", "risk-based AI"],
            rationale="The EU AI Act is the most comprehensive AI regulation globally.",
        ),
        SubQuestion(
            id="sq_2",
            question="What is the current state of AI governance in the United States?",
            priority=2,
            search_terms=["US AI policy", "NIST AI framework", "executive order AI"],
            rationale="The US approach contrasts significantly with the EU's binding legislation.",
        ),
        SubQuestion(
            id="sq_3",
            question="How does China regulate AI systems?",
            priority=3,
            search_terms=[
                "China AI regulation",
                "algorithm recommendation rules",
                "generative AI China",
            ],
            rationale="China has introduced sector-specific AI rules distinct from Western approaches.",
        ),
    ]


@pytest.fixture
def sample_research_plan(sample_sub_questions) -> ResearchPlan:
    return ResearchPlan(
        sub_questions=sample_sub_questions,
        research_approach=(
            "Compare regulatory approaches across three major jurisdictions by examining "
            "primary legislation, enforcement mechanisms, and industry impact."
        ),
        estimated_complexity="high",
    )


@pytest.fixture
def sample_findings(sample_sub_questions) -> List[ResearchFinding]:
    return [
        ResearchFinding(
            sub_question_id="sq_1",
            sub_question=sample_sub_questions[0].question,
            source_type=SourceType.BRAVE,
            title="EU AI Act: A Comprehensive Overview",
            url="https://example.com/eu-ai-act",
            content="The EU AI Act introduces a risk-based framework classifying AI systems into four risk tiers.",
            relevance_score=0.92,
            key_insights=[
                "Risk categories: unacceptable, high, limited, minimal",
                "High-risk AI requires conformity assessment before deployment",
            ],
            metadata={"authors": [], "published_date": "2024-03-15"},
        ),
        ResearchFinding(
            sub_question_id="sq_2",
            sub_question=sample_sub_questions[1].question,
            source_type=SourceType.ARXIV,
            title="Governing AI in the United States: Fragmented but Emerging",
            url="https://arxiv.org/abs/2401.00001",
            content="US AI governance relies on existing sectoral regulation rather than omnibus legislation.",
            relevance_score=0.85,
            key_insights=[
                "Executive Order 14110 mandates safety evaluations for frontier AI",
                "NIST AI RMF provides voluntary risk management guidance",
            ],
            metadata={"authors": ["Smith, J.", "Doe, A."], "published_date": "2024-01-10"},
        ),
        ResearchFinding(
            sub_question_id="sq_3",
            sub_question=sample_sub_questions[2].question,
            source_type=SourceType.WIKIPEDIA,
            title="Regulation of artificial intelligence in China",
            url="https://en.wikipedia.org/wiki/Regulation_of_artificial_intelligence_in_China",
            content="China has enacted sector-specific AI regulations including the Algorithm Recommendation Regulations (2022) and Generative AI Measures (2023).",
            relevance_score=0.88,
            key_insights=[
                "China mandates security assessments for generative AI services",
                "Algorithm transparency requirements for recommendation systems",
            ],
            metadata={"authors": [], "published_date": "2024-02-01"},
        ),
    ]


@pytest.fixture
def sample_analysis(sample_findings) -> AnalysisResult:
    return AnalysisResult(
        patterns=[
            "Pattern: All three jurisdictions require some form of risk assessment for high-stakes AI. Evidence: https://example.com/eu-ai-act, https://arxiv.org/abs/2401.00001",
            "Pattern: Generative AI has prompted new or updated regulations in all three regions in 2023-2024.",
        ],
        contradictions=[
            "Contradiction: EU mandates binding pre-deployment conformity assessments; US relies on voluntary frameworks.",
        ],
        gaps=[
            "Gap: Enforcement track record and penalties are not well-documented in current sources.",
            "Gap: International interoperability agreements are not covered.",
        ],
        claims=[
            Claim(
                id="claim_1",
                statement="The EU AI Act is the world's first comprehensive binding AI regulation, covering all sectors.",
                confidence=0.93,
                supporting_sources=["https://example.com/eu-ai-act"],
                contradicting_sources=[],
                sub_question_id="sq_1",
            ),
            Claim(
                id="claim_2",
                statement="US AI governance relies on voluntary frameworks and sector-specific rules rather than omnibus legislation.",
                confidence=0.87,
                supporting_sources=["https://arxiv.org/abs/2401.00001"],
                contradicting_sources=[],
                sub_question_id="sq_2",
            ),
        ],
        confidence_summary=(
            "The research base is strong for EU and Chinese regulatory specifics, with high-quality "
            "sources. US coverage is slightly weaker, relying more on secondary analysis. "
            "Overall confidence is moderate-to-high."
        ),
    )


@pytest.fixture
def sample_draft_report() -> str:
    return """# AI Regulatory Frameworks: EU, US, and China — A Comparative Analysis

## Executive Summary
Three major jurisdictions have adopted distinct approaches to AI governance. The European Union
has enacted binding, comprehensive legislation via the AI Act [1]. The United States relies on
voluntary frameworks and executive orders [2]. China has introduced sector-specific regulations
targeting recommendation algorithms and generative AI [3].

## Introduction
As AI capabilities expand, governments worldwide are racing to establish governance frameworks...

## The EU's Risk-Based Approach
The EU AI Act classifies AI systems into four risk tiers [1]. High-risk applications require
conformity assessments before deployment...

## US: A Voluntary, Sectoral Model
Unlike the EU, the US has not enacted omnibus AI legislation. Instead, the NIST AI Risk Management
Framework (AI RMF) provides voluntary guidance [2]...

## China's Sector-Specific Regulations
China enacted the Algorithm Recommendation Regulations in 2022 and Generative AI Measures in 2023 [3].
Security assessments are mandatory for generative AI services...

## Conclusion
The three approaches reflect deeper political and governance philosophies. The EU prioritizes
harmonized consumer protection; the US favors innovation with light-touch oversight; China
emphasizes content control and security...

## Bibliography

[1] *EU AI Act: A Comprehensive Overview* (2024-03-15). Web. <https://example.com/eu-ai-act> (accessed 2025-01-15)

[2] Smith, J. and Doe, A.. *Governing AI in the United States: Fragmented but Emerging* (2024-01-10). arXiv. <https://arxiv.org/abs/2401.00001> (accessed 2025-01-15)

[3] *Regulation of artificial intelligence in China* (2024-02-01). Wikipedia. <https://en.wikipedia.org/wiki/Regulation_of_artificial_intelligence_in_China> (accessed 2025-01-15)
"""


@pytest.fixture
def sample_critic_feedback_pass() -> CriticFeedback:
    return CriticFeedback(
        quality_score=0.88,
        unsupported_claims=[],
        missing_perspectives=["Industry/business stakeholder perspective is thin"],
        logical_gaps=[],
        citation_issues=[],
        requires_revision=False,
        feedback_summary="The report is well-structured with accurate citations. Minor gap on industry perspective.",
        strengths=[
            "Clear comparative structure across all three jurisdictions",
            "All claims are grounded in cited sources",
            "Conclusion accurately synthesizes key differences",
        ],
    )


@pytest.fixture
def sample_critic_feedback_fail() -> CriticFeedback:
    return CriticFeedback(
        quality_score=0.62,
        unsupported_claims=[
            "Claim about China having the strictest penalties is not supported by any provided source"
        ],
        missing_perspectives=["Civil society and advocacy group perspectives are absent"],
        logical_gaps=["The conclusion jumps to policy recommendations without sufficient evidence"],
        citation_issues=["[4] is cited in text but has no bibliography entry"],
        requires_revision=True,
        feedback_summary=(
            "The report has good structure but contains one unsupported claim and a missing "
            "citation. Revision required to fix citation [4] and remove or source the penalties claim."
        ),
        strengths=[
            "Strong executive summary",
            "EU section is comprehensive and well-cited",
        ],
    )


@pytest.fixture
def sample_state(
    sample_query,
    sample_sub_questions,
    sample_findings,
    sample_analysis,
    sample_draft_report,
) -> ResearchState:
    """A complete mid-pipeline ResearchState for testing nodes in isolation."""
    state = create_initial_state(sample_query)
    state["sub_questions"] = sample_sub_questions
    state["research_plan"] = "Compare EU, US, and China AI regulations."
    state["research_findings"] = sample_findings
    state["analysis"] = sample_analysis
    state["draft_report"] = sample_draft_report
    state["revision_count"] = 0
    return state


# ---------------------------------------------------------------------------
# Mock LLM Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_manager():
    """Return a mock LLMManager with async-compatible methods."""
    manager = MagicMock()
    manager.ainvoke = AsyncMock()
    manager.ainvoke_structured = AsyncMock()
    return manager
