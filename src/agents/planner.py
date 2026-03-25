"""
Planner Agent

Decomposes the user's research question into 3-5 structured sub-questions
and produces a research plan. This is the entry point of the pipeline.

The Planner uses structured output to guarantee a machine-parseable plan
that downstream agents can act on deterministically.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import AppConfig, get_config
from src.models.schemas import ResearchPlan, ThinkingStep
from src.state import ResearchState
from src.utils.llm import LLMManager, get_llm_manager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """You are the Planner agent for AgentForge, an advanced multi-agent research system.

Your role is to decompose a complex research question into 3-5 focused sub-questions that together provide comprehensive coverage of the topic.

## Instructions

1. **Analyze** the question's scope, complexity, and key dimensions (e.g. technical, policy, historical, comparative).
2. **Decompose** it into 3-5 distinct sub-questions. Each sub-question should:
   - Be independently researchable via web search, academic papers, and encyclopaedia sources
   - Cover a different facet of the main question
   - Be concrete and specific (avoid vague sub-questions)
3. **Prioritize** sub-questions (1 = most critical to answering the main question).
4. **Suggest search terms** for each sub-question — terms that would yield high-quality results on Brave Search, arXiv, and Wikipedia.
5. **Estimate complexity**: low (single perspective), medium (2-3 perspectives), high (multiple domains or actors).

## Output Requirements

Return a valid ResearchPlan with:
- 3-5 SubQuestion objects, each with a unique id (sq_1, sq_2, ...), question, priority (1=highest), search_terms list, and rationale
- A research_approach string describing the overall strategy
- An estimated_complexity of "low", "medium", or "high"

## Examples of Good Sub-Questions

Main: "Compare AI regulatory frameworks in EU, US, and China"
Sub-questions:
1. What are the key provisions and risk-based approach of the EU AI Act?
2. What is the current state of AI governance and policy in the United States (federal vs. state level)?
3. How does China's AI regulation (Algorithm Recommendation Rules, Generative AI Measures) differ from Western approaches?
4. What are the international implications and potential for regulatory fragmentation or convergence?
5. How do industry stakeholders and civil society evaluate these frameworks?

## Critical Rules

- Do NOT generate sub-questions that overlap significantly
- Do NOT add more than 5 sub-questions
- Each sub-question should have 3-6 diverse search terms
- The rationale should explain why this angle is essential to the overall answer
"""


# ---------------------------------------------------------------------------
# Agent Class
# ---------------------------------------------------------------------------


class PlannerAgent:
    """
    Decomposes a research question into a structured plan.

    This agent is stateless — it takes a query and returns a ResearchPlan
    without any side effects. The LangGraph node wraps this class.
    """

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self._llm = llm_manager or get_llm_manager()
        self._config = config or get_config()

    async def plan(self, query: str) -> ResearchPlan:
        """
        Decompose a research query into a structured ResearchPlan.

        Args:
            query: The user's research question.

        Returns:
            A validated ResearchPlan with sub-questions and research approach.

        Raises:
            RuntimeError: If the LLM fails to produce a valid plan.
        """
        logger.info("Planner: decomposing query → '%s'", query)

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Research Question: {query}\n\n"
                    f"Maximum sub-questions allowed: {self._config.max_sub_questions}\n\n"
                    "Please create a structured research plan."
                )
            ),
        ]

        plan = await self._llm.ainvoke_structured(messages, ResearchPlan)

        # Enforce the configured maximum
        if len(plan.sub_questions) > self._config.max_sub_questions:
            plan = ResearchPlan(
                sub_questions=plan.sub_questions[: self._config.max_sub_questions],
                research_approach=plan.research_approach,
                estimated_complexity=plan.estimated_complexity,
            )

        logger.info(
            "Planner: produced %d sub-questions (complexity=%s)",
            len(plan.sub_questions),
            plan.estimated_complexity,
        )
        return plan


# ---------------------------------------------------------------------------
# LangGraph Node
# ---------------------------------------------------------------------------


async def planner_node(state: ResearchState) -> dict:
    """
    LangGraph node: run the Planner agent and update the shared state.

    Args:
        state: Current ResearchState.

    Returns:
        State delta dict with sub_questions, research_plan, agent_statuses,
        and thinking_steps.
    """
    query = state["query"]
    agent = PlannerAgent()

    thinking_step = ThinkingStep(
        agent="planner",
        action="decomposing_query",
        content=f"Analyzing research question and decomposing into sub-questions: '{query}'",
        timestamp=datetime.now(),
    )

    try:
        plan = await agent.plan(query)
        sub_questions = plan.sub_questions

        thinking_step_result = ThinkingStep(
            agent="planner",
            action="plan_complete",
            content=(
                f"Created {len(sub_questions)} sub-questions. Approach: {plan.research_approach}"
            ),
            timestamp=datetime.now(),
        )

        return {
            "sub_questions": sub_questions,
            "research_plan": plan.research_approach,
            "agent_statuses": {**state.get("agent_statuses", {}), "planner": "done"},
            "thinking_steps": [thinking_step, thinking_step_result],
            "errors": [],
        }

    except Exception as exc:
        error_msg = f"Planner failed: {exc}"
        logger.exception("Planner node error")
        return {
            "sub_questions": [],
            "research_plan": "",
            "agent_statuses": {**state.get("agent_statuses", {}), "planner": "error"},
            "thinking_steps": [thinking_step],
            "errors": [error_msg],
        }
