"""
Cost tracking for AgentForge LLM calls.

Records input/output token counts per agent, calculates running cost
estimates, and exports structured summaries for logging and benchmarks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing Tables (per 1M tokens)
# ---------------------------------------------------------------------------

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # Anthropic
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-3-5": {"input": 0.80, "output": 4.00},
    "claude-haiku-3-5-20241022": {"input": 0.80, "output": 4.00},
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    # OpenAI (fallback)
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

# Fallback pricing when model not in table
DEFAULT_PRICING = {"input": 3.00, "output": 15.00}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class LLMCallRecord:
    """A single tracked LLM call."""

    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str
    structured_output: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "timestamp": self.timestamp,
            "structured_output": self.structured_output,
        }


@dataclass
class AgentCostSummary:
    """Aggregated cost data for a single agent."""

    agent: str
    call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }


# ---------------------------------------------------------------------------
# Cost Tracker
# ---------------------------------------------------------------------------


class CostTracker:
    """
    Tracks LLM token usage and cost across agent calls.

    Usage:
        tracker = CostTracker(model="claude-sonnet-4-5")
        tracker.track_call("planner", input_tokens=800, output_tokens=400)
        tracker.track_call("writer", input_tokens=3200, output_tokens=1000)
        summary = tracker.get_summary()
    """

    def __init__(self, model: str = "claude-sonnet-4-5") -> None:
        self._model = model
        self._pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
        self._calls: List[LLMCallRecord] = []
        self._agent_summaries: Dict[str, AgentCostSummary] = {}

    @property
    def model(self) -> str:
        return self._model

    @property
    def total_cost(self) -> float:
        return sum(c.cost_usd for c in self._calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.input_tokens + c.output_tokens for c in self._calls)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    @staticmethod
    def _calculate_cost(input_tokens: int, output_tokens: int, pricing: Dict[str, float]) -> float:
        """Calculate cost in USD for a given token count and pricing."""
        return (input_tokens / 1_000_000) * pricing["input"] + (
            output_tokens / 1_000_000
        ) * pricing["output"]

    def track_call(
        self,
        agent: str,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None,
        structured_output: bool = False,
    ) -> LLMCallRecord:
        """
        Record a single LLM call.

        Args:
            agent: Name of the agent that made the call (e.g. "planner", "writer").
            input_tokens: Number of input (prompt) tokens.
            output_tokens: Number of output (completion) tokens.
            model: Model used for this call. Defaults to tracker's model.
            structured_output: Whether this was a structured output call.

        Returns:
            The recorded LLMCallRecord.
        """
        call_model = model or self._model
        pricing = MODEL_PRICING.get(call_model, DEFAULT_PRICING)
        cost = self._calculate_cost(input_tokens, output_tokens, pricing)

        record = LLMCallRecord(
            agent=agent,
            model=call_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            timestamp=datetime.now().isoformat(),
            structured_output=structured_output,
        )
        self._calls.append(record)

        # Update per-agent summary
        if agent not in self._agent_summaries:
            self._agent_summaries[agent] = AgentCostSummary(agent=agent)
        summary = self._agent_summaries[agent]
        summary.call_count += 1
        summary.total_input_tokens += input_tokens
        summary.total_output_tokens += output_tokens
        summary.total_cost_usd += cost

        logger.debug(
            "CostTracker: %s [%s] %d in + %d out = $%.4f",
            agent,
            call_model,
            input_tokens,
            output_tokens,
            cost,
        )

        return record

    def get_summary(self) -> Dict[str, Any]:
        """
        Return an aggregated cost summary across all tracked calls.

        Returns:
            Dict with total cost, token counts, per-agent breakdowns,
            and individual call records.
        """
        total_input = sum(c.input_tokens for c in self._calls)
        total_output = sum(c.output_tokens for c in self._calls)

        return {
            "model": self._model,
            "total_calls": len(self._calls),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost_usd": round(self.total_cost, 6),
            "cost_by_agent": {
                name: round(s.total_cost_usd, 6) for name, s in self._agent_summaries.items()
            },
            "calls_by_agent": {name: s.call_count for name, s in self._agent_summaries.items()},
            "tokens_by_agent": {
                name: {
                    "input": s.total_input_tokens,
                    "output": s.total_output_tokens,
                    "total": s.total_tokens,
                }
                for name, s in self._agent_summaries.items()
            },
            "agent_summaries": [s.to_dict() for s in self._agent_summaries.values()],
        }

    def get_agent_cost(self, agent: str) -> float:
        """Return total cost in USD for a specific agent."""
        if agent in self._agent_summaries:
            return self._agent_summaries[agent].total_cost_usd
        return 0.0

    def reset(self) -> None:
        """Clear all tracked data."""
        self._calls.clear()
        self._agent_summaries.clear()

    def export_json(self, path: Optional[Path] = None) -> str:
        """
        Export full tracking data as JSON.

        Args:
            path: Optional file path to write to. If None, returns the
                  JSON string without writing to disk.

        Returns:
            JSON string of the complete tracking data.
        """
        data = {
            "summary": self.get_summary(),
            "calls": [c.to_dict() for c in self._calls],
        }
        json_str = json.dumps(data, indent=2)

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_str, encoding="utf-8")
            logger.info("CostTracker: exported to %s", path)

        return json_str

    def __repr__(self) -> str:
        return (
            f"CostTracker(model={self._model!r}, "
            f"calls={len(self._calls)}, "
            f"cost=${self.total_cost:.4f})"
        )
