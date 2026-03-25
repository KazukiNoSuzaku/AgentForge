# ADR-005: Claude as Primary LLM with OpenAI Fallback

**Status:** Accepted
**Date:** 2024-12-22

## Context

AgentForge makes 8–15 LLM calls per research run (Planner, Researcher × N sub-questions, Analyst, Writer, Critic, plus potential revisions). A single API failure — rate limit, transient 500, network timeout — aborts the entire pipeline unless handled. The system needs a reliability strategy.

Constraints:

1. **Anthropic Claude is the primary model.** The system prompt engineering, structured output behavior, and quality calibration are tuned for Claude Sonnet. Switching to a different primary model isn't trivial — it's not just an API key swap.
2. **API outages happen.** Anthropic's API has had multiple multi-hour outages in 2024. OpenAI's API has a longer track record but also experiences incidents. Neither is 100% reliable.
3. **Cost matters.** This is a research tool, not a production SaaS. Users have API budgets. The fallback should be cost-comparable to the primary.
4. **Structured output must work on both providers.** The Planner, Analyst, and Critic use `.with_structured_output()` to return Pydantic models. The fallback must support the same interface.

## Decision

Implement an `LLMManager` class that wraps both providers behind a unified interface with automatic fallback. Anthropic Claude is primary; OpenAI GPT-4o is the fallback, used only when the primary fails.

```python
class LLMManager:
    @property
    def primary(self) -> BaseChatModel:
        if self._primary is None:
            self._primary = ChatAnthropic(
                model=self._config.anthropic_model,
                api_key=self._config.anthropic_api_key,
                temperature=self._config.llm_temperature,
                max_tokens=self._config.llm_max_tokens,
            )
        return self._primary

    @property
    def fallback(self) -> Optional[BaseChatModel]:
        if self._fallback is None and self._config.has_openai_fallback:
            self._fallback = ChatOpenAI(
                model=self._config.openai_model,
                api_key=self._config.openai_api_key,
                temperature=self._config.llm_temperature,
                max_tokens=self._config.llm_max_tokens,
            )
        return self._fallback
```

The invocation pattern for both plain text and structured output:

```python
async def ainvoke(self, messages: List[BaseMessage]) -> str:
    primary_error = None
    try:
        response = await self.primary.ainvoke(messages)
        return response.content
    except Exception as primary_err:
        primary_error = primary_err
        logger.warning("Primary LLM (%s) failed: %s", ...)

    if self.fallback is None:
        raise RuntimeError(
            f"Primary LLM failed and no fallback is configured. "
            f"Set OPENAI_API_KEY in your .env file. "
            f"Primary error: {primary_error}"
        )

    try:
        response = await self.fallback.ainvoke(messages)
        return response.content
    except Exception as fallback_err:
        raise RuntimeError(
            f"Both LLMs failed. "
            f"Primary ({self._config.anthropic_model}): {primary_error}. "
            f"Fallback ({self._config.openai_model}): {fallback_err}"
        ) from fallback_err
```

Key design choices:

- **Lazy initialization.** Clients are created on first use, not at import time. This avoids the OpenAI client being constructed when no fallback key is configured.
- **Catch-all exception handling.** We catch `Exception`, not specific API errors, because LangChain wraps provider errors inconsistently. A rate limit might surface as `anthropic.RateLimitError`, `langchain_core.exceptions.OutputParserException`, or a generic `Exception` depending on the code path.
- **Error variable scoping.** `primary_error` is saved explicitly before the except block exits, because Python 3 deletes the `as` variable after the except block. (This was a real bug we caught via ruff F821.)
- **Singleton accessor.** `get_llm_manager()` returns a module-level singleton so all agents share the same client instances and configuration.
- **Fallback is optional.** If `OPENAI_API_KEY` isn't set, the system runs with Anthropic only and raises a clear error if the primary fails. This keeps the minimum configuration simple (one API key).

## Alternatives Considered

### Single Provider, No Fallback

- **Pros:** Simplest code. One dependency. No cross-provider behavior differences to worry about.
- **Cons:** A single API outage kills the entire pipeline. During Anthropic's December 2024 capacity issues, this would have meant zero research runs for hours. For a portfolio project being demoed live, that's unacceptable.
- **Why not:** The fallback adds ~60 lines of code to `llm.py`. The risk reduction is disproportionate to the implementation cost.

### LangChain's Built-in Fallback Chain

- **Pros:** LangChain provides `.with_fallbacks([fallback_model])` which chains models automatically. Less custom code.
- **Cons:** The built-in fallback mechanism doesn't handle structured output fallback well — if the primary's `.with_structured_output()` fails, the fallback chain may not correctly re-wrap the fallback model with the same schema. Error reporting is opaque (you get a `FallbackError` with nested exceptions). Retry behavior is hardcoded in the chain with limited configurability.
- **Why not:** We tried it. The structured output path produced inconsistent behavior when falling back — sometimes returning raw JSON strings instead of validated Pydantic models. Writing explicit try/except gave us reliable behavior and clear error messages at the cost of more code.

### Multi-Provider Load Balancing

- **Pros:** Distribute calls across providers for lower latency. Use the cheapest provider per call type. Automatic failover.
- **Cons:** Massive complexity increase. Different providers have different prompt optimization profiles — a prompt tuned for Claude may produce worse results on GPT-4o. Structured output schemas don't always translate cleanly across providers. Cost tracking becomes a matrix. All of this for a system that makes 8–15 calls per run, not thousands.
- **Why not:** Over-engineering. AgentForge is a single-user research tool, not a high-availability API service. Primary + fallback is the right level of redundancy.

### Retry with Exponential Backoff (Single Provider)

- **Pros:** Handles transient errors (429s, 503s) without switching providers. Simpler than multi-provider fallback.
- **Cons:** Doesn't help with sustained outages. Retries add latency to every failure. Rate limit backoff can push a single agent call from 3 seconds to 30+ seconds. Doesn't help when the provider is down entirely.
- **Why not:** Retries are complementary to fallback, not a substitute. LangChain's underlying `httpx` client already retries on transient HTTP errors. Adding application-level retries on top provides diminishing returns for transient issues and no benefit for outages.

## Consequences

### Positive

- **Pipeline resilience.** An Anthropic API outage triggers automatic fallback. Users see a log warning, not a crash. The quality may differ slightly (see Negative), but the pipeline completes.
- **Single interface for all agents.** Agents call `llm_manager.ainvoke(messages)` or `llm_manager.ainvoke_structured(messages, Schema)`. They don't know or care which provider responds. Swapping models is a config change, not a code change.
- **Clear error messages.** When both providers fail, the error includes both failure reasons. When only the primary fails, the fallback success is logged. When no fallback is configured, the error message tells you exactly what to do ("Set OPENAI_API_KEY in your .env file").
- **Testable.** Tests mock `LLMManager` at a single point. No need to mock two different provider clients separately.

### Negative

- **Quality variance on fallback.** System prompts are optimized for Claude's behavior. When GPT-4o takes over, structured output reliability, citation style, and analysis depth may differ. We don't re-optimize prompts per provider — the fallback is "good enough," not "equivalent."
- **Two provider dependencies.** `langchain_anthropic` and `langchain_openai` are both in the dependency tree, even if only one is used at runtime. This increases the potential for dependency conflicts and the attack surface for supply chain issues.
- **No per-call provider selection.** If the Planner call fails on Anthropic, all subsequent calls in that run still try Anthropic first. We could add circuit-breaker logic (after N failures, skip the primary for M seconds), but that's not implemented — each call independently tries primary then fallback.
- **Python 3 scoping gotcha.** The `except Exception as e` variable deletion required an explicit workaround (`primary_error = primary_err` inside the block). This is correct but non-obvious, and someone unfamiliar with the pattern might "simplify" it into a bug during refactoring.

## References

- `src/utils/llm.py` — `LLMManager` implementation
- `src/config.py` — `has_openai_fallback` property, API key configuration
- [LangChain ChatAnthropic](https://python.langchain.com/docs/integrations/chat/anthropic/)
- [LangChain ChatOpenAI](https://python.langchain.com/docs/integrations/chat/openai/)
- [Anthropic API status](https://status.anthropic.com/)
