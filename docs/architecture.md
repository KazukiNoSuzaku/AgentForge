# AgentForge: Architecture & Design Decisions

## Overview

AgentForge is a production-grade multi-agent research engine built on LangGraph and the Model Context Protocol (MCP). This document explains the key architectural decisions and the trade-offs considered during design.

---

## System Architecture

### Agent Pipeline (State Machine)

```
User Query
    │
    ▼
┌─────────┐    structured plan    ┌────────────┐
│ Planner │──────────────────────▶│ Researcher │
└─────────┘                       └─────┬──────┘
                                         │ parallel MCP calls
                          ┌──────────────┼──────────────┐
                          ▼              ▼              ▼
                    Brave Search      arXiv         Wikipedia
                          │              │              │
                          └──────────────┴──────────────┘
                                         │ all findings
                                         ▼
                                   ┌──────────┐
                                   │ Analyst  │
                                   └─────┬────┘
                                         │ structured analysis
                                         ▼
                                   ┌────────┐
                               ┌──▶│ Writer │◀──┐
                               │   └────┬───┘   │
                               │        │ draft  │
                               │        ▼       │
                               │   ┌────────┐   │
                               │   │ Critic │   │
                               │   └────┬───┘   │
                               │        │       │
                               │  score < 0.75  │
                               │  revision < 2  │
                               └────────┘       │
                                        │       │
                                   score ≥ 0.75 │
                                   OR at max     │
                                        │        │
                                        ▼
                                  Final Report
```

### State Flow

Each LangGraph node reads from and writes to a shared `ResearchState` TypedDict. Fields use `Annotated` reducers for lists that accumulate across updates:

| Field | Reducer | Set By |
|-------|---------|--------|
| `query` | overwrite | entry point |
| `sub_questions` | overwrite | planner |
| `research_findings` | `operator.add` (accumulate) | researcher |
| `analysis` | overwrite | analyst |
| `draft_report` | overwrite | writer |
| `critic_feedback` | overwrite | critic |
| `final_report` | overwrite | critic (on approval) |
| `thinking_steps` | `operator.add` (accumulate) | all agents |
| `errors` | `operator.add` (accumulate) | all agents |

---

## Design Decision 1: LangGraph over CrewAI

**Decision**: Use LangGraph as the orchestration framework.

**Rationale**:

LangGraph exposes the underlying state machine explicitly, giving us:

1. **Conditional edges with first-class support**: The Critic→Writer revision loop requires a conditional edge with bounded iteration. In LangGraph this is `add_conditional_edges()` with a routing function. In CrewAI, conditional flows require workarounds that break the declarative model.

2. **Fine-grained state control**: LangGraph's TypedDict state with Annotated reducers lets us specify exactly how each field is merged across node updates. CrewAI abstracts state away, making it difficult to reason about data lineage.

3. **Async-native**: LangGraph nodes are `async def` functions that participate in Python's asyncio event loop. The Researcher's parallel MCP calls use `asyncio.gather` — this would require thread pool hacks in CrewAI.

4. **Debugging and observability**: LangGraph's `stream()` method yields `{node_name: state_delta}` dicts, making it straightforward to build a real-time UI. CrewAI's callback system is more opaque.

5. **No hidden magic**: LangGraph is explicit about what each node does and how control flows. This matters for production reliability.

**Trade-off**: LangGraph requires more boilerplate than CrewAI's role-based abstractions. For simple linear pipelines with no branching, CrewAI's DSL is more convenient. AgentForge's revision loop makes the extra control worth it.

---

## Design Decision 2: MCP over Direct API Calls

**Decision**: Expose Brave Search, arXiv, and Wikipedia as MCP servers rather than calling APIs directly.

**Rationale**:

1. **Standardization**: MCP defines a standard protocol for LLM-to-tool communication. Any MCP-compatible client (Claude Desktop, VS Code Copilot, AgentForge) can use these servers without modification.

2. **Composability**: MCP servers are self-contained processes with their own dependency management. Adding a new data source means writing a new MCP server, not modifying the Researcher agent.

3. **Separation of concerns**: The Researcher agent doesn't need to know about HTTP clients, API authentication, or rate limiting — it just calls `session.call_tool()`. Each MCP server is responsible for its own error handling and API communication.

4. **Testability**: MCP servers can be tested in isolation by calling them directly from the CLI. The Researcher agent can be tested by mocking `call_mcp_tool()`.

5. **Future portability**: As the MCP ecosystem matures, community-built MCP servers (for databases, code execution, etc.) can be plugged in without changing the agent code.

**Trade-off**: MCP adds a subprocess-per-call overhead (each tool call spawns or communicates with a subprocess). For high-throughput production use, a persistent connection pool would be preferable. The current implementation prioritizes correctness and portability over call latency.

**Implementation note**: MCP servers use the `FastMCP` SDK (`mcp.server.fastmcp`) and communicate via stdio transport. Each call uses `asyncio.timeout` to prevent hanging on unresponsive servers.

---

## Design Decision 3: Critic Revision Loop

**Decision**: Include a Critic agent that can trigger up to 2 revision cycles.

**Rationale**:

1. **Hallucination risk**: LLMs sometimes fabricate citations or make unsupported claims. A dedicated Critic with access to the original research findings can cross-check claims against the actual source material.

2. **Quality floor**: A single-pass Writer has no feedback signal. The Critic provides a structured quality assessment (0.0–1.0) that grounds the revision process in measurable criteria.

3. **Bounded iteration**: The 2-revision cap prevents infinite loops while allowing meaningful improvement. Empirically, a single revision cycle resolves most structural issues; the second cycle catches residual citation problems.

4. **Separation of writing and evaluation**: Asking the Writer to self-evaluate introduces obvious bias. A separate Critic with a different system prompt provides a more honest assessment.

**Trade-off**: Each revision loop adds 1-2 LLM calls and 15-30 seconds of latency. For research tasks where quality matters more than speed, this is acceptable. A configurable `MAX_REVISION_LOOPS=0` option disables the loop for faster but lower-quality outputs.

---

## Design Decision 4: Parallel Research Execution

**Decision**: Research all sub-questions concurrently using `asyncio.gather`.

**Rationale**:

Network-bound MCP tool calls are the bottleneck in the Researcher phase. Searching Brave, arXiv, and Wikipedia for 3-5 sub-questions sequentially would take 60-120 seconds. Parallel execution reduces this to the latency of the slowest single call (~15-30 seconds).

```python
# Parallel: all sub-questions searched concurrently
tasks = [agent.research(sq) for sq in sub_questions]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

The `return_exceptions=True` flag ensures one failing task doesn't cancel the others — critical for resilience when one search source is unavailable.

**Trade-off**: Parallel calls may hit API rate limits faster than sequential calls. The `max_search_results` config parameter reduces request volume, and each MCP server implements respectful defaults (1-second delay between arXiv calls).

---

## Design Decision 5: Structured Output via Pydantic

**Decision**: All agent outputs are Pydantic models, retrieved via LangChain's `.with_structured_output()`.

**Rationale**:

1. **Reliability**: Unstructured LLM responses are brittle. Pydantic models with field-level validation catch malformed outputs at the agent boundary rather than propagating errors downstream.

2. **Type safety**: Downstream agents receive typed objects, not raw strings. This eliminates a whole class of KeyError/AttributeError bugs.

3. **Documentation**: Pydantic models serve as executable documentation of the data contract between agents.

4. **LangChain compatibility**: `.with_structured_output()` uses tool-calling under the hood, which is more reliable than JSON mode for complex nested schemas.

**Trade-off**: Structured output is slightly slower than raw text generation and can fail if the schema is too complex for the model to follow. The Researcher agent uses raw text + JSON parsing (with error handling) for the most complex schema — a pragmatic trade-off.

---

## LLM Fallback Strategy

```
Request
  │
  ▼
Anthropic claude-sonnet-4-5
  │ on failure
  ▼
OpenAI gpt-4o
  │ on failure
  ▼
RuntimeError (both unavailable)
```

The `LLMManager` class implements this pattern with a single `ainvoke_structured()` method. The fallback is transparent to calling code — agents just call the manager without knowing which provider responded.

---

## Error Handling Philosophy

AgentForge follows a **fail-gracefully** principle:

- **MCP tool failures**: Return `None`, log a warning, continue with available data. A research run without Brave Search is degraded but functional.
- **Agent failures**: Record the error in `state["errors"]`, mark the agent as "error", and allow downstream agents to continue. The Critic approves the draft on failure to prevent pipeline deadlock.
- **LLM failures**: Try fallback first, then raise `RuntimeError` with a descriptive message. The user sees a clear explanation of which provider failed and why.

---

## Performance Characteristics

| Phase | Typical Duration | Bottleneck |
|-------|-----------------|-----------|
| Planner | 3-8s | Single LLM call |
| Researcher | 20-45s | Parallel MCP calls (network I/O) |
| Analyst | 8-15s | Single LLM call with large context |
| Writer | 15-25s | Long-form generation |
| Critic | 8-15s | Structured output call |
| **Total (0 revisions)** | **55-110s** | |
| **Total (1 revision)** | **80-150s** | |

---

## Metrics & Evaluation

The following metrics are tracked per research run:

| Metric | Definition | Target |
|--------|-----------|--------|
| `quality_score` | Critic's final assessment (0-1) | ≥ 0.75 |
| `total_sources` | Unique sources retrieved | ≥ 8 |
| `high_confidence_claims` | Claims with confidence ≥ 0.8 | ≥ 3 |
| `revision_count` | Writer→Critic cycles | ≤ 2 |
| `total_duration_s` | End-to-end wall time | ≤ 150s |
| `hallucination_rate` | Unsupported claims / total claims | ≤ 0.10 |

For offline evaluation, compare AgentForge outputs against ground-truth research on known topics using ROUGE-L for coverage and G-Eval for factual accuracy.
