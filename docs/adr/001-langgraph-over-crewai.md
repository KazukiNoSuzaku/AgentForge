# ADR-001: Use LangGraph over CrewAI for Agent Orchestration

**Status:** Accepted
**Date:** 2024-12-15

## Context

AgentForge requires a framework to orchestrate five specialized agents (Planner, Researcher, Analyst, Writer, Critic) in a pipeline with one non-trivial control flow requirement: the Critic must be able to route the report back to the Writer for revision, creating a bounded feedback loop.

Key requirements:

1. **Conditional routing** — The Critic→Writer back-edge must be a first-class construct, not a hack layered on top of sequential execution.
2. **Bounded loops** — The revision cycle must be capped (default: 2 iterations) to prevent runaway LLM costs and infinite loops.
3. **Explicit state management** — Agents accumulate state (research findings grow, thinking steps append) while other fields get overwritten. The framework must support both patterns cleanly.
4. **Async-native execution** — The Researcher runs parallel searches across three MCP servers via `asyncio.gather`. The framework must not fight async.
5. **Streaming** — The Streamlit UI needs to update agent status cards as each node completes, which requires node-level streaming from the graph.

## Decision

Use **LangGraph** (v0.2+) as the orchestration framework, modeling the pipeline as a `StateGraph` with typed state and conditional edges.

The core pattern:

```python
from langgraph.graph import END, START, StateGraph

workflow = StateGraph(ResearchState)

workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("writer", writer_node)
workflow.add_node("critic", critic_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "writer")
workflow.add_edge("writer", "critic")

# Conditional back-edge: the revision loop
workflow.add_conditional_edges(
    "critic",
    should_revise,
    {
        "revise": "writer",
        "finish": END,
    },
)
```

The routing function reads the Critic's output from state:

```python
def should_revise(state: ResearchState) -> Literal["revise", "finish"]:
    feedback = state.get("critic_feedback")
    if feedback is None:
        return "finish"
    if feedback.requires_revision:
        return "revise"
    return "finish"
```

Loop bounding is enforced in the Critic agent itself — it sets `requires_revision=False` when `revision_count >= max_revision_loops`, so the routing function stays simple and the policy lives with the agent that owns the decision.

## Alternatives Considered

### CrewAI

- **Pros:** Higher-level API, less boilerplate, built-in role/goal/backstory abstractions, growing community.
- **Cons:** No native support for conditional back-edges between agents. Loops require workarounds (callback-based retries or wrapping in a manual while-loop). State management is implicit — you pass context between agents as strings, not typed dicts. Streaming is limited to final output, not per-node. The framework makes assumptions about agent autonomy (tool selection, self-delegation) that conflict with our deterministic pipeline topology.
- **Why not:** The revision loop is the single most important architectural feature of AgentForge. In CrewAI, it would be a bolted-on pattern rather than a native graph edge. We tried prototyping the Critic→Writer loop in CrewAI and the resulting code was fragile — the "sequential process" model doesn't accommodate cycles.

### AutoGen (Microsoft)

- **Pros:** Strong support for multi-agent conversation, built-in group chat patterns, Microsoft backing.
- **Cons:** Designed around conversational agent interaction (agents talking to each other), not structured data pipelines. State is conversational context, not a typed dictionary. The group chat model adds unpredictability — agents can interject or redirect conversation — which is the opposite of what we want for a deterministic research pipeline. Async support was immature at evaluation time (late 2024).
- **Why not:** AutoGen optimizes for emergent multi-agent behavior. AgentForge needs a predictable state machine. Wrong abstraction for the problem.

### Raw LangChain (LCEL chains)

- **Pros:** Maximum flexibility, no framework overhead, familiar to anyone who knows LangChain.
- **Cons:** No graph abstraction — you'd implement the state machine manually with while-loops and if-statements. State accumulation (appending to lists vs. overwriting fields) requires manual reducer logic. No built-in streaming at the node level. As the pipeline grows, the manual orchestration code becomes the most complex and error-prone part of the system.
- **Why not:** LangGraph exists specifically because raw LCEL doesn't handle cycles and state management well. Using raw chains would mean reimplementing half of LangGraph poorly.

## Consequences

### Positive

- **Conditional edges are a first-class primitive.** `add_conditional_edges` makes the Critic→Writer loop a three-line declaration, not a fragile control flow hack.
- **TypedDict state with annotated reducers.** `Annotated[List[ResearchFinding], operator.add]` means accumulated fields grow automatically across node updates. No manual list-merging.
- **Node-level streaming.** `graph.stream()` yields `{node_name: node_output}` dicts as each node completes, which maps directly to Streamlit UI updates.
- **Compiled graph singleton.** The graph compiles once and is reused across runs — no repeated setup cost.
- **Recursion limit as safety net.** LangGraph's `recursion_limit` config provides a hard ceiling on graph cycles independent of our application-level `max_revision_loops`.

### Negative

- **Steeper learning curve.** LangGraph's `StateGraph` + `TypedDict` + `Annotated` reducers pattern is not obvious to developers unfamiliar with LangGraph. The mental model (each node returns a *state delta* that gets merged, not the full state) is a common stumbling point.
- **Tighter coupling to LangChain ecosystem.** LangGraph depends on `langchain_core`, which means we inherit its message types (`HumanMessage`, `SystemMessage`), model abstractions (`BaseChatModel`), and release cadence. A breaking change in `langchain_core` affects us even though we don't use most of its features.
- **Testing requires patching at the graph module's namespace.** Because `build_graph()` imports node functions via `from src.agents.X import Y`, test patches must target `src.graph.planner_node`, not `src.agents.planner.planner_node`. This is a Python import binding issue, but LangGraph's compilation step makes it non-obvious.

## References

- [LangGraph documentation — StateGraph](https://langchain-ai.github.io/langgraph/)
- [LangGraph conditional edges](https://langchain-ai.github.io/langgraph/how-tos/branching/)
- [CrewAI documentation](https://docs.crewai.com/)
- [AutoGen documentation](https://microsoft.github.io/autogen/)
