# ADR-004: Use Pydantic Models for Agent State and Outputs

**Status:** Accepted
**Date:** 2024-12-16

## Context

In a multi-agent pipeline, each agent produces structured output that the next agent consumes. Without a schema contract at agent boundaries, failures manifest as:

- **Silent data corruption** — An agent receives a dict with a misspelled key, silently gets `None`, and produces degraded output downstream.
- **Ambiguous typing** — Is `confidence` a float between 0 and 1, or an integer percentage? Is `sources` a list of URLs or a list of dicts? Every agent must guess or check at runtime.
- **Untestable contracts** — Without a schema, you can't write unit tests that verify an agent's output structure independently of its content.
- **Opaque errors** — When something goes wrong three agents downstream, tracing the root cause back to a malformed output from the first agent is painful.

The state also has two different update semantics. Some fields accumulate across nodes (e.g., `research_findings` grows as the Researcher processes each sub-question). Others are overwritten (e.g., `draft_report` is replaced each time the Writer runs). The schema system must support both patterns.

## Decision

Use **Pydantic v2** `BaseModel` classes for all agent outputs and **`TypedDict` with `Annotated` reducers** for the LangGraph shared state.

### Agent Output Schemas

Each pipeline stage has explicit Pydantic models. Examples from `src/models/schemas.py`:

**Planner output:**
```python
class SubQuestion(BaseModel):
    id: str = Field(description="Unique identifier, e.g. 'sq_1'")
    question: str = Field(description="The sub-question to research")
    priority: int = Field(default=1, ge=1, le=5)
    search_terms: List[str] = Field(default_factory=list)
    rationale: str = Field(default="")

class ResearchPlan(BaseModel):
    sub_questions: List[SubQuestion]
    research_approach: str
    estimated_complexity: Literal["low", "medium", "high"]
```

**Researcher output:**
```python
class ResearchFinding(BaseModel):
    sub_question_id: str
    sub_question: str
    source_type: SourceType  # Enum: brave, arxiv, wikipedia
    title: str
    url: str
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    key_insights: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

**Analyst output:**
```python
class AnalysisResult(BaseModel):
    patterns: List[str]
    contradictions: List[str]
    gaps: List[str]
    claims: List[Claim]  # Each Claim has id, statement, confidence, sources
    confidence_summary: str
```

**Critic output:**
```python
class CriticFeedback(BaseModel):
    quality_score: float = Field(ge=0.0, le=1.0)
    unsupported_claims: List[str] = Field(default_factory=list)
    missing_perspectives: List[str] = Field(default_factory=list)
    logical_gaps: List[str] = Field(default_factory=list)
    citation_issues: List[str] = Field(default_factory=list)
    requires_revision: bool
    feedback_summary: str
    strengths: List[str] = Field(default_factory=list)
```

### LangGraph State Schema

The shared state uses `TypedDict` with `Annotated` fields for accumulation semantics:

```python
class ResearchState(TypedDict, total=False):
    query: str
    sub_questions: List[SubQuestion]
    research_findings: Annotated[List[ResearchFinding], operator.add]  # accumulates
    analysis: Optional[AnalysisResult]
    draft_report: str
    revision_count: int
    critic_feedback: Optional[CriticFeedback]
    final_report: str
    thinking_steps: Annotated[List[ThinkingStep], operator.add]  # accumulates
    errors: Annotated[List[str], operator.add]  # accumulates
```

Fields annotated with `operator.add` are accumulated: when a node returns `{"research_findings": [new_finding]}`, LangGraph *appends* to the existing list rather than replacing it. All other fields are overwritten by the latest node output.

### Structured LLM Output

Pydantic models are also used for LLM-driven structured output via LangChain's `.with_structured_output()`:

```python
structured_llm = self.primary.with_structured_output(ResearchPlan)
plan = await structured_llm.ainvoke(messages)
# plan is a validated ResearchPlan instance, not a raw dict
```

This leverages the LLM provider's tool-calling mechanism to produce JSON that's parsed and validated against the Pydantic schema automatically.

## Alternatives Considered

### Plain Dictionaries

- **Pros:** No schema boilerplate. Fastest to write initially. Maximum flexibility.
- **Cons:** No validation — a missing field, wrong type, or out-of-range value passes silently until something breaks downstream. Every consumer must defensively check types and handle missing keys. Refactoring is dangerous because nothing enforces the interface contract. IDE support (autocomplete, type hints) doesn't work.
- **Why not:** The pipeline has five agents reading each other's output. With dicts, each boundary is an implicit contract that exists only in documentation (if that). The debugging cost of a single silent-`None` bug in a multi-agent system exceeds the cost of writing Pydantic models for all schemas.

### Dataclasses

- **Pros:** Built into Python. Lighter than Pydantic. Good IDE support. Familiar to most Python developers.
- **Cons:** No built-in validation. `Field(ge=0.0, le=1.0)` constraints don't exist — you'd write custom `__post_init__` validators, which is reimplementing Pydantic poorly. No `.with_structured_output()` integration in LangChain — the structured output pipeline specifically requires Pydantic models. Serialization/deserialization requires manual `asdict()` / `from_dict()` boilerplate.
- **Why not:** The two features we rely on most — field-level validation constraints and LangChain structured output — are Pydantic-specific. Dataclasses would require us to write validation and serialization code that Pydantic provides out of the box.

### Pydantic v1

- **Pros:** More widely used in existing LangChain tutorials and examples (as of late 2024).
- **Cons:** v1 is in maintenance mode. LangChain's core has migrated to v2 compatibility. v2 is significantly faster (Rust-based validation core), has cleaner `model_config` over `class Config`, and `Field` replaces `validator` decorators with `field_validator`. Starting a new project on v1 means an inevitable migration later.
- **Why not:** No reason to start on a deprecated version. v2 works with LangChain 0.2+ and all our dependencies.

## Consequences

### Positive

- **Fail-fast at agent boundaries.** If the Planner produces a `SubQuestion` with `priority=0` (below `ge=1`), Pydantic raises a `ValidationError` immediately — not three agents later when the Researcher tries to sort by priority.
- **Self-documenting interfaces.** The `Field(description=...)` annotations serve as inline documentation. `schemas.py` is the single source of truth for all data shapes in the system.
- **LLM structured output works out of the box.** `.with_structured_output(ResearchPlan)` returns a validated `ResearchPlan` instance. No manual JSON parsing or schema enforcement.
- **Typed accumulation in LangGraph.** `Annotated[List[ResearchFinding], operator.add]` provides clean append-semantics for fields that grow across nodes, while overwrite-semantics apply to everything else. The TypedDict + Pydantic combination gives us both the LangGraph state contract and the per-field type safety.
- **Testability.** Unit tests construct Pydantic models directly to create fixtures. The schema enforces that test data is realistic — you can't accidentally create a `Claim` with `confidence=5.0` in a test.

### Negative

- **Schema maintenance cost.** Every change to agent output requires updating the corresponding Pydantic model, which may trigger changes in downstream agents. In a five-agent pipeline, a schema change can cascade.
- **Verbose for simple cases.** `SubQuestion(id="sq_1", question="...", priority=1, search_terms=[], rationale="")` is more boilerplate than `{"id": "sq_1", "question": "..."}`. Default factories help, but the models are still more code than raw dicts.
- **Pydantic v2 migration friction.** Some LangChain community packages still use v1 patterns. Mixing v1 and v2 models in the same codebase causes subtle serialization bugs. We've avoided this by using v2 exclusively, but it limits which third-party LangChain tools we can use.
- **TypedDict + Pydantic is two type systems.** The LangGraph state uses `TypedDict` (for LangGraph compatibility), but agent outputs use Pydantic `BaseModel`. The state *contains* Pydantic models but is itself a TypedDict. This dual-schema pattern works but isn't immediately obvious.

## References

- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/)
- [LangChain structured output](https://python.langchain.com/docs/how_to/structured_output/)
- [LangGraph state management](https://langchain-ai.github.io/langgraph/concepts/low_level/#state)
- `src/models/schemas.py` — All Pydantic models
- `src/state.py` — LangGraph TypedDict state with annotated reducers
