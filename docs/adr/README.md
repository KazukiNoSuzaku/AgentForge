# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for AgentForge. ADRs document significant technical decisions made during the design and development of this system, including the context that led to each decision, the alternatives we considered, and the trade-offs we accepted.

## Why ADRs?

Multi-agent systems involve compounding architectural choices — orchestration framework, communication protocol, state management, quality control strategy, LLM provider abstraction. Each decision constrains or enables future ones. ADRs capture the *reasoning* behind these decisions so that contributors (and future-us) can understand not just what the system does, but why it's built the way it is.

## Index

| # | Title | Status | Date |
|---|-------|--------|------|
| [001](001-langgraph-over-crewai.md) | Use LangGraph over CrewAI for Agent Orchestration | Accepted | 2024-12-15 |
| [002](002-mcp-protocol-for-tools.md) | Use Model Context Protocol (MCP) for Tool Integration | Accepted | 2024-12-18 |
| [003](003-critic-revision-loop.md) | Implement Bounded Critic Revision Loop | Accepted | 2024-12-20 |
| [004](004-pydantic-state-schema.md) | Use Pydantic Models for Agent State and Outputs | Accepted | 2024-12-16 |
| [005](005-anthropic-primary-openai-fallback.md) | Claude as Primary LLM with OpenAI Fallback | Accepted | 2024-12-22 |

## Template

When adding a new ADR, copy this template:

```markdown
# ADR-NNN: Title

**Status:** Proposed | Accepted | Deprecated | Superseded by [ADR-NNN](NNN-title.md)
**Date:** YYYY-MM-DD

## Context

What is the problem or force driving this decision? What constraints exist?

## Decision

What did we decide, and why this option over the alternatives?

## Alternatives Considered

### Alternative A
- **Pros:** ...
- **Cons:** ...
- **Why not:** ...

### Alternative B
- **Pros:** ...
- **Cons:** ...
- **Why not:** ...

## Consequences

### Positive
- ...

### Negative
- ...

## References
- [Link](url) — description
```

## Conventions

- Number ADRs sequentially: `001`, `002`, etc.
- Use kebab-case filenames: `001-short-descriptive-title.md`
- An ADR is never deleted — if a decision is reversed, write a new ADR that supersedes it and update the original's status
- Keep context and consequences honest. If a decision has downsides, say so.
