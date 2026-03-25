# ADR-003: Implement Bounded Critic Revision Loop

**Status:** Accepted
**Date:** 2024-12-20

## Context

LLMs hallucinate. Specifically, in research report generation, they:

- Invent citations that look plausible but don't correspond to real sources
- Misattribute claims to the wrong source
- Omit sub-questions that were part of the research plan
- Draw conclusions that don't follow from the evidence presented
- Present one-sided analysis of genuinely contested topics

A single-pass pipeline (Planner → Researcher → Analyst → Writer → done) produces reports with these issues roughly 40–60% of the time in our testing. The question is how to catch and fix them before delivery.

We need a quality control mechanism that:

1. **Evaluates reports across multiple dimensions**, not just "is this good?"
2. **Produces actionable feedback** that the Writer can use to improve the draft
3. **Terminates** — the loop must be bounded to prevent runaway LLM cost
4. **Degrades gracefully** — if the Critic itself fails, the pipeline should still produce output

## Decision

Implement a **Critic agent** that evaluates the Writer's draft across five weighted dimensions, producing a 0–1 quality score. If the score falls below a configurable threshold (default: 0.75) and the revision count is under the maximum (default: 2), the report routes back to the Writer with structured feedback. Otherwise, it's approved as final.

### Quality Dimensions

| Dimension | Weight | What It Catches |
|-----------|--------|-----------------|
| **Citation accuracy** | 30% | Hallucinated citations, misattributed claims, invented URLs |
| **Completeness** | 25% | Missing sub-questions, unacknowledged gaps |
| **Logical consistency** | 20% | Internal contradictions, non-sequiturs, unsupported conclusions |
| **Balanced perspective** | 15% | One-sided analysis, missing stakeholder viewpoints |
| **Factual grounding** | 10% | Claims not traceable to provided research findings |

Citation accuracy gets the highest weight because hallucinated references are the most damaging failure mode — they're hard for users to detect and undermine the report's credibility.

### Score Rubric

- **0.90–1.00:** Publish-ready. No significant issues.
- **0.75–0.89:** Minor issues. Approved with feedback attached for context.
- **0.50–0.74:** Significant issues. Triggers revision.
- **0.00–0.49:** Major problems. Triggers revision with detailed remediation.

### Loop Bounding

The Critic enforces the revision cap internally:

```python
# In CriticAgent.critique()
if revision_count >= self._config.max_revision_loops:
    feedback = CriticFeedback(
        quality_score=feedback.quality_score,
        unsupported_claims=feedback.unsupported_claims,
        # ... preserve all feedback fields ...
        requires_revision=False,  # Force approval
        feedback_summary=feedback.feedback_summary,
        strengths=feedback.strengths,
    )
```

This means the routing function (`should_revise`) stays simple — it just reads `requires_revision` — and the bounding policy lives with the agent that owns the quality decision.

### Automatic Citation Validation

Before the LLM evaluates the report, a deterministic pre-check runs:

```python
citation_map = build_citation_map(findings)
citation_issues_auto = validate_citations(report, citation_map)
```

This catches citation numbers that don't correspond to any source in the findings, without consuming LLM tokens. These issues are injected into the Critic's prompt so the LLM can incorporate them into its overall assessment.

### Graceful Degradation

If the Critic node throws an exception, the pipeline approves the draft and continues:

```python
except Exception as exc:
    # On Critic failure, approve the draft to avoid blocking
    return {
        "critic_feedback": None,
        "final_report": draft,  # Pass through as-is
        "errors": [error_msg],
    }
```

A report with no quality gate is better than no report at all.

## Alternatives Considered

### Single-Pass Generation (No Critic)

- **Pros:** Simpler pipeline, lower latency, lower cost (no Critic LLM call, no revision Writer calls).
- **Cons:** Hallucinated citations reach the user. Missing sub-questions go undetected. In our testing, single-pass reports had a ~45% rate of at least one significant citation or completeness issue.
- **Why not:** The quality improvement from even one revision cycle is substantial. Reports that go through the Critic score 15–20 points higher on our evaluation rubric. The cost of one additional LLM call (~$0.03–0.05) is negligible compared to the value of a credible report.

### Human-in-the-Loop Review

- **Pros:** Highest quality. Catches issues LLMs miss. Adds domain expertise.
- **Cons:** Breaks the automated pipeline model. Introduces latency (minutes to hours vs. seconds). Not viable for the intended use case (fast, on-demand research synthesis). Could be added as an optional post-pipeline step, but can't replace automated quality gating.
- **Why not:** AgentForge is designed for sub-2-minute end-to-end execution. A human review step contradicts the core value proposition. The Critic is the automated approximation of a human reviewer.

### External Fact-Checking API

- **Pros:** Deterministic, reproducible results. No LLM hallucination risk in the quality check itself.
- **Cons:** No general-purpose fact-checking API exists that handles arbitrary research claims across all domains. Existing services (Google Fact Check Tools API, ClaimBuster) focus on political claims and news. Building a domain-agnostic fact-checker is a harder problem than the one we're solving.
- **Why not:** The technology doesn't exist for our use case. The Critic's LLM-based evaluation, supplemented by deterministic citation validation, is the pragmatic middle ground.

### Higher Revision Limit (5+ Cycles)

- **Pros:** More opportunities to fix issues. Could converge on higher quality.
- **Cons:** In practice, most improvement happens in the first revision. The second revision catches diminishing returns — often the Writer fixes the flagged issues but introduces new minor ones. Beyond two cycles, the LLM tends to oscillate (fixing A, breaking B, fixing B, breaking A). Cost scales linearly with revision count.
- **Why not:** Empirically, 2 revisions captures ~90% of the quality improvement available. The default cap of 2 is configurable (`MAX_REVISION_LOOPS`) for users who want to experiment, but the marginal return on revisions 3+ is low.

## Consequences

### Positive

- **Measurable quality improvement.** Reports score 0.75–0.90 on the Critic's rubric after revision, vs. 0.55–0.70 on first drafts. The biggest gains are in citation accuracy and completeness.
- **Structured, actionable feedback.** The `CriticFeedback` schema (`unsupported_claims`, `missing_perspectives`, `logical_gaps`, `citation_issues`) gives the Writer specific items to fix, not vague instructions.
- **Transparent quality signal.** The `quality_score` is included in pipeline metrics and the final report, giving users visibility into how the system assessed its own output.
- **Deterministic + LLM hybrid.** The automatic citation validation catches mechanical citation errors without LLM cost; the LLM catches semantic issues (logical gaps, missing perspectives) that rules can't.
- **Bounded cost.** Worst case: 2 extra Writer calls + 2 extra Critic calls ≈ $0.10–0.15 additional cost. Typical case: 0–1 revisions.

### Negative

- **The Critic is itself an LLM.** It can hallucinate issues that don't exist, or miss real issues. The quality score is an estimate, not ground truth. We partially mitigate this with the deterministic citation pre-check, but semantic evaluation is inherently approximate.
- **Revision increases latency.** Each revision adds ~15–25 seconds (Writer + Critic calls). A report that needs 2 revisions takes ~40–50 seconds longer than a single-pass report.
- **Feedback quality depends on prompt engineering.** The Critic's system prompt (evaluation criteria, rubric, output format) is a critical piece of the system that requires maintenance as we observe failure modes. It's not a "set and forget" component.
- **No memory across runs.** The Critic evaluates each report independently. If the Writer consistently makes the same mistake (e.g., always omitting a "limitations" section), the Critic catches it each time but doesn't learn to preempt it.

## References

- `src/agents/critic.py` — Critic agent implementation and system prompt
- `src/utils/citations.py` — Deterministic citation validation
- `src/graph.py` — `should_revise()` conditional edge function
- `src/config.py` — `quality_threshold` and `max_revision_loops` config fields
