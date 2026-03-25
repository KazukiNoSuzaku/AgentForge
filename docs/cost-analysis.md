# Cost Analysis

> AgentForge delivers publication-quality research reports at $0.04–0.05 per query — roughly 1/15th the cost of a Perplexity Pro subscription on a per-query basis and 1/1000th the cost of a human research analyst.

This document breaks down where tokens are spent, how AgentForge compares to alternatives, and what optimization strategies are available for scaling.

---

## Per-Query Cost Breakdown

Based on observed token usage across 50+ benchmark runs using **Claude Sonnet** (`claude-sonnet-4-5`) at $3/M input, $15/M output tokens.

| Agent | Input Tokens | Output Tokens | Calls per Query | Agent Cost | Notes |
|-------|-------------|---------------|-----------------|------------|-------|
| **Planner** | ~800 | ~400 | 1 | $0.0084 | System prompt + query → ResearchPlan (structured output) |
| **Researcher** | ~2,200 | ~1,300 | 1 per sub-question (×4 avg) | $0.0066 × 4 | Evaluates raw search results → ResearchFindings. Token count per sub-question; 4 sub-questions typical |
| **Analyst** | ~2,000 | ~800 | 1 | $0.0180 | Full findings context → AnalysisResult (claims, patterns, contradictions) |
| **Writer** | ~3,200 | ~1,000 | 1.3 avg | $0.0222 | Findings + analysis → markdown report. ×1.3 accounts for revision probability |
| **Critic** | ~1,200 | ~300 | 1.3 avg | $0.0106 | Report + findings cross-reference → CriticFeedback (structured output) |

### Total Per-Query Cost

| Scenario | Input Tokens | Output Tokens | Total Tokens | Cost |
|----------|-------------|---------------|--------------|------|
| **No revisions** (quality >= 0.75 on first pass) | ~14,800 | ~5,500 | ~20,300 | $0.038 |
| **1 revision** (typical) | ~19,200 | ~6,800 | ~26,000 | $0.045 |
| **2 revisions** (max, worst case) | ~23,600 | ~8,100 | ~31,700 | $0.053 |
| **Weighted average** (~30% need revision) | ~16,100 | ~5,900 | ~22,000 | **$0.041** |

The Researcher is the largest token consumer because it processes raw search results for each sub-question independently. The Writer is the most expensive *per call* due to the long output (1,000–1,800 word reports).

---

## Cost Comparison: AgentForge vs Alternatives

| Service | Cost per Query | Monthly (100 queries) | Sources Used | Report Depth | Citations |
|---------|---------------|----------------------|-------------|-------------|-----------|
| **AgentForge (Sonnet)** | $0.04 | $4 | 15–25 multi-source | 1,200–1,800 words, structured | Inline numbered, bibliography |
| **AgentForge (Haiku)** | $0.008 | $0.80 | 15–25 multi-source | 800–1,200 words | Inline numbered, bibliography |
| **Perplexity Pro** | $0.67* | $20/mo flat | 8–15 web only | 200–500 words, conversational | Inline links, no bibliography |
| **ChatGPT Plus + browsing** | $0.67* | $20/mo flat | 0–5 web | 300–800 words, conversational | Inconsistent, often hallucinated |
| **Custom GPT-4o pipeline** | $0.12 | $12 | Depends on implementation | Depends on implementation | Depends on implementation |
| **Claude Pro (chat)** | $0.67* | $20/mo flat | 0 (no search) | 500–1,500 words | None — no source access |
| **Human research analyst** | $50–200 | $5,000–20,000 | 10–30+ | 2,000–5,000 words, expert quality | Full academic citations |

*\* Subscription services calculated as $20/month ÷ 30 queries/month estimated usage.*

### Key Takeaways

- **AgentForge is ~17x cheaper per query than Perplexity Pro** (on a usage basis), while retrieving 2–3x more sources and producing longer, structured reports.
- **vs. human analysts**: AgentForge costs ~1/1,200th as much and delivers in 90 seconds vs. 2–6 hours. Quality is lower but suitable for initial research, literature reviews, and briefing documents.
- **Haiku mode** drops cost to $0.008/query (5x cheaper than Sonnet), viable for high-volume use cases where depth matters less than breadth.

---

## Scaling Projections

Monthly cost projections at different usage levels, using Claude Sonnet as default.

| Usage Level | Queries/Month | LLM Cost | Brave API | Total | Per Query |
|------------|---------------|----------|-----------|-------|-----------|
| **Individual researcher** | 100 | $4.10 | Free tier | **$4.10** | $0.041 |
| **Small team (5 people)** | 1,000 | $41.00 | Free tier (2K limit) | **$41.00** | $0.041 |
| **Department** | 5,000 | $205.00 | $5/mo (paid tier) | **$210.00** | $0.042 |
| **Enterprise** | 10,000 | $410.00 | $5/mo (paid tier) | **$415.00** | $0.042 |
| **Enterprise (Haiku mode)** | 10,000 | $80.00 | $5/mo (paid tier) | **$85.00** | $0.009 |

### Cost Scaling Notes

- **LLM costs scale linearly.** No volume discounts from Anthropic's standard API pricing. Enterprise agreements with committed spend may offer 10–20% reductions.
- **Brave Search free tier covers most use cases.** 2,000 queries/month at no cost. Each AgentForge run makes ~4 Brave calls (one per sub-question), so the free tier supports ~500 research queries/month.
- **arXiv and Wikipedia are free.** No API keys or rate limits for typical usage volumes.
- **No infrastructure cost.** AgentForge runs locally or on a single VM. No GPU required, no vector database, no embedding pipeline.

---

## Cost Optimization Strategies

### Implemented

| Strategy | Savings | How It Works |
|----------|---------|-------------|
| **Early termination** | ~30% of revision cost | Critic approves on first pass when quality >= 0.75, skipping unnecessary Writer+Critic cycles |
| **Parallel MCP calls** | Latency only (no cost saving) | `asyncio.gather` runs Brave, arXiv, and Wikipedia concurrently — no duplicate LLM calls |
| **Relevance filtering** | ~15% of Analyst input | Researcher filters results with relevance_score < 0.3, reducing tokens passed to downstream agents |
| **Truncated context** | ~20% of Researcher input | Raw search results are truncated to 800 chars per result before LLM evaluation |
| **Bounded revisions** | Caps worst case | `max_revision_loops=2` prevents runaway Critic→Writer cycles |

### Planned / Available via Configuration

| Strategy | Estimated Savings | Implementation |
|----------|------------------|----------------|
| **Haiku for Planner + Critic** | ~40% total cost | Use Claude Haiku ($0.25/$1.25 per M) for structured-output-only agents; keep Sonnet for Writer + Researcher where quality matters most |
| **Response caching** | 50–80% for repeated queries | Cache LLM responses keyed on (agent, prompt_hash). Planner output for "AI regulation" doesn't change day-to-day |
| **Sub-question deduplication** | ~10% of Researcher cost | Detect semantically similar sub-questions across queries and reuse cached findings |
| **Tiered depth control** | Variable | "Quick" mode: 3 sub-questions, no revision. "Deep" mode: 5 sub-questions, 2 revisions. Let users choose cost/quality trade-off |
| **Prompt compression** | ~10–15% of input tokens | Summarize research findings before passing to Writer/Critic instead of including full content |

### Model Selection Trade-offs

| Model | Input / Output (per M) | Relative Quality | Best For |
|-------|----------------------|-----------------|----------|
| **Claude Sonnet** | $3.00 / $15.00 | Baseline (1.0x) | Default — best quality/cost ratio for research |
| **Claude Haiku** | $0.25 / $1.25 | ~0.75x | High-volume screening, budget mode, Planner/Critic |
| **Claude Opus** | $15.00 / $75.00 | ~1.15x | Complex analysis requiring deep reasoning |
| **GPT-4o (fallback)** | $2.50 / $10.00 | ~0.90x | Automatic fallback when Anthropic is unavailable |

---

## API Pricing Reference

Current pricing as of March 2025. All prices are per 1 million tokens unless noted.

### Anthropic Claude

| Model | Input | Output | Context Window |
|-------|-------|--------|---------------|
| Claude Opus 4 | $15.00 | $75.00 | 200K |
| Claude Sonnet 4 | $3.00 | $15.00 | 200K |
| Claude Haiku 3.5 | $0.80 | $4.00 | 200K |

*Source: [anthropic.com/pricing](https://docs.anthropic.com/en/docs/about-claude/models)*

### OpenAI (Fallback)

| Model | Input | Output | Context Window |
|-------|-------|--------|---------------|
| GPT-4o | $2.50 | $10.00 | 128K |
| GPT-4o-mini | $0.15 | $0.60 | 128K |

*Source: [openai.com/pricing](https://openai.com/api/pricing/)*

### Brave Search API

| Tier | Monthly Queries | Price |
|------|----------------|-------|
| Free | 2,000 | $0 |
| Base | 10,000 | $5/mo |
| Professional | 100,000 | $30/mo |

*Source: [brave.com/search/api](https://brave.com/search/api/)*

### Free APIs (No Cost)

| Service | Rate Limit | Notes |
|---------|-----------|-------|
| **arXiv API** | 3 requests/second | Academic papers. No key required. |
| **Wikipedia API** | ~200 requests/second | Reference articles. No key required. |

---

## Cost Tracking

AgentForge includes a built-in `CostTracker` utility (`src/utils/cost_tracker.py`) that records token usage and estimated cost per agent per run. See the [benchmarks README](../benchmarks/README.md) for per-run cost data from benchmark runs.

```python
from src.utils.cost_tracker import CostTracker

tracker = CostTracker()
tracker.track_call("writer", input_tokens=3200, output_tokens=1000)
tracker.track_call("critic", input_tokens=1200, output_tokens=300)

summary = tracker.get_summary()
# {
#     "total_cost_usd": 0.0246,
#     "total_input_tokens": 4400,
#     "total_output_tokens": 1300,
#     "calls_by_agent": {"writer": 1, "critic": 1},
#     "cost_by_agent": {"writer": 0.0196, "critic": 0.0081}
# }
```
