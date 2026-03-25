# ADR-002: Use Model Context Protocol (MCP) for Tool Integration

**Status:** Accepted
**Date:** 2024-12-18

## Context

The Researcher agent needs to query three external data sources: Brave Search (web), arXiv (academic papers), and Wikipedia (reference articles). These integrations have different APIs, authentication mechanisms, and response formats. We need an integration pattern that:

1. **Isolates tool logic from agent logic** — The Researcher agent should evaluate and synthesize results, not manage HTTP clients, rate limits, or response parsing for three different APIs.
2. **Supports adding new sources without modifying agent code** — Adding a Semantic Scholar or Google Scholar source should be a new server, not a change to `researcher.py`.
3. **Runs tools as separate processes** — Tool failures (crashes, timeouts, memory leaks) should not take down the main pipeline process.
4. **Works with the emerging AI tool ecosystem** — The protocol should be something that other tools and clients can interoperate with, not a proprietary format.

## Decision

Use **Model Context Protocol (MCP)** with FastMCP servers communicating over **stdio transport**. Each data source is implemented as a standalone MCP server script (e.g., `src/mcp_servers/brave_search.py`). The Researcher agent invokes tools via the MCP client SDK:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(server_script)],
    env={**os.environ},
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool(tool_name, arguments)
```

Each MCP server is a self-contained Python script that:
- Registers its tools via `@mcp.tool()` decorators
- Handles its own API authentication (reads keys from environment)
- Returns structured JSON responses
- Runs as a subprocess — spawned per call, then exits

The stdio transport was chosen over HTTP/SSE because:
- No port management or service discovery needed
- Subprocess lifecycle is tied to the tool call — no orphaned servers
- Environment variables (API keys) pass through naturally
- Simpler deployment — no container networking or reverse proxies

## Alternatives Considered

### Direct API Calls in Agent Code

- **Pros:** Simplest possible approach. No abstraction layer. One fewer dependency.
- **Cons:** The Researcher agent becomes a 500+ line file managing `httpx` clients, retry logic, rate limiting, and response parsing for three different APIs. Adding a fourth source means modifying the agent. API-specific error handling pollutes the research evaluation logic. Testing requires mocking three different HTTP APIs inline.
- **Why not:** Mixing I/O orchestration with LLM-driven evaluation produces code that's hard to test, hard to extend, and hard to read. The Researcher's job is to *evaluate* results, not to *fetch* them.

### LangChain Tool Abstractions

- **Pros:** LangChain has built-in tool wrappers for many APIs (`BraveSearchTool`, `ArxivLoader`, `WikipediaRetriever`). Tight integration with the LLM invocation pipeline — tools can be bound to models via `.bind_tools()`.
- **Cons:** LangChain tool abstractions run in-process. A tool that hangs blocks the agent's event loop. Tool wrappers vary in quality and maintenance — the arXiv wrapper, for instance, had stale dependencies at evaluation time. The `.bind_tools()` pattern is designed for the model to *choose* which tools to call, but our Researcher always calls all three sources — tool selection adds latency and unpredictability for no benefit. Vendor lock-in to LangChain's tool interface.
- **Why not:** We don't need LLM-driven tool selection (the Researcher's search strategy is deterministic), and in-process execution eliminates the fault isolation we want.

### Custom Protocol (e.g., gRPC, REST microservices)

- **Pros:** Full control over the interface. Could optimize for batch queries or streaming results.
- **Cons:** Inventing a protocol means writing and maintaining both client and server sides, plus serialization, versioning, and documentation. gRPC adds protobuf compilation. REST microservices need a service registry and port management. All of this for what are fundamentally simple request-response tool calls.
- **Why not:** The engineering overhead is disproportionate to the problem. MCP provides a standard protocol with existing client/server SDKs. Building a custom protocol would be over-engineering.

## Consequences

### Positive

- **Clean separation of concerns.** Each MCP server is a single-file script (~80–120 lines) that does one thing: call an API and return structured results. The Researcher agent is purely evaluation logic.
- **Process isolation.** A Brave Search timeout or arXiv API error kills a subprocess, not the pipeline. `asyncio.gather` with `return_exceptions=True` collects results from successful sources and logs failures.
- **Ecosystem compatibility.** MCP servers work with any MCP client — Claude Desktop, other agent frameworks, or standalone test harnesses. The same `brave_search.py` server could be used outside AgentForge without modification.
- **Simple testing.** MCP tool calls return JSON strings. Tests mock `call_mcp_tool` at a single point rather than patching three different HTTP clients. Integration tests can run real MCP servers against test fixtures.
- **Easy to extend.** Adding a Google Scholar source means writing a new `google_scholar.py` MCP server and adding one `search_google_scholar()` function to the Researcher. No changes to the agent's evaluation logic.

### Negative

- **Subprocess overhead per call.** Each tool invocation spawns a Python subprocess, initializes the MCP session, makes one call, and exits. For three sources × five sub-questions, that's 15 subprocess spawns per run. In practice, the overhead is ~200–400ms per spawn, which is small relative to the API call latency (~1–3s), but it's not zero.
- **No connection pooling.** Unlike an HTTP client that reuses connections, stdio transport creates a fresh process each time. For high-throughput scenarios (not our current use case), this would need optimization — likely switching to a long-running server with HTTP/SSE transport.
- **Debugging is harder.** Subprocess output goes to stderr, which requires explicit log capture. A crash in an MCP server produces a cryptic "MCP tool failed" log message unless you run the server standalone to reproduce the issue.
- **MCP is still maturing.** The protocol spec and SDK are evolving. Breaking changes in `mcp` package updates are possible, though the core `call_tool` interface is stable.

## References

- [Model Context Protocol specification](https://modelcontextprotocol.io/)
- [MCP Python SDK (`mcp`)](https://github.com/modelcontextprotocol/python-sdk)
- [FastMCP server pattern](https://github.com/modelcontextprotocol/python-sdk#fastmcp)
- AgentForge MCP servers: `src/mcp_servers/brave_search.py`, `arxiv_search.py`, `wikipedia.py`
