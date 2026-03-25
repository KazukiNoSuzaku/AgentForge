"""
AgentForge Streamlit Frontend

Real-time research interface showing:
  - Left sidebar: Agent status panel (idle / active / done / error)
  - Main panel: Live agent progress, thinking steps, and final report

Architecture note: LangGraph's synchronous stream() is run inside a thread
via asyncio so Streamlit's main thread stays responsive.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime
from queue import Empty, Queue
from typing import Any, Dict

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page Config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AgentForge",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Imports (after env load)
# ---------------------------------------------------------------------------

from src.graph import build_graph
from src.models.schemas import AgentStatus
from src.state import create_initial_state

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_SEQUENCE = ["planner", "researcher", "analyst", "writer", "critic"]

STATUS_ICONS = {
    AgentStatus.IDLE: "⭕",
    AgentStatus.ACTIVE: "🔄",
    AgentStatus.DONE: "✅",
    AgentStatus.ERROR: "❌",
}

STATUS_COLORS = {
    AgentStatus.IDLE: "#6c757d",
    AgentStatus.ACTIVE: "#0d6efd",
    AgentStatus.DONE: "#198754",
    AgentStatus.ERROR: "#dc3545",
}

AGENT_DESCRIPTIONS = {
    "planner": "Decomposes the query into sub-questions",
    "researcher": "Searches Brave, arXiv, and Wikipedia",
    "analyst": "Identifies patterns, claims, and gaps",
    "writer": "Generates the markdown report",
    "critic": "Reviews quality and triggers revision if needed",
}

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .agent-card {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 4px solid #6c757d;
    }
    .agent-card.active { border-left-color: #0d6efd; }
    .agent-card.done   { border-left-color: #198754; }
    .agent-card.error  { border-left-color: #dc3545; }
    .agent-name { font-weight: 700; font-size: 0.9rem; }
    .agent-desc { font-size: 0.78rem; color: #9aa0b0; margin-top: 2px; }
    .thinking-step {
        background: #252840;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.85rem;
        border-left: 3px solid #6c757d;
    }
    .metric-box {
        text-align: center;
        background: #1e2130;
        border-radius: 8px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    """Initialise all required session state keys with defaults."""
    defaults = {
        "running": False,
        "completed": False,
        "agent_statuses": {a: "idle" for a in AGENT_SEQUENCE},
        "thinking_steps": [],
        "final_report": None,
        "findings_count": 0,
        "claims_count": 0,
        "quality_score": None,
        "revision_count": 0,
        "errors": [],
        "current_agent": None,
        "update_queue": Queue(),
        "start_time": None,
        "sub_questions": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    """Reset all pipeline state for a new research run."""
    keys_to_reset = [
        "running",
        "completed",
        "agent_statuses",
        "thinking_steps",
        "final_report",
        "findings_count",
        "claims_count",
        "quality_score",
        "revision_count",
        "errors",
        "current_agent",
        "start_time",
        "sub_questions",
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.update_queue = Queue()
    init_session_state()


# ---------------------------------------------------------------------------
# Pipeline Runner (runs in background thread)
# ---------------------------------------------------------------------------


def run_pipeline_in_thread(query: str, queue: Queue) -> None:
    """
    Execute the LangGraph pipeline in a background thread.

    Puts node updates into the queue as they arrive, so the main
    Streamlit thread can pick them up and render UI updates.

    Args:
        query: Research question.
        queue: Thread-safe queue for passing updates to the UI thread.
    """

    async def _run():
        graph = build_graph()
        initial_state = create_initial_state(query)

        try:
            for update in graph.stream(
                initial_state,
                config={"recursion_limit": 25},
            ):
                queue.put(("update", update))
        except Exception as exc:
            queue.put(("error", str(exc)))
        finally:
            queue.put(("done", None))

    # Run the async coroutine in a new event loop within this thread
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------


def render_sidebar(agent_statuses: Dict[str, str]) -> None:
    """Render the agent status panel in the sidebar."""
    with st.sidebar:
        st.markdown("## 🤖 Agent Pipeline")
        st.markdown("---")

        for agent in AGENT_SEQUENCE:
            raw_status = agent_statuses.get(agent, "idle")
            # Map to AgentStatus enum safely
            try:
                status = AgentStatus(raw_status)
            except ValueError:
                status = AgentStatus.IDLE

            icon = STATUS_ICONS[status]
            css_class = raw_status if raw_status in ("active", "done", "error") else ""

            st.markdown(
                f"""
                <div class="agent-card {css_class}">
                    <div class="agent-name">{icon} {agent.capitalize()}</div>
                    <div class="agent-desc">{AGENT_DESCRIPTIONS[agent]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### 📊 Metrics")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Findings", st.session_state.findings_count)
            st.metric("Revisions", st.session_state.revision_count)
        with col2:
            st.metric("Claims", st.session_state.claims_count)
            if st.session_state.quality_score is not None:
                score = st.session_state.quality_score
                color = "normal" if score >= 0.75 else "inverse"
                st.metric("Quality", f"{score:.2f}", delta_color=color)

        if st.session_state.start_time and st.session_state.running:
            elapsed = time.time() - st.session_state.start_time
            st.metric("Elapsed", f"{elapsed:.0f}s")

        if st.session_state.errors:
            st.markdown("---")
            st.markdown("### ⚠️ Warnings")
            for error in st.session_state.errors[-3:]:
                st.warning(error[:150])


def render_sub_questions(sub_questions: list) -> None:
    """Render the planner's sub-questions as a styled list."""
    if sub_questions:
        with st.expander("📋 Research Sub-Questions", expanded=True):
            for sq in sub_questions:
                priority_color = "🔴" if sq.priority == 1 else "🟡" if sq.priority == 2 else "🟢"
                st.markdown(f"{priority_color} **[{sq.id}]** {sq.question}")


def render_thinking_steps(steps: list) -> None:
    """Render accumulated thinking steps in a collapsible section."""
    if steps:
        with st.expander(f"🧠 Agent Thinking Steps ({len(steps)} steps)", expanded=False):
            for step in steps:
                agent_color = {
                    "planner": "blue",
                    "researcher": "green",
                    "analyst": "orange",
                    "writer": "violet",
                    "critic": "red",
                }.get(step.agent, "gray")

                st.markdown(
                    f"""
                    <div class="thinking-step">
                        <strong style="color:{agent_color}">{step.agent.upper()}</strong>
                        → <em>{step.action}</em><br/>
                        {step.content}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def process_update(update: Dict[str, Any]) -> None:
    """
    Process a single LangGraph state update and update session state.

    Args:
        update: {node_name: state_delta} dict from graph.stream().
    """
    for node_name, node_output in update.items():
        # Mark previous agent as done, current as active
        st.session_state.current_agent = node_name

        # Update statuses
        new_statuses = node_output.get("agent_statuses", {})
        st.session_state.agent_statuses.update(new_statuses)

        # Accumulate thinking steps
        new_steps = node_output.get("thinking_steps", [])
        st.session_state.thinking_steps.extend(new_steps)

        # Accumulate errors
        new_errors = node_output.get("errors", [])
        st.session_state.errors.extend(new_errors)

        # Update metrics
        findings = node_output.get("research_findings", [])
        if findings:
            st.session_state.findings_count += len(findings)

        sub_questions = node_output.get("sub_questions", [])
        if sub_questions:
            st.session_state.sub_questions = sub_questions

        analysis = node_output.get("analysis")
        if analysis:
            st.session_state.claims_count = len(analysis.claims)

        feedback = node_output.get("critic_feedback")
        if feedback:
            st.session_state.quality_score = feedback.quality_score
            if feedback.requires_revision:
                st.session_state.revision_count += 1

        final_report = node_output.get("final_report")
        if final_report:
            st.session_state.final_report = final_report


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------


def main() -> None:
    """Main Streamlit application entry point."""
    init_session_state()

    # ------------------------------------------------------------------ #
    # Header
    # ------------------------------------------------------------------ #
    st.title("🔬 AgentForge")
    st.markdown("_A production-grade multi-agent research engine powered by LangGraph + MCP_")
    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Query Input
    # ------------------------------------------------------------------ #
    col_input, col_button = st.columns([5, 1])

    with col_input:
        query = st.text_input(
            "Research Question",
            placeholder="e.g., Compare AI regulatory frameworks in the EU, US, and China",
            disabled=st.session_state.running,
            key="query_input",
        )

    with col_button:
        st.markdown("<br/>", unsafe_allow_html=True)
        run_button = st.button(
            "🚀 Research",
            type="primary",
            disabled=st.session_state.running or not query,
            use_container_width=True,
        )

    if st.session_state.completed and not st.session_state.running:
        if st.button("🔄 New Research", use_container_width=False):
            reset_session()
            st.rerun()

    # ------------------------------------------------------------------ #
    # Sidebar
    # ------------------------------------------------------------------ #
    render_sidebar(st.session_state.agent_statuses)

    # ------------------------------------------------------------------ #
    # Start Pipeline
    # ------------------------------------------------------------------ #
    if run_button and query and not st.session_state.running:
        reset_session()
        st.session_state.running = True
        st.session_state.start_time = time.time()

        # Start background thread
        thread = threading.Thread(
            target=run_pipeline_in_thread,
            args=(query, st.session_state.update_queue),
            daemon=True,
        )
        thread.start()
        st.rerun()

    # ------------------------------------------------------------------ #
    # Live Progress Panel
    # ------------------------------------------------------------------ #
    if st.session_state.running or st.session_state.completed:
        # Drain the queue (non-blocking — process all available updates)
        pipeline_done = False
        while True:
            try:
                event_type, payload = st.session_state.update_queue.get_nowait()
                if event_type == "update":
                    process_update(payload)
                elif event_type == "error":
                    st.session_state.errors.append(f"Pipeline error: {payload}")
                    pipeline_done = True
                elif event_type == "done":
                    pipeline_done = True
            except Empty:
                break

        if pipeline_done:
            st.session_state.running = False
            st.session_state.completed = True

        # Show sub-questions
        render_sub_questions(st.session_state.sub_questions)

        # Show thinking steps
        render_thinking_steps(st.session_state.thinking_steps)

        # Show status if still running
        if st.session_state.running:
            current = st.session_state.current_agent
            if current:
                st.info(
                    f"⏳ **{current.capitalize()} agent** is working... "
                    f"(this may take 30-60 seconds per agent)"
                )
            st.rerun()  # Trigger refresh to drain queue again

    # ------------------------------------------------------------------ #
    # Final Report
    # ------------------------------------------------------------------ #
    if st.session_state.final_report:
        st.markdown("---")
        st.markdown("## 📄 Research Report")

        # Download button
        st.download_button(
            label="⬇️ Download Report (Markdown)",
            data=st.session_state.final_report,
            file_name=f"agentforge_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )

        st.markdown(st.session_state.final_report)

    elif st.session_state.completed and not st.session_state.final_report:
        st.error(
            "The pipeline completed but no final report was generated. "
            "Check the warnings in the sidebar for details."
        )

    # ------------------------------------------------------------------ #
    # Footer
    # ------------------------------------------------------------------ #
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#6c757d; font-size:0.8rem;'>"
        "AgentForge · Built with LangGraph, MCP, Anthropic Claude · "
        "<a href='https://github.com/yourusername/AgentForge'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
