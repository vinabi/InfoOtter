# app.py
import os
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from importlib import reload

# --- Load configuration early ---
load_dotenv(override=False)  # local dev
if hasattr(st, "secrets"):   # Streamlit Cloud
    for k, v in st.secrets.items():
        os.environ[str(k)] = str(v)

# --- Page config & styling ---
st.set_page_config(page_title="Market Research Multiagent", page_icon="📈", layout="wide")

st.markdown("""
<style>
div[data-testid="stAlert"], div[role="alert"], div.stAlert {
  background: #3D155F !important;
  color: #ffffff !important;
  border: 1px solid #2a0e43 !important;
  border-radius: 8px !important;
}
div[data-testid="stAlert"] *, div[role="alert"] * {
  color: #ffffff !important;
  fill: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Market Research Multiagent")
st.caption("Query → Search → Analyze → Write → Markdown")

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    default_topic = os.getenv("QUERY", "").strip().strip('"') or "agent-to-agent (A2A) and Model Context Protocol (MCP)"
    topic = st.text_area("Research topic", value=default_topic, height=90,
                         placeholder="e.g., artificial intelligence applications in healthcare")

    colA, colB = st.columns(2)
    with colA:
        max_sources = st.number_input("Max sources", 3, 30, int(os.getenv("MAX_SOURCES", "10")), 1)
    with colB:
        min_non_empty = st.number_input("Min non-empty", 1, 20, int(os.getenv("MIN_NON_EMPTY_SOURCES", "5")), 1)

    llm_mode = st.selectbox("LLM mode", ["groq", "stub"],
                            index=0 if os.getenv("LLM_MODE", "groq").lower() == "groq" else 1)
    tracing_on = st.toggle("Enable LangSmith tracing",
                           value=os.getenv("LANGSMITH_ENABLED", "false").lower() in ("1","true","yes","on"))
    allow_stubs = st.toggle("Allow offline stubs when search fails",
                            value=os.getenv("ALLOW_STUBS", "false").lower() in ("1","true","yes","on"))
    http_timeout = st.slider("HTTP timeout (s)", 5, 60, int(os.getenv("HTTP_TIMEOUT", "15")))
    run_btn = st.button("Run research", type="primary", use_container_width=True)

# Sync env for current session
os.environ.update({
    "MAX_SOURCES": str(max_sources),
    "MIN_NON_EMPTY_SOURCES": str(min_non_empty),
    "LLM_MODE": llm_mode,
    "LANGSMITH_ENABLED": "true" if tracing_on else "false",
    "ALLOW_STUBS": "true" if allow_stubs else "false",
    "HTTP_TIMEOUT": str(http_timeout),
})

# --- Import pipeline AFTER env is ready ---
import src.agents as agents
import src.graph as graph
import src.observability as observability
reload(agents); reload(graph); reload(observability)

compiled = graph.compiled
get_callbacks = observability.get_callbacks

def render_markdown(brief: Dict) -> str:
    # Prefer writer’s markdown if present
    if brief.get("_markdown"):
        return brief["_markdown"]
    # Fallback: simple renderer
    lines = [f"# Market Brief: {brief.get('topic','')}"]
    if brief.get("summary"): lines += ["", brief["summary"], ""]
    if brief.get("key_facts"):
        lines.append("## Key Facts")
        for f in brief["key_facts"]:
            ev = f.get("evidence_url","")
            lines.append(f"- {f.get('fact','')}")
            if ev: lines.append(f"  Evidence: {ev} (confidence {f.get('confidence',0):.2f})")
    if brief.get("sources"):
        lines.append("## References")
        for i, s in enumerate(brief["sources"], 1):
            lines.append(f"{i}. [{s.get('title','Untitled')}]({s.get('url','')})")
    return "\n".join(lines)

def run_pipeline(q: str) -> Dict[str, Any]:
    state_in = {"query": q, "failure_count": 0}
    callbacks = get_callbacks()
    return compiled.invoke(state_in, config={"callbacks": callbacks}).get("brief") or {}

# --- UI regions ---
left, right = st.columns([2, 1])
if run_btn:
    if not topic.strip():
        st.error("Please enter a topic to research."); st.stop()

    with st.status("Running research pipeline…", expanded=True) as status:
        try:
            brief = run_pipeline(topic.strip())
            status.update(label="Done", state="complete")
        except Exception as e:
            status.update(label="Error", state="error"); st.exception(e); st.stop()

    md = render_markdown(brief)
    (ARTIFACTS/"brief.md").write_text(md, encoding="utf-8")
    (ARTIFACTS/"sample_output.json").write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")

    with left:
        st.subheader("Brief (Markdown)")
        st.markdown(md)
        st.download_button("Download Markdown", md.encode("utf-8"), "brief.md", "text/markdown")
        st.download_button("Download JSON", json.dumps(brief, indent=2, ensure_ascii=False).encode("utf-8"),
                           "sample_output.json", "application/json")

    with right:
        st.subheader("Sources")
        srcs = brief.get("sources") or []
        if srcs:
            st.dataframe(pd.DataFrame(srcs), use_container_width=True, hide_index=True)
        else:
            st.info("No sources found.")

        st.subheader("Facts")
        facts = brief.get("key_facts") or []
        if facts:
            st.dataframe(pd.DataFrame(facts), use_container_width=True, hide_index=True)
        else:
            st.info("No extracted facts available.")
else:
    st.markdown('<div class="brand-info">Enter a topic in the sidebar and click **Run research** to generate a brief.</div>',
                unsafe_allow_html=True)
