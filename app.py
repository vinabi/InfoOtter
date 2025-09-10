# app.py
import os
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# --- Load .env once (works on Windows/macOS/Linux) ---
load_dotenv(override=False)

# Project-relative paths
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

from src.graph import compiled
from src.agents import get_llm 
from src.observability import get_callbacks

# ---------- UI ----------
st.set_page_config(page_title="Market Research Multiagent", page_icon="📈", layout="wide")

st.markdown("""
<style>
/* Try all Streamlit alert selectors */
div[data-testid="stAlert"],               /* new */
div[role="alert"],                        /* generic */
div.stAlert,                              /* older */
section.main div[data-baseweb="notification"] /* some builds */ {
  background: #3D155F !important;
  color: #ffffff !important;
  border: 1px solid #2a0e43 !important;
  border-radius: 8px !important;
}

/* Ensure inner text/icon inherits white */
div[data-testid="stAlert"] * ,
div[role="alert"] * {
  color: #ffffff !important;
  fill: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Market Research Multiagent")
st.caption("Query → Search → Analyze → Write → Markdown")

# Sidebar controls
with st.sidebar:
    st.header("Settings")

    default_topic = os.getenv("QUERY", "").strip().strip('"') or "agent-to-agent (A2A) and Model Context Protocol (MCP)"
    topic = st.text_area("Research topic", value=default_topic, height=90,
                         placeholder="e.g., artificial intelligence applications in healthcare")

    colA, colB = st.columns(2)
    with colA:
        max_sources = st.number_input("Max sources", min_value=3, max_value=30,
                                      value=int(os.getenv("MAX_SOURCES", "10")), step=1)
    with colB:
        min_non_empty = st.number_input("Min non-empty", min_value=1, max_value=20,
                                        value=int(os.getenv("MIN_NON_EMPTY_SOURCES", "5")), step=1)

    # LLM toggle (keeps your .env defaults)
    llm_mode = st.selectbox("LLM mode", options=["groq", "stub"],
                            index=0 if os.getenv("LLM_MODE", "groq").lower() == "groq" else 1)

    # Tracing toggle (safe if you lack a LangSmith key)
    tracing_on = st.toggle("Enable LangSmith tracing", value=os.getenv("LANGSMITH_ENABLED", "false").lower() in ("1","true","yes","on"))

    # Optional: allow stubs (kept off by default now)
    allow_stubs = st.toggle("Allow offline stubs when search fails", value=os.getenv("ALLOW_STUBS", "false").lower() in ("1","true","yes","on"))

    # Network timeouts, etc.
    http_timeout = st.slider("HTTP timeout (s)", min_value=5, max_value=60, value=int(os.getenv("HTTP_TIMEOUT", "15")))

    run_btn = st.button("Run research", type="primary", use_container_width=True)

# Keep env in sync for the current process (does not overwrite your .env)
os.environ["MAX_SOURCES"] = str(max_sources)
os.environ["MIN_NON_EMPTY_SOURCES"] = str(min_non_empty)
os.environ["LLM_MODE"] = llm_mode
os.environ["LANGSMITH_ENABLED"] = "true" if tracing_on else "false"
os.environ["ALLOW_STUBS"] = "true" if allow_stubs else "false"
os.environ["HTTP_TIMEOUT"] = str(http_timeout)

# ---------- Run ----------
def run_pipeline(q: str) -> Dict[str, Any]:
    """
    Invokes your LangGraph with the current settings/env, returns final state['brief'] dict.
    """
    state_in = {"query": q, "failure_count": 0}
    # If tracing is on and properly configured, callbacks will be populated; otherwise []
    callbacks = get_callbacks()
    final_state = compiled.invoke(state_in, config={"callbacks": callbacks})
    brief = final_state.get("brief") or {}
    return brief

# ---------- UI Regions ----------
left, right = st.columns([2, 1])

if run_btn:
    if not topic.strip():
        st.error("Please enter a topic to research.")
        st.stop()

    with st.status("Running research pipeline…", expanded=True) as status:
        st.write("• Searching & collecting sources")
        st.write("• Extracting facts")
        st.write("• Writing the brief")
        try:
            brief = run_pipeline(topic.strip())
            status.update(label="Done", state="complete")
        except Exception as e:
            status.update(label="Error", state="error")
            st.exception(e)
            st.stop()

    # Persist artifacts
    md = brief.get("_markdown") or render_markdown_brief(brief)
    (ARTIFACTS / "brief.md").write_text(md, encoding="utf-8")
    (ARTIFACTS / "sample_output.json").write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")

    # Display the brief
    with left:
        st.subheader("Brief (Markdown)")
        st.markdown(md)

        st.download_button(
            "Download Markdown",
            data=md.encode("utf-8"),
            file_name="brief.md",
            mime="text/markdown",
            use_container_width=True
        )

        st.download_button(
            "Download JSON",
            data=json.dumps(brief, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="sample_output.json",
            mime="application/json",
            use_container_width=True
        )

    # Display sources & facts
    with right:
        st.subheader("Sources")
        srcs = brief.get("sources") or []
        if srcs:
            df = pd.DataFrame([{
                "title": s.get("title",""),
                "url": s.get("url",""),
                "published_at": s.get("published_at","")
            } for s in srcs])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No sources found.")

        st.subheader("Facts")
        facts = brief.get("key_facts") or []
        if facts:
            df_f = pd.DataFrame([{
                "fact": f.get("fact",""),
                "evidence_url": f.get("evidence_url",""),
                "confidence": f.get("confidence", 0.0),
            } for f in facts])
            st.dataframe(df_f, use_container_width=True, hide_index=True)
        else:
            st.info("No extracted facts available.")

else:
    message = "Enter a topic in the sidebar and click 𝗥𝘂𝗻 𝗿𝗲𝘀𝗲𝗮𝗿𝗰𝗵 to generate a brief."
    st.markdown("""
    <style>
    .brand-info {
      background:#3D155F;
      color:#ffffff;
      border:1px solid #2a0e43;
      border-radius:8px;
      padding:12px 16px;
      font-size:0.95rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="brand-info">{message}</div>', unsafe_allow_html=True)





