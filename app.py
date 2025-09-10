# app.py
import os
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---------- Load config EARLY ----------
# 1) Load .env if present (local dev)
load_dotenv(override=False)

# 2) Load Streamlit secrets into os.environ (Streamlit Cloud)
#    MUST be before importing your pipeline modules.
if hasattr(st, "secrets"):
    for k, v in st.secrets.items():
        os.environ[str(k)] = str(v)

# ---------- Page + theme ----------
st.set_page_config(page_title="Market Research Multiagent", page_icon="📈", layout="wide")

# Make the info bar purple (#3D155F)
st.markdown("""
<style>
/* Cover all alert variants Streamlit uses */
div[data-testid="stAlert"],
div[role="alert"],
div.stAlert {
  background: #3D155F !important;
  color: #ffffff !important;
  border: 1px solid #2a0e43 !important;
  border-radius: 8px !important;
}
div[data-testid="stAlert"] * ,
div[role="alert"] * {
  color: #ffffff !important;
  fill: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ---------- Sidebar (choose settings BEFORE building the graph) ----------
st.sidebar.header("Settings")

default_topic = os.getenv("QUERY", "").strip().strip('"') or "agent-to-agent (A2A) and Model Context Protocol (MCP)"
topic = st.sidebar.text_area("Research topic", value=default_topic, height=90)

colA, colB = st.sidebar.columns(2)
max_sources = colA.number_input("Max sources", min_value=3, max_value=30, value=int(os.getenv("MAX_SOURCES", "10")), step=1)
min_non_empty = colB.number_input("Min non-empty", min_value=1, max_value=20, value=int(os.getenv("MIN_NON_EMPTY_SOURCES", "5")), step=1)

llm_mode = st.sidebar.selectbox("LLM mode", options=["groq", "stub"],
                                index=0 if os.getenv("LLM_MODE", "groq").lower() == "groq" else 1)

tracing_on = st.sidebar.toggle("Enable LangSmith tracing", value=os.getenv("LANGSMITH_ENABLED", "false").lower() in ("1","true","yes","on"))
allow_stubs = st.sidebar.toggle("Allow offline stubs when search fails", value=os.getenv("ALLOW_STUBS", "false").lower() in ("1","true","yes","on"))
http_timeout = st.sidebar.slider("HTTP timeout (s)", min_value=5, max_value=60, value=int(os.getenv("HTTP_TIMEOUT", "15")))

run_btn = st.sidebar.button("Run research", type="primary", use_container_width=True)

# Apply sidebar choices to env for THIS session (so the pipeline picks them up)
os.environ["MAX_SOURCES"] = str(max_sources)
os.environ["MIN_NON_EMPTY_SOURCES"] = str(min_non_empty)
os.environ["LLM_MODE"] = llm_mode
os.environ["LANGSMITH_ENABLED"] = "true" if tracing_on else "false"
os.environ["ALLOW_STUBS"] = "true" if allow_stubs else "false"
os.environ["HTTP_TIMEOUT"] = str(http_timeout)

# ---------- Build/Reload pipeline AFTER settings ----------
# Import AFTER env is ready; if user changes settings, we can reload modules.
from importlib import reload
import src.agents as agents
import src.graph as graph
import src.observability as observability

reload(agents)        # picks up env (GROQ key/mode, etc.)
reload(graph)         # rebuilds graph with current llm + settings
reload(observability)

compiled = graph.compiled
get_callbacks = observability.get_callbacks

def run_pipeline(q: str) -> Dict[str, Any]:
    state_in = {"query": q, "failure_count": 0}
    callbacks = get_callbacks()
    final_state = compiled.invoke(state_in, config={"callbacks": callbacks})
    return final_state.get("brief") or {}

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

