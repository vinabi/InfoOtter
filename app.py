# app.py — Streamlit Cloud–safe, secrets-first; no dependency on render_markdown_brief
import os
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from importlib import reload

# ---------- 0) Load config EARLY ----------
load_dotenv(override=False)  # local dev
if hasattr(st, "secrets"):   # Streamlit Cloud: secrets.toml -> env
    for k, v in st.secrets.items():
        os.environ[str(k)] = str(v)

# ---------- 1) Page + styles ----------
st.set_page_config(page_title="Market Research Multiagent", page_icon="📈", layout="wide")
st.markdown("""
<style>
div[data-testid="stAlert"], div[role="alert"], div.stAlert {
  background: #3D155F !important; color: #fff !important;
  border: 1px solid #2a0e43 !important; border-radius: 8px !important;
}
div[data-testid="stAlert"] *, div[role="alert"] * { color:#fff !important; fill:#fff !important; }
</style>
""", unsafe_allow_html=True)

# ---------- 2) Paths ----------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ---------- 3) Sidebar FIRST (so env is set before building graph) ----------
st.sidebar.header("Settings")

default_topic = os.getenv("QUERY", "").strip().strip('"') or "agent-to-agent (A2A) and Model Context Protocol (MCP)"
topic = st.sidebar.text_area("Research topic", value=default_topic, height=90)

c1, c2 = st.sidebar.columns(2)
max_sources = c1.number_input("Max sources", 3, 30, int(os.getenv("MAX_SOURCES", "10")), 1)
min_non_empty = c2.number_input("Min non-empty", 1, 20, int(os.getenv("MIN_NON_EMPTY_SOURCES", "5")), 1)

llm_mode = st.sidebar.selectbox("LLM mode", ["groq", "stub"],
                                index=0 if os.getenv("LLM_MODE", "groq").lower() == "groq" else 1)

tracing_on = st.sidebar.toggle("Enable LangSmith tracing",
                               value=os.getenv("LANGSMITH_ENABLED", "false").lower() in ("1","true","yes","on"))
allow_stubs = st.sidebar.toggle("Allow offline stubs when search fails",
                                value=os.getenv("ALLOW_STUBS", "false").lower() in ("1","true","yes","on"))
http_timeout = st.sidebar.slider("HTTP timeout (s)", 5, 60, int(os.getenv("HTTP_TIMEOUT", "15")))

run_btn = st.sidebar.button("Run research", type="primary", use_container_width=True)

# Apply sidebar choices to env for this process
os.environ["MAX_SOURCES"] = str(max_sources)
os.environ["MIN_NON_EMPTY_SOURCES"] = str(min_non_empty)
os.environ["LLM_MODE"] = llm_mode
os.environ["LANGSMITH_ENABLED"] = "true" if tracing_on else "false"
os.environ["ALLOW_STUBS"] = "true" if allow_stubs else "false"
os.environ["HTTP_TIMEOUT"] = str(http_timeout)

# ---------- 4) Import / reload pipeline AFTER env is ready ----------
import src.agents as agents
import src.graph as graph
import src.observability as observability

reload(agents)
reload(graph)
reload(observability)

compiled = graph.compiled
get_callbacks = observability.get_callbacks

# ---------- 5) Safe renderer (no dependency on agents.render_markdown_brief) ----------
def _render_markdown_fallback(brief: Dict) -> str:
    lines = [f"# Market Brief: {brief.get('topic','')}", ""]
    if brief.get("summary"): lines += [brief["summary"], ""]
    if brief.get("key_facts"):
        lines.append("## Key Facts")
        for f in brief["key_facts"]:
            ev = f.get("evidence_url","")
            lines.append(f"- {f.get('fact','')}")
            if ev: lines.append(f"  Evidence: {ev} (confidence {f.get('confidence',0):.2f})")
        lines.append("")
    srcs = brief.get("sources") or []
    if srcs:
        lines.append("## References")
        for i, s in enumerate(srcs, 1):
            title = s.get("title") or "Untitled"
            url = s.get("url","")
            pub = s.get("published_at") or ""
            lines.append(f"{i}. [{title}]({url}) {pub}")
        lines.append("")
    return "\n".join(lines)

def render_markdown(brief: Dict) -> str:
    if brief.get("_markdown"):   # preferred: writer synthesized markdown
        return brief["_markdown"]
    # optional: use agents.render_markdown_brief if your file defines it
    fn = getattr(agents, "render_markdown_brief", None)
    if callable(fn):
        try: return fn(brief)
        except Exception: pass
    return _render_markdown_fallback(brief)

# ---------- 6) Runner ----------
def run_pipeline(q: str) -> Dict[str, Any]:
    state_in = {"query": q, "failure_count": 0}
    callbacks = get_callbacks()  # [] if tracing disabled/misconfigured
    final_state = compiled.invoke(state_in, config={"callbacks": callbacks})
    return final_state.get("brief") or {}

# ---------- 7) UI ----------
st.title("Market Research Multiagent")
st.caption("Query → Search → Analyze → Write → Markdown")

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

    md = render_markdown(brief)
    (ARTIFACTS / "brief.md").write_text(md, encoding="utf-8")
    (ARTIFACTS / "sample_output.json").write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")

    with left:
        st.subheader("Brief (Markdown)")
        st.markdown(md)
        st.download_button("Download Markdown", md.encode("utf-8"), "brief.md", "text/markdown", use_container_width=True)
        st.download_button("Download JSON",
            json.dumps(brief, indent=2, ensure_ascii=False).encode("utf-8"),
            "sample_output.json", "application/json", use_container_width=True)

    with right:
        st.subheader("Sources")
        srcs = brief.get("sources") or []
        if srcs:
            df = pd.DataFrame([{"title": s.get("title",""), "url": s.get("url",""), "published_at": s.get("published_at","")} for s in srcs])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No sources found.")

        st.subheader("Facts")
        facts = brief.get("key_facts") or []
        if facts:
            df_f = pd.DataFrame([{"fact": f.get("fact",""), "evidence_url": f.get("evidence_url",""), "confidence": f.get("confidence", 0.0)} for f in facts])
            st.dataframe(df_f, use_container_width=True, hide_index=True)
        else:
            st.info("No extracted facts available.")
else:
    st.markdown(
        '<div style="background:#3D155F;color:#fff;border:1px solid #2a0e43;border-radius:8px;padding:12px 16px;">'
        'Enter a topic in the sidebar and click <b>Run research</b> to generate a brief.'
        '</div>',
        unsafe_allow_html=True,
    )
