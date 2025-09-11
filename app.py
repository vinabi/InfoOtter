# app.py  — replace everything
import os
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from importlib import reload

# --- 0) Load config EARLY (fixes Cloud "refs-only" brief) ---
load_dotenv(override=False)                       # local dev
if hasattr(st, "secrets"):                        # Streamlit Cloud
    for k, v in st.secrets.items():
        os.environ[str(k)] = str(v)

# --- 1) Page + styling ---
st.set_page_config(page_title="Market Research Multiagent", page_icon="📈", layout="wide")
st.markdown("""
<style>
div[data-testid="stAlert"], div[role="alert"], div.stAlert{
  background:#3D155F!important;color:#fff!important;border:1px solid #2a0e43!important;border-radius:8px!important;
}
div[data-testid="stAlert"] *,div[role="alert"] *{color:#fff!important;fill:#fff!important;}
</style>
""", unsafe_allow_html=True)

st.title("Market Research Multiagent")
st.caption("Query → Search → Analyze → Write → Markdown")

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# --- 2) Sidebar FIRST (so env reflects user choices) ---
with st.sidebar:
    st.header("Settings")
    default_topic = os.getenv("QUERY", "").strip().strip('"') or "agent-to-agent (A2A) and Model Context Protocol (MCP)"
    topic = st.text_area("Research topic", value=default_topic, height=90)

    c1, c2 = st.columns(2)
    max_sources    = c1.number_input("Max sources", 3, 30, int(os.getenv("MAX_SOURCES", "10")), 1)
    min_non_empty  = c2.number_input("Min non-empty", 1, 20, int(os.getenv("MIN_NON_EMPTY_SOURCES", "5")), 1)

    llm_mode  = st.selectbox("LLM mode", ["groq", "stub"],
                             index=0 if os.getenv("LLM_MODE","groq").lower()=="groq" else 1)
    tracing_on = st.toggle("Enable LangSmith tracing",
                           value=os.getenv("LANGSMITH_ENABLED","false").lower() in ("1","true","yes","on"))
    allow_stubs = st.toggle("Allow offline stubs when search fails",
                            value=os.getenv("ALLOW_STUBS","false").lower() in ("1","true","yes","on"))
    http_timeout = st.slider("HTTP timeout (s)", 5, 60, int(os.getenv("HTTP_TIMEOUT","15")))
    run_btn = st.button("Run research", type="primary", use_container_width=True)

# Apply sidebar → env (so imports see correct config)
os.environ.update({
    "MAX_SOURCES": str(max_sources),
    "MIN_NON_EMPTY_SOURCES": str(min_non_empty),
    "LLM_MODE": llm_mode,
    "LANGSMITH_ENABLED": "true" if tracing_on else "false",
    "ALLOW_STUBS": "true" if allow_stubs else "false",
    "HTTP_TIMEOUT": str(http_timeout),
})

# --- 3) Import pipeline AFTER env is ready; reload to pick up changes ---
import src.agents as agents
import src.graph as graph
import src.observability as observability
reload(agents); reload(graph); reload(observability)

compiled = graph.compiled
get_callbacks = observability.get_callbacks

# --- 4) Safe renderer: use writer’s markdown, else fallback ---
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
    md = brief.get("_markdown")
    if md: return md
    fn = getattr(agents, "render_markdown_brief", None)
    if callable(fn):
        try: return fn(brief)
        except Exception: pass
    return _render_markdown_fallback(brief)

# --- 5) Runner ---
def run_pipeline(q: str) -> Dict[str, Any]:
    state_in = {"query": q, "failure_count": 0}
    callbacks = get_callbacks()  # [] if tracing disabled/missing
    out = compiled.invoke(state_in, config={"callbacks": callbacks})
    return out.get("brief") or {}

# --- 6) UI body ---
left, right = st.columns([2, 1])

if run_btn:
    if not topic.strip():
        st.error("Please enter a topic to research."); st.stop()

    with st.status("Running research pipeline…", expanded=True) as status:
        st.write("• Searching & collecting sources")
        st.write("• Extracting facts")
        st.write("• Writing the brief")
        try:
            brief = run_pipeline(topic.strip())
            status.update(label="Done", state="complete")
        except Exception as e:
            status.update(label="Error", state="error"); st.exception(e); st.stop()

    md = render_markdown(brief)

    # Persist artifacts
    (ARTIFACTS/"brief.md").write_text(md, encoding="utf-8")
    (ARTIFACTS/"sample_output.json").write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")

    # Left: report + downloads
    with left:
        st.subheader("Brief (Markdown)")
        st.markdown(md)
        st.download_button("Download Markdown", md.encode("utf-8"), "brief.md", "text/markdown", use_container_width=True)
        st.download_button("Download JSON",
                           json.dumps(brief, indent=2, ensure_ascii=False).encode("utf-8"),
                           "sample_output.json", "application/json", use_container_width=True)

    # Right: sources & facts
    with right:
        st.subheader("Sources")
        srcs = brief.get("sources") or []
        if srcs:
            df = pd.DataFrame([{
                "title": s.get("title",""), "url": s.get("url",""),
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
    st.markdown(
        '<div style="background:#3D155F;color:#fff;border:1px solid #2a0e43;border-radius:8px;'
        'padding:12px 16px;font-size:.95rem;">Enter a topic in the sidebar and click '
        '<b>Run research</b> to generate a brief.</div>',
        unsafe_allow_html=True
    )
