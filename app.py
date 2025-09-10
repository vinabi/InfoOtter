# app.py  — drop-in replacement
import os
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from importlib import reload

# ---------- 0) Load configuration EARLY ----------
# Local dev: .env
load_dotenv(override=False)

# Streamlit Cloud: secrets.toml → os.environ
if hasattr(st, "secrets"):
    for k, v in st.secrets.items():
        os.environ[str(k)] = str(v)

# ---------- 1) Page + minimal theming ----------
st.set_page_config(page_title="Market Research Multiagent", page_icon="📈", layout="wide")

# Info bar color override (#3D155F)
st.markdown("""
<style>
div[data-testid="stAlert"], div[role="alert"], div.stAlert {
  background: #3D155F !important;
  color: #ffffff !important;
  border: 1px solid #2a0e43 !important;
  border-radius: 8px !important;
}
div[data-testid="stAlert"] *, div[role="alert"] * { color:#fff !important; fill:#fff !important; }
</style>
""", unsafe_allow_html=True)

# ---------- 2) Paths ----------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ---------- 3) Sidebar (choose settings BEFORE importing pipeline) ----------
st.sidebar.header("Settings")

default_topic = os.getenv("QUERY", "").strip().strip('"') or "agent-to-agent (A2A) and Model Context Protocol (MCP)"
topic = st.sidebar.text_area("Research topic", value=default_topic, height=90,
                             placeholder="e.g., artificial intelligence applications in healthcare")

c1, c2 = st.sidebar.columns(2)
max_sources = c1.number_input("Max sources", min_value=3, max_value=30,
                              value=int(os.getenv("MAX_SOURCES", "10")), step=1)
min_non_empty = c2.number_input("Min non-empty", min_value=1, max_value=20,
                                value=int(os.getenv("MIN_NON_EMPTY_SOURCES", "5")), step=1)

llm_mode = st.sidebar.selectbox("LLM mode", options=["groq", "stub"],
                                index=0 if os.getenv("LLM_MODE", "groq").lower() == "groq" else 1)

tracing_on = st.sidebar.toggle("Enable LangSmith tracing",
                               value=os.getenv("LANGSMITH_ENABLED", "false").lower() in ("1","true","yes","on"))
allow_stubs = st.sidebar.toggle("Allow offline stubs when search fails",
                                value=os.getenv("ALLOW_STUBS", "false").lower() in ("1","true","yes","on"))
http_timeout = st.sidebar.slider("HTTP timeout (s)", min_value=5, max_value=60,
                                 value=int(os.getenv("HTTP_TIMEOUT", "15")))

run_btn = st.sidebar.button("Run research", type="primary", use_container_width=True)

# Apply sidebar choices to the current process env (so imports below see them)
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

# Safe renderer: use writer’s markdown if present, else fallback function
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
    # Prefer writer-generated markdown
    md = brief.get("_markdown")
    if md: return md
    # Fallback to agents.render_markdown_brief if it exists
    fn = getattr(agents, "render_markdown_brief", None)
    if callable(fn):
        try:
            return fn(brief)
        except Exception:
            pass
    # Final fallback
    return _render_markdown_fallback(brief)

# ---------- 5) Runner ----------
def run_pipeline(q: str) -> Dict[str, Any]:
    state_in = {"query": q, "failure_count": 0}
    callbacks = get_callbacks()  # [] if tracing disabled or misconfigured
    final_state = compiled.invoke(state_in, config={"callbacks": callbacks})
    return final_state.get("brief") or {}

# ---------- 6) UI body ----------
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
    # Persist artifacts
    (ARTIFACTS / "brief.md").write_text(md, encoding="utf-8")
    (ARTIFACTS / "sample_output.json").write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")

    # Left: Markdown + downloads
    with left:
        st.subheader("Brief (Markdown)")
        st.markdown(md)
        st.download_button("Download Markdown", data=md.encode("utf-8"),
                           file_name="brief.md", mime="text/markdown", use_container_width=True)
        st.download_button("Download JSON",
                           data=json.dumps(brief, indent=2, ensure_ascii=False).encode("utf-8"),
                           file_name="sample_output.json", mime="application/json", use_container_width=True)

    # Right: sources & facts
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
    # Branded info bar when idle
    message = "Enter a topic in the sidebar and click 𝗥𝘂𝗻 𝗿𝗲𝘀𝗲𝗮𝗿𝗰𝗵 to generate a brief."
    st.markdown("""
    <style>
    .brand-info {
      background:#3D155F; color:#ffffff; border:1px solid #2a0e43;
      border-radius:8px; padding:12px 16px; font-size:0.95rem;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="brand-info">{message}</div>', unsafe_allow_html=True)
