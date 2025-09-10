# app.py — ultra-thin wrapper around your existing multi-agent graph
import os, sys, json, tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

# ------------------------- Secrets → env (Cloud-safe) -------------------------
def _load_secrets_into_env():
    try:
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)):
                os.environ[str(k)] = str(v)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    os.environ[str(kk)] = str(vv)
    except Exception:
        pass

_load_secrets_into_env()

# ------------------------- Make 'src' importable ------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Require your actual code — if this fails, we want to see it
from src.graph import compiled
from src.agents import render_markdown_brief  # only for display fallback
try:
    from src.observability import get_callbacks
except Exception:
    def get_callbacks(): return []

# ------------------------- Streamlit UI --------------------------------------
st.set_page_config(page_title="Market Brief Agent", page_icon="📈", layout="wide")
st.title("📈 Market Brief Agent")
st.caption("Runs your original multi-agent chain exactly as in local.")

with st.sidebar:
    st.header("Settings")
    query = st.text_area(
        "Enter a topic or a URL",
        value=os.getenv("QUERY", ""),
        height=90,
        placeholder="e.g., agent-to-agent (A2A) and MCP OR https://example.com/post"
    )

    col1, col2 = st.columns(2)
    with col1:
        max_sources = st.number_input("MAX_SOURCES", 3, 50, int(os.getenv("MAX_SOURCES", "10")))
    with col2:
        min_non_empty = st.number_input("MIN_NON_EMPTY_SOURCES", 1, 20, int(os.getenv("MIN_NON_EMPTY_SOURCES", "5")))
    http_timeout = st.slider("HTTP_TIMEOUT (s)", 5, 60, int(os.getenv("HTTP_TIMEOUT", "15")))
    tracing = st.toggle("LangSmith tracing", value=os.getenv("LANGSMITH_ENABLED", "false").lower() in ("1","true","yes","on"))
    run_btn = st.button("▶️ Run", type="primary", use_container_width=True)

# Mirror sidebar to env so your existing nodes/tools read them
os.environ["MAX_SOURCES"] = str(max_sources)
os.environ["MIN_NON_EMPTY_SOURCES"] = str(min_non_empty)
os.environ["HTTP_TIMEOUT"] = str(http_timeout)
os.environ["LANGSMITH_ENABLED"] = "true" if tracing else "false"

# Quick health panel: confirms Cloud actually sees your keys
with st.expander("🔎 Health / Keys (redacted)"):
    def _mask(v: str) -> str:
        if not v: return "—"
        v = str(v)
        return v[:4] + "…" + v[-4:] if len(v) > 10 else "set"
    st.write({
        "GROQ_API_KEY": _mask(os.getenv("GROQ_API_KEY")),
        "RAPIDAPI_KEY": _mask(os.getenv("RAPIDAPI_KEY")),
        "TAVILY_API_KEY": _mask(os.getenv("TAVILY_API_KEY")),
        "SERP_API_KEY": _mask(os.getenv("SERP_API_KEY")),
        "NEWSAPI_KEY": _mask(os.getenv("NEWSAPI_KEY")),
        "LANGSMITH_ENABLED": os.getenv("LANGSMITH_ENABLED"),
        "MAX_SOURCES": os.getenv("MAX_SOURCES"),
        "MIN_NON_EMPTY_SOURCES": os.getenv("MIN_NON_EMPTY_SOURCES"),
        "HTTP_TIMEOUT": os.getenv("HTTP_TIMEOUT"),
    })

# ------------------------- Run your chain ------------------------------------
left, right = st.columns([2, 1])

def _artifacts_dir() -> Path:
    # temp is always writable on Streamlit Cloud
    p = Path(tempfile.gettempdir()) / "market_agent_artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p

if run_btn:
    q = (query or "").strip()
    if not q:
        st.error("Please enter a topic or a URL.")
        st.stop()

    with st.status("Running your pipeline…", expanded=True) as status:
        status.write("• invoking compiled LangGraph")
        state_in: Dict[str, Any] = {"query": q, "failure_count": 0}
        try:
            final = compiled.invoke(state_in, config={"callbacks": get_callbacks()})
            brief = (final or {}).get("brief") or {}
            status.update(label="Done ✅", state="complete")
        except Exception as e:
            status.update(label="Error ❌", state="error")
            st.exception(e)
            st.stop()

    # Render using whatever your writer produced
    md = (brief.get("_markdown") or "").strip()
    if not md:
        # fallback: render minimal from your structured brief
        md = render_markdown_brief(brief)

    # Persist artifacts to Cloud-safe temp dir
    outdir = _artifacts_dir()
    (outdir / "brief.md").write_text(md, encoding="utf-8")
    (outdir / "sample_output.json").write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")

    # Display Markdown
    with left:
        st.subheader("Brief (Markdown)")
        st.markdown(md)
        st.download_button("💾 Download Markdown", md.encode("utf-8"), "brief.md", "text/markdown", use_container_width=True)
        st.download_button("💾 Download JSON", json.dumps(brief, indent=2, ensure_ascii=False).encode("utf-8"),
                           "sample_output.json", "application/json", use_container_width=True)

    # Display Sources & Facts (as produced by YOUR pipeline)
    with right:
        st.subheader("Sources")
        srcs = brief.get("sources") or []
        if srcs:
            df = pd.DataFrame([{
                "title": s.get("title",""), "url": s.get("url",""), "published_at": s.get("published_at","")
            } for s in srcs])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No sources found.")

        st.subheader("Facts")
        facts = brief.get("key_facts") or []
        if facts:
            df_f = pd.DataFrame([{
                "fact": f.get("fact",""), "evidence_url": f.get("evidence_url",""), "confidence": f.get("confidence", 0.0)
            } for f in facts])
            st.dataframe(df_f, use_container_width=True, hide_index=True)
        else:
            st.info("No extracted facts available.")

else:
    st.info("Enter a topic or a URL in the sidebar and click **Run**.")