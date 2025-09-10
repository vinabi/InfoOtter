# app.py — Streamlit Cloud: prefer your toolchain; keep graph as first try
import os, re, json, tempfile, sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# ------------------------- 1) Load secrets into env ---------------------------
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

# Ensure repo root on sys.path so "src" imports work on Cloud
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ------------------------- 2) Import your pipeline & tools --------------------
# LangGraph compiled (first attempt)
try:
    from src.graph import compiled
except Exception:
    compiled = None

# Your agents and tools (used for direct fallback path and also to render)
# If any import fails on Cloud, we’ll raise a visible error instead of silently stubbing.
from src.agents import get_llm, run_analyst, run_writer, render_markdown_brief
from src.tools.search import aggregate_search, enrich_with_content
from src.tools.url2md import url_to_markdown  # used when user inputs a direct URL

# ------------------------- 3) UI config ---------------------------------------
st.set_page_config(page_title="Market Brief Agent", page_icon="📈", layout="wide")
st.title("📈 Market Brief Agent")
st.caption("Topic or URL → Search → Analyze → Write → Markdown")

URL_RE = re.compile(r"^https?://", re.I)
def is_url(s: str) -> bool:
    return bool(URL_RE.match((s or "").strip()))

def _safe_artifacts_dir() -> Path:
    base = Path(tempfile.gettempdir()) / "market_agent_artifacts"
    base.mkdir(parents=True, exist_ok=True)
    return base

# ------------------------- 4) Runners -----------------------------------------
def run_graph_pipeline(query: str) -> Dict[str, Any]:
    """Try your LangGraph compiled pipeline (if available)."""
    if compiled is None:
        return {}
    cfg = {}
    try:
        from src.observability import get_callbacks
        cfg = {"callbacks": get_callbacks()}
    except Exception:
        pass
    state_in = {"query": query, "failure_count": 0}
    try:
        out = compiled.invoke(state_in, config=cfg)
        return (out or {}).get("brief") or {}
    except Exception as e:
        # Surface to UI log but continue to fallback
        st.toast(f"Graph failed: {type(e).__name__}", icon="⚠️")
        return {}

def run_direct_toolchain(query: str) -> Dict[str, Any]:
    """
    Use your src.tools + src.agents directly:
    - search → enrich
    - analyst facts
    - writer (uses url2md path for full markdown extraction)
    This mirrors the local behavior that produced rich reports.
    """
    llm = get_llm()

    # If query is a URL, use it as the primary (seed) source and still allow a few more
    if is_url(query):
        seed = [{"title": query, "url": query, "description": "User-provided URL", "source": "user"}]
        # Try extracting upfront so writer has content even if search fails on Cloud
        try:
            _ = url_to_markdown(query)
        except Exception:
            pass
        fetched = seed + aggregate_search(query, max_results=int(os.getenv("MAX_SOURCES", "10")) - 1)
    else:
        fetched = aggregate_search(query, max_results=int(os.getenv("MAX_SOURCES", "10")))

    enriched = enrich_with_content(fetched)
    facts = run_analyst(llm, query, enriched)
    brief = run_writer(llm, query, facts, enriched)  # returns dict with _markdown, sources, etc.
    return brief or {}

def is_substantive(md: str) -> bool:
    text = (md or "").strip()
    return bool(text and len(text) >= 800 and "References" in text)

# ------------------------- 5) Sidebar controls --------------------------------
with st.sidebar:
    st.header("Settings")
    user_input = st.text_area(
        "Enter a topic or a URL",
        value=os.getenv("QUERY", "").strip(),
        height=90,
        placeholder="e.g., artificial intelligence in healthcare OR https://example.com/post"
    )
    col1, col2 = st.columns(2)
    with col1:
        max_sources = st.number_input("Max sources", 3, 30, int(os.getenv("MAX_SOURCES", "10")))
    with col2:
        min_non_empty = st.number_input("Min non-empty", 1, 20, int(os.getenv("MIN_NON_EMPTY_SOURCES", "5")))
    llm_mode = st.selectbox("LLM mode", ["groq", "stub"], index=0 if os.getenv("LLM_MODE","groq").lower()=="groq" else 1)
    tracing_on = st.toggle("LangSmith tracing", value=os.getenv("LANGSMITH_ENABLED","false").lower() in ("1","true","yes","on"))
    http_timeout = st.slider("HTTP timeout (s)", 5, 60, int(os.getenv("HTTP_TIMEOUT","15")))
    run_btn = st.button("▶️ Run", type="primary", use_container_width=True)

# Mirror sidebar to env for backend nodes/tools
os.environ["MAX_SOURCES"] = str(max_sources)
os.environ["MIN_NON_EMPTY_SOURCES"] = str(min_non_empty)
os.environ["LLM_MODE"] = llm_mode
os.environ["LANGSMITH_ENABLED"] = "true" if tracing_on else "false"
os.environ["HTTP_TIMEOUT"] = str(http_timeout)

# ------------------------- 6) Execute + render --------------------------------
left, right = st.columns([2, 1])

if run_btn:
    q = (user_input or "").strip()
    if not q:
        st.error("Please enter a topic or a URL.")
        st.stop()

    with st.status("Running research pipeline…", expanded=True) as status:
        status.write("• Trying graph pipeline")
        brief = run_graph_pipeline(q)

        # If the graph returned nothing or only refs, switch to your direct toolchain
        md = brief.get("_markdown") if brief else ""
        if not is_substantive(md):
            status.write("• Falling back to direct toolchain (tools + agents)")
            brief = run_direct_toolchain(q)
            md = brief.get("_markdown") or ""

        # As an extra safety, render minimal markdown if writer omitted it
        if not md:
            md = render_markdown_brief(brief)

        status.update(label="Done ✅", state="complete")

    # Save artifacts to Cloud-writable temp dir
    art_dir = _safe_artifacts_dir()
    (art_dir / "brief.md").write_text(md, encoding="utf-8")
    (art_dir / "sample_output.json").write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")

    with left:
        st.subheader("Brief (Markdown)")
        st.markdown(md)
        st.download_button("💾 Download Markdown", md.encode("utf-8"), "brief.md", "text/markdown", use_container_width=True)
        st.download_button("💾 Download JSON", json.dumps(brief, indent=2, ensure_ascii=False).encode("utf-8"), "sample_output.json", "application/json", use_container_width=True)

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
    st.info("Enter a topic or a URL in the sidebar and click **Run**.")