import os
import re
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# ---- 1) Wire secrets into environment (Cloud-safe) ---------------------------
def _load_secrets_into_env():
    try:
        for k, v in st.secrets.items():
            # st.secrets may be a Mapping or Section; flatten shallowly
            if isinstance(v, (str, int, float, bool)):
                os.environ[str(k)] = str(v)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    os.environ[str(kk)] = str(vv)
    except Exception:
        pass

_load_secrets_into_env()

# ---- 2) Imports from your pipeline (unchanged backend) -----------------------
from src.graph import compiled
from src.agents import render_markdown_brief, get_llm
from src.observability import get_callbacks
from src.tools.url2md import url_to_markdown  # used for force-expansion

# ---- 3) App config + theme ---------------------------------------------------
st.set_page_config(page_title="Market Brief Agent", page_icon="📈", layout="wide")
st.title("📈 Market Brief Agent")
st.caption("Topic or URL → Search → Analyze → Write → Markdown")

# ---- 4) Helpers --------------------------------------------------------------
URL_RE = re.compile(r"^https?://", re.I)

def is_url(s: str) -> bool:
    return bool(URL_RE.match(s.strip()))

def safe_artifacts_dir() -> Path:
    # On Streamlit Cloud, writing to repo root may be read-only.
    # Use a session-safe temp directory.
    base = Path(tempfile.gettempdir()) / "market_agent_artifacts"
    base.mkdir(parents=True, exist_ok=True)
    return base

def run_pipeline(query: str) -> Dict[str, Any]:
    """Invoke your compiled LangGraph pipeline."""
    state_in = {"query": query, "failure_count": 0}
    callbacks = get_callbacks()  # [] if tracing disabled/misconfigured
    final_state = compiled.invoke(state_in, config={"callbacks": callbacks})
    return final_state.get("brief") or {}

def _llm_summarize_from_sections(query: str, sections_md: str, facts_json: str) -> str:
    """If the brief is too short, synthesize a full report from extracted sections."""
    llm = get_llm()
    prompt = f"""
Create a decision-ready market brief on **{query}** using ONLY the material in the sections below.
Structure:
- Executive Summary (≤ 6 sentences)
- Key Insights (bullets, include inline [#] citations)
- Competitive / Ecosystem Snapshot
- Outlook (near-term)
- References (numbered at end — do NOT invent links)

### Source Sections
{sections_md}

### Extracted Facts (JSON)
{facts_json}

Return pure Markdown, no extra JSON.
"""
    try:
        return llm.invoke(prompt).content
    except Exception:
        # Last-resort fallback
        return (
            f"# Market Brief: {query}\n\n"
            f"**Summary**: Sources were fetched but the summarizer was unavailable. "
            f"See the sections and references below.\n\n{sections_md}\n"
        )

def guarantee_full_markdown(query: str, brief: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure we return a substantive Markdown brief, not just references.
    If _markdown is missing/short, we expand by fetching each source → markdown sections,
    then synthesize (or stitch) a full report.
    """
    md = (brief.get("_markdown") or "").strip()
    sources = brief.get("sources") or []
    facts = brief.get("key_facts") or []
    # If already substantive, return as-is
    if md and len(md) >= 800 and ("## References" in md or "### References" in md):
        return {**brief, "_markdown": md}

    # Build sections from sources (RapidAPI → Tavily Extract → Jina → local markdownify)
    sections = []
    live_sources = []
    for idx, s in enumerate(sources[:10], 1):
        url = s.get("url")
        title = s.get("title") or (url or f"Source {idx}")
        if not url:
            continue
        try:
            section_md = url_to_markdown(url)
        except Exception:
            section_md = ""
        section_md = "\n".join(section_md.splitlines()[:160])
        sections.append(f"#### [{idx}] {title}\n{section_md}\n")
        live_sources.append({"title": title, "url": url, "published_at": s.get("published_at")})

    # If we had no sources from the graph, still try to produce something
    if not sections:
        body = brief.get("summary", "").strip() or "No detailed sections available."
        md_final = f"# Market Brief: {query}\n\n{body}\n"
        return {**brief, "_markdown": md_final}

    sections_md = "\n\n---\n\n".join(sections)
    facts_json = json.dumps(facts, ensure_ascii=False, indent=2)

    # Synthesize a full draft from sections
    draft = _llm_summarize_from_sections(query, sections_md, facts_json)

    # Append numbered references (only live URLs we actually used)
    refs = "\n".join([f"{i}. [{s['title']}]({s['url']})" for i, s in enumerate(live_sources, 1)])
    md_final = f"{draft}\n\n## References\n{refs}\n"

    # Keep structured fields
    return {
        **brief,
        "sources": live_sources or sources,
        "_markdown": md_final
    }

def inject_seed_url_if_needed(user_input: str, brief: Dict[str, Any]) -> Dict[str, Any]:
    """
    If user gave a URL, ensure it appears in sources even if the graph didn't include it.
    This keeps the report grounded on the exact page they asked to analyze.
    """
    if not is_url(user_input):
        return brief
    url = user_input.strip()
    sources = brief.get("sources") or []
    if not any((s.get("url") or "").strip().lower() == url.lower() for s in sources):
        sources = [{"title": url, "url": url, "published_at": None}, *sources]
    brief["sources"] = sources
    return brief

# ---- 5) Sidebar controls -----------------------------------------------------
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

    llm_mode = st.selectbox("LLM mode", ["groq", "stub"], index=0 if os.getenv("LLM_MODE", "groq").lower()=="groq" else 1)
    tracing_on = st.toggle("LangSmith tracing", value=os.getenv("LANGSMITH_ENABLED", "false").lower() in ("1","true","yes","on"))
    http_timeout = st.slider("HTTP timeout (s)", 5, 60, int(os.getenv("HTTP_TIMEOUT", "15")))
    run_btn = st.button("▶️ Run", type="primary", use_container_width=True)

# Reflect sidebar settings into env for backend nodes/tools
os.environ["MAX_SOURCES"] = str(max_sources)
os.environ["MIN_NON_EMPTY_SOURCES"] = str(min_non_empty)
os.environ["LLM_MODE"] = llm_mode
os.environ["LANGSMITH_ENABLED"] = "true" if tracing_on else "false"
os.environ["HTTP_TIMEOUT"] = str(http_timeout)

# ---- 6) Main action ----------------------------------------------------------
left, right = st.columns([2, 1])

if run_btn:
    q = (user_input or "").strip()
    if not q:
        st.error("Please enter a topic or a URL.")
        st.stop()

    with st.status("Running research pipeline…", expanded=True) as status:
        status.write("• Searching & collecting sources")
        status.write("• Extracting facts")
        status.write("• Writing the brief")

        # 6a) If URL, keep same pipeline but later force-include that URL in sources
        try:
            brief = run_pipeline(q)
        except Exception as e:
            status.update(label="Error ❌", state="error")
            st.exception(e)
            st.stop()

        # 6b) Ensure the user URL (if any) is considered a source
        brief = inject_seed_url_if_needed(q, brief)

        # 6c) Guarantee a full Markdown body (not just references)
        brief = guarantee_full_markdown(q, brief)

        status.update(label="Done ✅", state="complete")

    # 6d) Persist artifacts to a Cloud-safe temp dir
    art_dir = safe_artifacts_dir()
    md = brief.get("_markdown") or render_markdown_brief(brief)
    (art_dir / "brief.md").write_text(md, encoding="utf-8")
    (art_dir / "sample_output.json").write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")

    # 6e) Display
    with left:
        st.subheader("Brief (Markdown)")
        st.markdown(md)

        st.download_button(
            "💾 Download Markdown",
            data=md.encode("utf-8"),
            file_name="brief.md",
            mime="text/markdown",
            use_container_width=True
        )
        st.download_button(
            "💾 Download JSON",
            data=json.dumps(brief, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="sample_output.json",
            mime="application/json",
            use_container_width=True
        )

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
    st.markdown(
        '<div style="background:#3D155F;color:#fff;border:1px solid #2a0e43;border-radius:8px;'
        'padding:12px 16px;font-size:.95rem;">Enter a topic in the sidebar and click '
        '<b>Run research</b> to generate a brief.</div>',
        unsafe_allow_html=True
    )
