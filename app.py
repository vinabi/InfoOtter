# app.py — Streamlit Cloud–ready; resilient imports + local fallbacks
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

# Ensure repo root is on sys.path for "src" package
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ------------------------- 2) Import pipeline (with shims) --------------------
# compiled graph
try:
    from src.graph import compiled
except Exception as e:
    # last-resort: dynamic import with path shim
    import importlib
    compiled = importlib.import_module("src.graph").compiled  # may still raise

# url->markdown extractor (we rely on this to guarantee full body)
try:
    from src.tools.url2md import url_to_markdown
except Exception:
    # tiny fallback to avoid crash; very last resort
    import requests
    from html import unescape
    def url_to_markdown(url: str, timeout: int = 15) -> str:
        try:
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=timeout)
            r.raise_for_status()
            return unescape(r.text[:40000])
        except Exception:
            return f"# Unable to fetch\n\n{url}\n"

# try official helpers; otherwise provide local fallbacks
try:
    from src.agents import render_markdown_brief as _render_md_from_agents
    from src.agents import get_llm as _get_llm_from_agents
except Exception:
    _render_md_from_agents = None
    _get_llm_from_agents = None

def render_markdown_brief(brief: Dict[str, Any]) -> str:
    if _render_md_from_agents:
        try:
            return _render_md_from_agents(brief)
        except Exception:
            pass
    # Local safe renderer
    topic = brief.get("topic","")
    summary = brief.get("summary","")
    facts = brief.get("key_facts") or []
    sources = brief.get("sources") or []
    lines = [f"# Market Brief: {topic}", "", summary, ""]
    if facts:
        lines.append("## Key Facts")
        for f in facts:
            ev = f.get("evidence_url",""); conf = f.get("confidence", 0)
            bullet = f"- {f.get('fact','')}"
            lines.append(bullet)
            if ev: lines.append(f"  Evidence: {ev} (confidence {conf:.2f})")
        lines.append("")
    if sources:
        lines.append("## References")
        for i, s in enumerate(sources, 1):
            title = s.get("title") or "Untitled"; url = s.get("url","")
            pub = s.get("published_at") or ""
            lines.append(f"{i}. [{title}]({url}) {pub}")
        lines.append("")
    return "\n".join(lines)

def get_llm():
    if _get_llm_from_agents:
        try:
            return _get_llm_from_agents()
        except Exception:
            pass
    # Local stub / GROQ minimal
    class _Stub:
        def invoke(self, prompt: str):
            class R: 
                def __init__(self, text): self.content = text
            # naive condensation
            lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()][:60]
            return R("\n".join(lines) + "\n\n(Stub summary)")
    # Try Groq via langchain_groq if key present
    if os.getenv("GROQ_API_KEY"):
        try:
            from langchain_groq import ChatGroq
            model = os.getenv("GROQ_MODEL","llama3-70b-8192")
            return ChatGroq(model_name=model, temperature=0.2)
        except Exception:
            pass
    return _Stub()

# ------------------------- 3) UI config ---------------------------------------
st.set_page_config(page_title="Market Brief Agent", page_icon="📈", layout="wide")
st.title("📈 Market Brief Agent")
st.caption("Topic or URL → Search → Analyze → Write → Markdown")

URL_RE = re.compile(r"^https?://", re.I)
def is_url(s: str) -> bool: return bool(URL_RE.match((s or "").strip()))

def _safe_artifacts_dir() -> Path:
    # temp is writable on Streamlit Cloud
    base = Path(tempfile.gettempdir()) / "market_agent_artifacts"
    base.mkdir(parents=True, exist_ok=True)
    return base

def _run_pipeline(query: str) -> Dict[str, Any]:
    state_in = {"query": query, "failure_count": 0}
    # callbacks are optional; import inside to avoid import errors
    try:
        from src.observability import get_callbacks
        cfg = {"callbacks": get_callbacks()}
    except Exception:
        cfg = {}
    out = compiled.invoke(state_in, config=cfg)
    return out.get("brief") or {}

def _llm_summarize_from_sections(query: str, sections_md: str, facts_json: str) -> str:
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
        return f"# Market Brief: {query}\n\n_Summarizer unavailable. See sections below._\n\n{sections_md}\n"

def _guarantee_full_markdown(query: str, brief: Dict[str, Any]) -> Dict[str, Any]:
    md = (brief.get("_markdown") or "").strip()
    sources = brief.get("sources") or []
    facts = brief.get("key_facts") or []
    if md and len(md) >= 800 and ("## References" in md or "### References" in md):
        return {**brief, "_markdown": md}

    # build sections from live URLs
    sections, live_sources = [], []
    for idx, s in enumerate(sources[:10], 1):
        url = s.get("url"); title = s.get("title") or (url or f"Source {idx}")
        if not url: continue
        try:
            sec = url_to_markdown(url)
        except Exception:
            sec = ""
        sec = "\n".join(sec.splitlines()[:160])
        sections.append(f"#### [{idx}] {title}\n{sec}\n")
        live_sources.append({"title": title, "url": url, "published_at": s.get("published_at")})
    if not sections:
        # nothing to expand; render minimal but valid
        return {**brief, "_markdown": render_markdown_brief(brief)}

    import json as _json
    sections_md = "\n\n---\n\n".join(sections)
    facts_json = _json.dumps(facts, ensure_ascii=False, indent=2)
    draft = _llm_summarize_from_sections(query, sections_md, facts_json)
    refs = "\n".join([f"{i}. [{s['title']}]({s['url']})" for i, s in enumerate(live_sources, 1)])
    md_final = f"{draft}\n\n## References\n{refs}\n"
    return {**brief, "sources": live_sources or sources, "_markdown": md_final}

def _inject_seed_url(user_input: str, brief: Dict[str, Any]) -> Dict[str, Any]:
    if not is_url(user_input):
        return brief
    url = user_input.strip()
    srcs = brief.get("sources") or []
    if not any((s.get("url") or "").strip().lower() == url.lower() for s in srcs):
        srcs = [{"title": url, "url": url, "published_at": None}, *srcs]
    brief["sources"] = srcs
    return brief

# ------------------------- 4) Sidebar controls --------------------------------
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

# Mirror sidebar to env for backend tools
os.environ["MAX_SOURCES"] = str(max_sources)
os.environ["MIN_NON_EMPTY_SOURCES"] = str(min_non_empty)
os.environ["LLM_MODE"] = llm_mode
os.environ["LANGSMITH_ENABLED"] = "true" if tracing_on else "false"
os.environ["HTTP_TIMEOUT"] = str(http_timeout)

# ------------------------- 5) Run + display -----------------------------------
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

        try:
            brief = _run_pipeline(q)
        except Exception as e:
            status.update(label="Error ❌", state="error")
            st.exception(e)
            st.stop()

        brief = _inject_seed_url(q, brief)
        brief = _guarantee_full_markdown(q, brief)
        status.update(label="Done ✅", state="complete")

    # Save artifacts in a Cloud-writable temp dir
    art_dir = _safe_artifacts_dir()
    md = brief.get("_markdown") or render_markdown_brief(brief)
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
    st.info("Enter a topic or a URL in the sidebar and click **Run**.")