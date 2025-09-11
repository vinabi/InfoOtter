# app.py — Streamlit wrapper that enriches URLs via Jina Reader
import os, sys, json, tempfile, requests
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# ------------------------- Secrets → env -------------------------------------
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

# ------------------------- Make 'src' importable -----------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import your compiled graph + writer
from src.graph import compiled
from src.agents import run_writer, get_llm
try:
    from src.observability import get_callbacks
except Exception:
    def get_callbacks(): return []

# ------------------------- Jina Reader helper --------------------------------
def jina_extract(url: str, timeout: int = 20) -> str:
    """Fetch content as Markdown-like text via Jina Reader API."""
    try:
        api = f"https://r.jina.ai/http://{url.split('://',1)[-1]}"
        r = requests.get(api, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        return f"# [Jina fetch failed] {url}\n\n{e}\n"

# ------------------------- Markdown fallback ---------------------------------
def render_markdown_brief(brief: Dict[str, Any]) -> str:
    topic = brief.get("topic", "")
    summary = brief.get("summary", "")
    facts = brief.get("key_facts") or []
    sources = brief.get("sources") or []
    lines = [f"# Market Brief: {topic}", "", summary, ""]
    if facts:
        lines.append("## Key Facts")
        for f in facts:
            lines.append(f"- {f.get('fact','')}")
        lines.append("")
    if sources:
        lines.append("## References")
        for i, s in enumerate(sources, 1):
            lines.append(f"{i}. [{s.get('title','')}]({s.get('url','')})")
        lines.append("")
    return "\n".join(lines)

# ------------------------- Streamlit UI --------------------------------------
st.set_page_config(page_title="Market Brief Agent", page_icon="📈", layout="wide")
st.title("📈 Market Brief Agent")
st.caption("Runs your chain, then fetches all URLs via Jina Reader to guarantee full content.")

with st.sidebar:
    st.header("Settings")
    query = st.text_area(
        "Enter a topic or a URL",
        value=os.getenv("QUERY", ""),
        height=90,
        placeholder="e.g., voice search optimization OR https://example.com/post"
    )
    http_timeout = st.slider("HTTP_TIMEOUT (s)", 5, 60, int(os.getenv("HTTP_TIMEOUT", "20")))
    run_btn = st.button("▶️ Run", type="primary", use_container_width=True)

# ------------------------- Run pipeline --------------------------------------
left, right = st.columns([2, 1])

def _artifacts_dir() -> Path:
    p = Path(tempfile.gettempdir()) / "market_agent_artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p

if run_btn:
    q = (query or "").strip()
    if not q:
        st.error("Please enter a topic or a URL.")
        st.stop()

    with st.status("Running pipeline…", expanded=True) as status:
        status.write("• Invoking compiled LangGraph")
        try:
            init_state = {"query": q, "failure_count": 0}
            final = compiled.invoke(init_state, config={"callbacks": get_callbacks()})
            brief = (final or {}).get("brief") or {}
        except Exception as e:
            status.update(label="Error ❌", state="error")
            st.exception(e)
            st.stop()

        # Enrich URLs via Jina
        sources = brief.get("sources") or []
        enriched_sources: List[Dict[str, Any]] = []
        for s in sources:
            url = s.get("url")
            if not url:
                continue
            md_text = jina_extract(url, timeout=http_timeout)
            enriched_sources.append({**s, "content": md_text})
        if enriched_sources:
            brief["sources"] = enriched_sources

        # Re-run writer with enriched sources
        status.write("• Rewriting with Jina-enriched sources")
        try:
            llm = get_llm()
            facts = brief.get("key_facts") or []
            brief2 = run_writer(llm, q, facts, enriched_sources)
            if isinstance(brief2, dict):
                brief.update(brief2)
        except Exception as e:
            st.warning(f"Writer failed: {e}")

        md = (brief.get("_markdown") or "").strip()
        if not md:
            md = render_markdown_brief(brief)

        status.update(label="Done ✅", state="complete")

    # Save artifacts
    outdir = _artifacts_dir()
    (outdir / "brief.md").write_text(md, encoding="utf-8")
    (outdir / "sample_output.json").write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")

    # Show output
    with left:
        st.subheader("Brief (Markdown)")
        st.markdown(md)
        st.download_button("💾 Download Markdown", md.encode("utf-8"), "brief.md", "text/markdown", use_container_width=True)
        st.download_button("💾 Download JSON", json.dumps(brief, indent=2, ensure_ascii=False).encode("utf-8"),
                           "sample_output.json", "application/json", use_container_width=True)

    with right:
        st.subheader("Sources")
        if sources:
            df = pd.DataFrame([{"title": s.get("title",""), "url": s.get("url","")} for s in sources])
            st.dataframe(df, width="stretch", hide_index=True)
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
            st.dataframe(df_f, width="stretch", hide_index=True)
        else:
            st.info("No extracted facts available.")

else:
    st.info("Enter a topic or a URL in the sidebar and click **Run**.")