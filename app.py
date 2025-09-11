# app.py — URL → Report (Streamlit + Jina Reader + optional Groq LLM)
import os
import re
import math
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ----------------------------- Setup ---------------------------------
load_dotenv(override=False)

st.set_page_config(page_title="URL → Report", page_icon="🧩", layout="wide")

st.markdown(
    """
    <style>
      .callout {background:#3D155F; color:#fff; border:1px solid #2a0e43; border-radius:10px; padding:12px 14px;}
      .muted {opacity:.85}
      code, pre {white-space: pre-wrap !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# ------------------------ Utility Functions --------------------------
URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+")
STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","when","while","of","to","in","on","for","with",
    "as","by","from","at","is","it","this","that","these","those","be","been","are","was","were","will",
    "can","may","might","should","would","could","we","you","they","he","she","i","me","my","our","your",
    "their","them","his","her","its","about","into","over","under","between","within","per","via","not",
}

def extract_urls(block: str) -> List[str]:
    seen, out = set(), []
    for m in URL_RE.finditer(block or ""):
        u = m.group(0).strip().rstrip(".,);]")
        if u not in seen:
            seen.add(u); out.append(u)
    return out

# ---------- Fetch via Jina Reader (Markdown-like) ----------
def fetch_via_jina(url: str, timeout: int = 20) -> Tuple[str, str]:
    api = f"https://r.jina.ai/http://{url.split('://',1)[-1]}"
    r = requests.get(api, timeout=timeout)
    r.raise_for_status()
    md = r.text
    # crude title guess = first markdown heading or the URL
    title = url
    for line in md.splitlines():
        if line.startswith("#"):
            title = line.lstrip("# ").strip() or url
            break
    return title, md

# ---------- Local HTML → text (fallback) ----------
def fetch_html(url: str, timeout: int = 15, max_bytes: int = 5_000_000) -> Tuple[str, str]:
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    content = r.content[:max_bytes]
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "aside"]):
        tag.decompose()
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            title = og["content"].strip()
    text = " ".join(t.strip() for t in soup.stripped_strings)
    return (title or url, text)

# ---------------------- Summarization Engines ------------------------
def split_sentences(text: str) -> List[str]:
    text = (text or "").replace("\n", " ")
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def score_sentences(text: str) -> List[Tuple[float, str]]:
    sentences = split_sentences(text)
    if not sentences: return []
    freqs: Dict[str, float] = {}
    for s in sentences:
        for w in WORD_RE.findall(s.lower()):
            if w in STOPWORDS or len(w) <= 2: continue
            freqs[w] = freqs.get(w, 0) + 1.0
    if freqs:
        maxf = max(freqs.values())
        for k in list(freqs.keys()):
            freqs[k] /= maxf
    scored: List[Tuple[float, str]] = []
    for s in sentences:
        score = 0.0
        for w in WORD_RE.findall(s.lower()):
            score += freqs.get(w, 0.0)
        n_words = max(1, len(WORD_RE.findall(s)))
        length_penalty = 0.8 if (n_words < 8 or n_words > 40) else 1.0
        scored.append((score * length_penalty, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

def summarize_extractive(text: str, max_sentences: int = 5) -> List[str]:
    ranked = score_sentences(text)
    if not ranked: return []
    top = {s for _, s in ranked[: max(1, max_sentences * 2)]}
    sentences = split_sentences(text)
    kept: List[str] = []
    for s in sentences:
        if s in top: kept.append(s)
        if len(kept) >= max_sentences: break
    return kept

# ---------- Optional Groq LLM summarizer ----------
GROQ_KEY = os.getenv("GROQ_API_KEY")
try:
    if GROQ_KEY:
        from langchain_groq import ChatGroq  # lightweight LC wrapper
        _groq = ChatGroq(model_name=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"), temperature=0.2)
        LLM_AVAILABLE = True
    else:
        _groq = None
        LLM_AVAILABLE = False
except Exception:
    _groq = None
    LLM_AVAILABLE = False

def summarize_with_groq(text: str, max_points: int = 5) -> List[str]:
    if not LLM_AVAILABLE or not _groq:
        return summarize_extractive(text, max_points)
    prompt = (
        "Summarize the following webpage content into concise bullet points "
        f"(max {max_points}). Focus on key facts, claims, metrics, guidance.\n\n"
        "TEXT:\n" + text[:120000]
    )
    try:
        resp = _groq.invoke(prompt)
        msg = getattr(resp, "content", "") or ""
        bullets = [re.sub(r"^[\\-\\d\\.\\)\\s]+", "", ln).strip() for ln in msg.splitlines() if ln.strip()]
        return [b for b in bullets if b][:max_points]
    except Exception:
        return summarize_extractive(text, max_points)

# --------------------------- Reporting -------------------------------
def build_markdown_report(topic: str, items: List[Dict], cite: bool = True) -> str:
    lines: List[str] = []
    lines.append(f"# Report: {topic or 'Collected URLs'}\n")

    pooled: List[Tuple[float, str, int]] = []
    for idx, it in enumerate(items):
        for s, sc in it.get("scored", [])[:6]:
            pooled.append((sc, s, idx))
    pooled.sort(key=lambda x: x[0], reverse=True)
    exec_pts, seen = [], set()
    for _, s, _ in pooled:
        if s.lower() in seen: continue
        exec_pts.append(s); seen.add(s.lower())
        if len(exec_pts) >= 6: break

    if exec_pts:
        lines.append("## Executive Summary")
        for s in exec_pts: lines.append(f"- {s}")
        lines.append("")

    for i, it in enumerate(items, 1):
        title = it.get("title") or f"Source {i}"
        url = it.get("url") or ""
        lines.append(f"## {title}")
        if url: lines.append(f"Source: {url}")
        bullets = it.get("bullets") or []
        if bullets:
            for b in bullets: lines.append(f"- {b}")
        else:
            lines.append("- (No content extracted)")
        lines.append("")

    if cite:
        lines.append("## References")
        for i, it in enumerate(items, 1):
            u = it.get("url", "")
            t = (it.get("title") or u or f"Source {i}").strip()
            lines.append(f"{i}. [{t}]({u})")
        lines.append("")
    return "\n".join(lines)

# ---------------------------- UI ------------------------------------
with st.sidebar:
    st.header("Settings")
    use_jina = st.toggle("Use Jina Reader (no key)", value=True, help="Reads pages as Markdown via r.jina.ai")
    use_llm = st.toggle("Use Groq summarizer", value=bool(GROQ_KEY), help="Needs GROQ_API_KEY in secrets/env")
    max_points = st.slider("Bullets per source", 3, 10, 5)
    timeout = st.slider("HTTP timeout (s)", 5, 45, int(os.getenv("HTTP_TIMEOUT", "20")))
    cite = st.toggle("Include references section", value=True)

st.title("URL → Report")
st.caption("Paste URLs or upload a file. I’ll fetch each page (Jina or HTML), summarize (Groq or extractive), then compile a Markdown report.")

left, right = st.columns([2, 1])

with left:
    url_text = st.text_area("Paste URLs or any text containing them", value=os.getenv("URLS", ""), height=160,
                            placeholder="One per line, or paste any text—I’ll auto-extract links…")
    uploaded = st.file_uploader("…or upload a .txt / .csv / .json file of URLs", type=["txt", "csv", "json"], accept_multiple_files=False)

    urls: List[str] = []
    if uploaded is not None:
        try:
            name = uploaded.name.lower()
            data = uploaded.read()
            if name.endswith(".txt"):
                urls = extract_urls(data.decode("utf-8", "ignore"))
            elif name.endswith(".csv"):
                df = pd.read_csv(uploaded)
                for col in df.columns:
                    vals = [str(v) for v in df[col].dropna().tolist()]
                    if any(v.startswith("http") for v in vals):
                        urls = [v for v in vals if v.startswith("http")]
                        break
            elif name.endswith(".json"):
                try:
                    obj = json.loads(data.decode("utf-8", "ignore"))
                except Exception:
                    obj = []
                if isinstance(obj, dict) and "urls" in obj and isinstance(obj["urls"], list):
                    urls = [str(u) for u in obj["urls"] if isinstance(u, str)]
                elif isinstance(obj, list):
                    urls = [str(u) for u in obj if isinstance(u, str)]
        except Exception as e:
            st.error(f"Failed to parse uploaded file: {e}")

    urls = (extract_urls(url_text) or []) if not urls else list(dict.fromkeys(urls + extract_urls(url_text)))

    col1, col2 = st.columns([1,1])
    with col1:
        run = st.button("Build report", type="primary")
    with col2:
        clear = st.button("Clear URLs")
        if clear:
            st.experimental_rerun()

with right:
    st.subheader("Detected URLs")
    if urls:
        st.dataframe(pd.DataFrame({"url": urls}), width="stretch", hide_index=True)
    else:
        st.info("No URLs detected yet.")

if 'results' not in st.session_state:
    st.session_state.results = []
if 'report_md' not in st.session_state:
    st.session_state.report_md = ""

if run:
    if not urls:
        st.error("Please provide at least one URL.")
    else:
        results: List[Dict] = []
        progress = st.progress(0.0, text="Fetching…")
        for i, u in enumerate(urls, 1):
            try:
                if use_jina:
                    title, text = fetch_via_jina(u, timeout=timeout)
                else:
                    title, text = fetch_html(u, timeout=timeout)
                if not text:
                    bullets, scored = [], []
                else:
                    if use_llm and LLM_AVAILABLE:
                        bullets = summarize_with_groq(text, max_points=max_points)
                        scored = [(s[0], s[1]) for s in score_sentences(" ".join(bullets))]
                    else:
                        bullets = summarize_extractive(text, max_points)
                        scored = score_sentences(text)[:12]
                results.append({"url": u, "title": title, "bullets": bullets, "scored": scored})
            except Exception as e:
                results.append({"url": u, "title": "(fetch failed)", "bullets": [f"Error: {e}"], "scored": []})
            progress.progress(i/len(urls), text=f"Processed {i}/{len(urls)}")
            time.sleep(0.05)

        st.session_state.results = results
        topic_guess = "Summaries for provided URLs"
        if len(results) == 1 and results[0].get("title"):
            topic_guess = results[0]["title"]
        st.session_state.report_md = build_markdown_report(topic_guess, results, cite=cite)

# -------------------------- Output Pane ------------------------------
if st.session_state.report_md:
    st.subheader("Report (Markdown)")
    st.markdown(st.session_state.report_md)

    (ARTIFACTS / "report.md").write_text(st.session_state.report_md, encoding="utf-8")
    (ARTIFACTS / "report.json").write_text(json.dumps(st.session_state.results, indent=2, ensure_ascii=False), encoding="utf-8")

    colA, colB = st.columns(2)
    with colA:
        st.download_button("Download report.md",
                           data=st.session_state.report_md.encode("utf-8"),
                           file_name="report.md", mime="text/markdown", width="stretch")
    with colB:
        st.download_button("Download raw report.json",
                           data=json.dumps(st.session_state.results, indent=2, ensure_ascii=False).encode("utf-8"),
                           file_name="report.json", mime="application/json", width="stretch")
else:
    st.markdown(
        '<div class="callout">Paste URLs in the box on the left, or upload a list, then click <b>Build report</b>.</div>',
        unsafe_allow_html=True,
    )
