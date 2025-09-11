import os, json, textwrap, traceback, datetime
from typing import List, Dict
from .state import validate_facts, validate_brief
from .guardrails.moderation import basic_moderation
from .tools.url2md import url_to_markdown
from .tools.search import aggregate_search, enrich_with_content
from .observability import trace
# --- Reference summarization helpers ---------------------------------
import re
from bs4 import BeautifulSoup

_REF_LINE = re.compile(r"^\s*\d+\.\s*\[(?P<title>[^\]]+)\]\((?P<url>[^)]+)\)\s*$")
_HEADERS = re.compile(r"(?im)^\s{0,3}#{2,3}\s+references\s*$")  # ## References / ### References

def _extract_links_from_references(md: str) -> list[dict]:
    """Return [{'title':..., 'url':...}, ...] from the References section."""
    if not md: return []
    lines = md.splitlines()
    refs_start = None
    for i, ln in enumerate(lines):
        if _HEADERS.match(ln):
            refs_start = i + 1
            break
    if refs_start is None:
        return []
    out = []
    for ln in lines[refs_start:]:
        if ln.strip().startswith("## ") and not ln.strip().lower().startswith("## references"):
            break  # next major section
        m = _REF_LINE.match(ln)
        if m:
            out.append({"title": m.group("title").strip(), "url": m.group("url").strip()})
    return out

HTTP_TIMEOUT = 12

def _jina_markdown(url: str, timeout: int = HTTP_TIMEOUT) -> str:
    """
    Dummy implementation for _jina_markdown.
    Replace this with the actual markdown extraction logic or import if available.
    """
    return ""

def _fetch_markdownish(url: str, timeout: int = HTTP_TIMEOUT) -> str:
    """Jina Reader first; fallback to HTML text via BeautifulSoup."""
    md = _jina_markdown(url, timeout=timeout)
    if md.strip():
        return md
    try:
        import requests
        _UA = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=_UA, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script","style","noscript","header","footer","nav","form","aside"]): tag.decompose()
        text = " ".join(t.strip() for t in soup.stripped_strings)
        return text
    except Exception:
        return ""

def _split_sentences(text: str) -> list[str]:
    text = (text or "").replace("\n", " ")
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def _score_sentences(text: str) -> list[tuple[float,str]]:
    WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+")
    STOP = {"the","a","an","and","or","but","if","then","else","when","while","of","to","in","on","for","with",
            "as","by","from","at","is","it","this","that","these","those","be","been","are","was","were","will",
            "can","may","might","should","would","could","we","you","they","he","she","i","me","my","our","your",
            "their","them","his","her","its","about","into","over","under","between","within","per","via","not"}
    sents = _split_sentences(text)
    if not sents: return []
    freqs: dict[str,float] = {}
    for s in sents:
        for w in WORD_RE.findall(s.lower()):
            if w in STOP or len(w) <= 2: continue
            freqs[w] = freqs.get(w, 0.0) + 1.0
    if freqs:
        mx = max(freqs.values())
        for k in list(freqs.keys()):
            freqs[k] /= mx
    scored: list[tuple[float,str]] = []
    for s in sents:
        score = sum(freqs.get(w,0.0) for w in WORD_RE.findall(s.lower()))
        n = max(1, len(WORD_RE.findall(s)))
        if n < 8 or n > 40: score *= 0.8
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

def _summarize_local(text: str, n: int = 5) -> list[str]:
    ranked = _score_sentences(text)
    if not ranked: return []
    keep = {s for _, s in ranked[:max(1, n*2)]}
    out = []
    for s in _split_sentences(text):
        if s in keep:
            out.append(s)
        if len(out) >= n: break
    return out

def _summarize_with_llm(llm, text: str, n: int = 5) -> list[str]:
    """Use your Groq LLM if available; else local extractive."""
    if isinstance(llm, StubLLM):
        return _summarize_local(text, n)
    prompt = (
        "Summarize the following page into concise bullet points "
        f"(max {n}). Focus on concrete findings, numbers, names, guidance.\n\n"
        "TEXT:\n" + (text or "")[:120000]
    )
    try:
        resp = llm.invoke(prompt)
        msg = getattr(resp, "content", "") or ""
        bullets = [re.sub(r"^[\\-\\d\\.\\)\\s]+","",ln).strip() for ln in msg.splitlines() if ln.strip()]
        return [b for b in bullets if b][:n] or _summarize_local(text, n)
    except Exception:
        return _summarize_local(text, n)

def _append_reference_summaries(md: str, llm, max_points: int = 5) -> str:
    """Build '## Reference Summaries' from links in '## References'."""
    links = _extract_links_from_references(md)
    if not links:
        return md
    sections = ["## Reference Summaries"]
    for i, it in enumerate(links, 1):
        url = it["url"]; title = it.get("title") or url
        body = _fetch_markdownish(url)
        if not body.strip():
            sections.append(f"### {i}. {title}\n- (Content unavailable)\n")
            continue
        bullets = _summarize_with_llm(llm, body, n=max_points)
        sections.append(f"### {i}. {title}\n" + "\n".join(f"- {b}" for b in bullets) + "\n")
    # Append just before the next top-level section after References, else at end
    return md.rstrip() + "\n\n" + "\n".join(sections) + "\n"

LLM_MODE   = os.getenv("LLM_MODE", "groq").lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

class StubLLM:
    def invoke(self, prompt: str):
        class R: 
            def __init__(self, text): self.content = text
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()][:28]
        return R("\n".join(lines) + "\n\n(Stub summary)")

def get_llm():
    if LLM_MODE == "groq" and os.getenv("GROQ_API_KEY"):
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(model_name=GROQ_MODEL, temperature=0.2)
        except Exception:
            pass
    return StubLLM()

# ---------- Researcher ----------
def _score_result(q_terms: List[str], it: Dict) -> float:
    text = (it.get("title","") + " " + it.get("description","")).lower()
    term_hits = sum(1 for t in q_terms if t in text)
    # Recency boost if published_at present
    recency = 0.0
    dt = it.get("published_at")
    if dt:
        try:
            # handle isoformats
            d = datetime.datetime.fromisoformat(dt.replace("Z","+00:00")).timestamp()
            now = datetime.datetime.now(datetime.timezone.utc).timestamp()
            age_days = max(1.0, (now - d) / 86400)
            recency = 1.0 / age_days  # newer => bigger
        except Exception:
            pass
    return term_hits + 0.1 * recency

def run_researcher(search_fn, enrich_fn, query: str, max_sources: int, min_non_empty: int) -> List[Dict]:
    with trace("researcher", {"q": query}):
        try:
            raw = search_fn(query, max_results=max_sources * 3)  # collect wider
            enriched = enrich_fn(raw)
            q_terms = [t for t in query.lower().split() if len(t) > 2]
            enriched.sort(key=lambda it: _score_result(q_terms, it), reverse=True)
            # keep unique domains first
            seen_domains, chosen = set(), []
            for it in enriched:
                url = it.get("url","")
                dom = url.split("/")[2] if "://" in url else url.split("/")[0]
                if dom in seen_domains:
                    continue
                seen_domains.add(dom)
                chosen.append(it)
                if len(chosen) >= max_sources:
                    break
            return chosen or enriched[:max_sources]
        except Exception:
            traceback.print_exc()
            return []

# ---------- Analyst ----------
def _facts_prompt(query: str, sources: List[Dict]) -> str:
    snippets = []
    for i, s in enumerate(sources[:8], 1):
        t = s.get("title") or s.get("url","")
        u = s.get("url","")
        c = (s.get("description") or s.get("content") or "")[:900]
        snippets.append(f"{i}) {t}\nURL: {u}\n{c}")
    return textwrap.dedent(f"""\
    You are a precise market analyst. From the source snippets below, extract 6 concise facts about **{query}**.
    Each fact MUST include an "evidence_url" from the provided URLs and a numeric "confidence" 0-1.
    Return ONLY valid JSON list: [{{"fact":"...", "evidence_url":"...", "confidence":0.7}}, ...].

    SOURCE SNIPPETS:
    {os.linesep*2}{os.linesep.join(snippets)}
    """)

def run_analyst(llm, query: str, sources: List[Dict]) -> List[Dict]:
    with trace("analyst"):
        try:
            resp = llm.invoke(_facts_prompt(query, sources)).content
            facts = json.loads(resp)
        except Exception:
            fallback_url = (sources[0].get("url") if sources else "https://example.com")
            facts = [{"fact": f"Market for {query} shows active ecosystem of tools and protocols.",
                      "evidence_url": fallback_url, "confidence": 0.55}]
        err = validate_facts(facts)
        if err: facts = [facts[0]]
        return facts

# ---------- Writer ----------
def _writer_prompt(query: str, source_sections_md: str, facts_json: str) -> str:
    return textwrap.dedent(f"""\
    Create a decision-ready market brief on **{query}** using ONLY the material in the sections below.
    Structure:
    - Executive Summary (≤ 6 sentences)
    - Key Insights (bullets, include inline [#] citations)
    - Competitive / Ecosystem Snapshot
    - Outlook (near-term)
    - References (numbered, provided; do not invent links)

    Use bracketed numeric citations [1], [2], etc. that map to the numbered References.

    ### Source Sections
    {source_sections_md}

    ### Extracted Facts (JSON)
    {facts_json}

    Return pure Markdown, no extra JSON.
    """)

def run_writer(llm, query: str, facts: List[Dict], sources: List[Dict]) -> Dict:
    with trace("writer"):
        if not basic_moderation(query):
            raise ValueError("Query failed moderation")
        if not sources:
            md = f"# Market Brief: {query}\n\n_No sources found._\n"
            return {"topic": query, "summary": md, "key_facts": facts, "sources": [], "_markdown": md}

        # Convert each URL to contextual markdown
        sections = []
        live_sources = []
        for idx, s in enumerate(sources[:10], 1):
            url = s.get("url"); title = s.get("title") or url
            if not url: continue
            try:
                md = url_to_markdown(url)
            except Exception:
                md = ""
            md_excerpt = "\n".join(md.splitlines()[:160])
            sections.append(f"#### [{idx}] {title}\n{md_excerpt}\n")
            live_sources.append({"title": title, "url": url, "published_at": s.get("published_at")})

        facts_json = json.dumps(facts, ensure_ascii=False, indent=2)
        source_sections_md = "\n\n---\n\n".join(sections)
        try:
            draft = llm.invoke(_writer_prompt(query, source_sections_md, facts_json)).content
        except Exception:
            draft = "Summary unavailable; see references below."

        refs = "\n".join([f"{i}. [{s['title']}]({s['url']})" for i, s in enumerate(live_sources, 1)])
        md_final = f"{draft}\n\n## References\n{refs}\n"

        # NEW: summarize each reference and append a section
        md_final = _append_reference_summaries(md_final, llm, max_points=5)

        brief = {"topic": query, "summary": (draft or "")[:1500], "key_facts": facts, "sources": live_sources, "_markdown": md_final}
        _ = validate_brief(brief)
        return brief

def render_markdown_brief(brief: dict) -> str:
    """Render a market brief dictionary as Markdown with summary, facts, and citations."""
    lines = []
    lines.append(f"# Market Brief: {brief.get('topic', '')}\n")
    summary = brief.get('summary', '').strip()
    if summary:
        lines.append(f"**Summary:** {summary}\n")
    facts = brief.get('key_facts')
    if facts:
        lines.append("## Key Facts")
        for fact in facts:
            # If fact is a dict, try to extract text and evidence
            if isinstance(fact, dict):
                fact_text = fact.get('fact') or str(fact)
                evidence = fact.get('evidence_url')
                if evidence:
                    lines.append(f"- {fact_text} ([source]({evidence}))")
                else:
                    lines.append(f"- {fact_text}")
            else:
                lines.append(f"- {fact}")
        lines.append("")
    sources = brief.get('sources')
    if sources:
        lines.append("## References")
        for i, s in enumerate(sources, 1):
            title = s.get('title', s.get('url', f'Source {i}'))
            url = s.get('url', '')
            lines.append(f"{i}. [{title}]({url})")
        lines.append("")
    return "\n".join(lines)
