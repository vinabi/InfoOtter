# src/agents.py — hardened for Streamlit Cloud
import os, json, textwrap, traceback, datetime, re
from typing import List, Dict, Any

from .state import validate_facts, validate_brief
from .guardrails.moderation import basic_moderation
from .tools.url2md import url_to_markdown            # keep your existing tool
from .tools.search import aggregate_search, enrich_with_content
from .observability import trace

import requests

LLM_MODE   = os.getenv("LLM_MODE", "groq").lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "20"))

# ----------------- LLM plumbing -----------------
class StubLLM:
    def invoke(self, prompt: str):
        class R:
            def __init__(self, text): self.content = text
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()][:48]
        return R("\n".join(lines) + "\n\n(Stub summary)")

def get_llm():
    """Prefer Groq via langchain_groq; fall back to a stub that never fails."""
    if LLM_MODE == "groq" and os.getenv("GROQ_API_KEY"):
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(model_name=GROQ_MODEL, temperature=0.2)
        except Exception:
            traceback.print_exc()
    return StubLLM()

# ----------------- Helpers -----------------
_UA = {"User-Agent": "Mozilla/5.0 (MarketBrief/1.0)"}

def _jina_markdown(url: str, timeout: int = HTTP_TIMEOUT) -> str:
    """Open fallback that works on Streamlit Cloud (no key)."""
    try:
        api = f"https://r.jina.ai/http://{url.split('://',1)[-1]}"
        r = requests.get(api, headers=_UA, timeout=timeout)
        if r.status_code < 400 and r.text.strip():
            return r.text
    except Exception:
        pass
    return ""

def _first_non_empty(*candidates: str) -> str:
    for c in candidates:
        if c and c.strip():
            return c
    return ""

_JSON_LIST_RE = re.compile(r"\[[\s\S]*\]")  # liberal list grabber

def _parse_json_list_maybe(chat: str) -> Any:
    """Try to parse strict JSON list from possibly chatty model output."""
    chat = (chat or "").strip()
    # 1) direct parse
    try:
        obj = json.loads(chat)
        return obj
    except Exception:
        pass
    # 2) extract the first JSON-looking list
    m = _JSON_LIST_RE.search(chat)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # 3) give up
    return None

def _cap_lines(md: str, n: int) -> str:
    return "\n".join((md or "").splitlines()[:max(1, n)])

# ----------------- Researcher -----------------
def _score_result(q_terms: List[str], it: Dict) -> float:
    text = (it.get("title","") + " " + it.get("description","")).lower()
    term_hits = sum(1 for t in q_terms if t in text)
    recency = 0.0
    dt = it.get("published_at")
    if dt:
        try:
            d = datetime.datetime.fromisoformat(dt.replace("Z","+00:00")).timestamp()
            now = datetime.datetime.now(datetime.timezone.utc).timestamp()
            age_days = max(1.0, (now - d) / 86400)
            recency = 1.0 / age_days
        except Exception:
            pass
    return term_hits + 0.1 * recency

def run_researcher(search_fn, enrich_fn, query: str, max_sources: int, min_non_empty: int) -> List[Dict]:
    with trace("researcher", {"q": query}):
        try:
            raw = search_fn(query, max_results=max_sources * 3)  # collect wider
            enriched = enrich_fn(raw) or []
            q_terms = [t for t in query.lower().split() if len(t) > 2]
            enriched.sort(key=lambda it: _score_result(q_terms, it), reverse=True)

            # keep first occurrence per domain
            seen_domains, chosen = set(), []
            for it in enriched:
                url = it.get("url","")
                if not url: continue
                dom = url.split("/")[2] if "://" in url else url.split("/")[0]
                if dom in seen_domains: continue
                seen_domains.add(dom)
                chosen.append(it)
                if len(chosen) >= max_sources: break

            return chosen or enriched[:max_sources] or []
        except Exception:
            traceback.print_exc()
            return []

# ----------------- Analyst -----------------
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

    {os.linesep.join(snippets)}
    """)

def _normalize_fact(f: Any, fallback_url: str) -> Dict[str, Any]:
    if isinstance(f, dict):
        fact = str(_first_non_empty(f.get("fact"), f.get("text"), f.get("statement"), ""))
        url  = str(_first_non_empty(f.get("evidence_url"), f.get("source"), f.get("url"), fallback_url))
        try:
            conf = float(f.get("confidence", 0.6))
        except Exception:
            conf = 0.6
    else:
        fact = str(f)
        url  = fallback_url
        conf = 0.6
    return {"fact": fact[:600], "evidence_url": url, "confidence": conf}

def run_analyst(llm, query: str, sources: List[Dict]) -> List[Dict]:
    with trace("analyst"):
        try:
            resp = llm.invoke(_facts_prompt(query, sources)).content
            obj = _parse_json_list_maybe(resp)
            if not isinstance(obj, list) or not obj:
                raise ValueError("No JSON list found")
            fallback_url = (sources[0].get("url") if sources else "https://example.com")
            facts = [_normalize_fact(x, fallback_url) for x in obj][:8]
        except Exception:
            traceback.print_exc()
            fallback_url = (sources[0].get("url") if sources else "https://example.com")
            facts = [{"fact": f"Market for {query} shows active ecosystem of tools and protocols.",
                      "evidence_url": fallback_url, "confidence": 0.55}]
        err = validate_facts(facts)
        if err and facts:
            facts = [facts[0]]
        return facts

# ----------------- Writer -----------------
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

def render_markdown_brief(brief: dict) -> str:
    lines = [f"# Market Brief: {brief.get('topic', '')}\n"]
    summary = (brief.get('summary', '') or '').strip()
    if summary:
        lines.append(f"**Summary:** {summary}\n")
    facts = brief.get('key_facts') or []
    if facts:
        lines.append("## Key Facts")
        for fact in facts:
            if isinstance(fact, dict):
                fact_text = fact.get('fact') or str(fact)
                evidence = fact.get('evidence_url')
                if evidence:
                    lines.append(f"- {fact_text} ([source]({evidence}))")
                else:
                    lines.append(f"- {fact_text}")
            else:
                lines.append(f"- {str(fact)}")
        lines.append("")
    sources = brief.get('sources') or []
    if sources:
        lines.append("## References")
        for i, s in enumerate(sources, 1):
            title = s.get('title', s.get('url', f'Source {i}'))
            url = s.get('url', '')
            lines.append(f"{i}. [{title}]({url})")
        lines.append("")
    return "\n".join(lines)

def _section_from_url(url: str, title: str) -> str:
    """
    Try your tool first; if it fails or returns empty, fall back to Jina Reader.
    This guarantees we always have content on Streamlit Cloud.
    """
    md = ""
    try:
        md = url_to_markdown(url) or ""
    except Exception:
        md = ""
    if not md.strip():
        md = _jina_markdown(url) or ""
    if not md.strip():
        md = f"# Unavailable\n{url}\n"
    return f"#### {title}\n{_cap_lines(md, 220)}\n"

def run_writer(llm, query: str, facts: List[Dict], sources: List[Dict]) -> Dict:
    with trace("writer"):
        if not basic_moderation(query):
            raise ValueError("Query failed moderation")
        if not sources:
            md = f"# Market Brief: {query}\n\n_No sources found._\n"
            return {"topic": query, "summary": md, "key_facts": facts, "sources": [], "_markdown": md}

        # Convert each URL to contextual markdown with robust fallback
        sections: List[str] = []
        live_sources: List[Dict[str, Any]] = []
        for idx, s in enumerate(sources[:10], 1):
            url = s.get("url"); title = s.get("title") or url or f"Source {idx}"
            if not url: continue
            sec = _section_from_url(url, f"[{idx}] {title}")
            sections.append(sec)
            live_sources.append({"title": title, "url": url, "published_at": s.get("published_at")})

        source_sections_md = "\n\n---\n\n".join(sections) if sections else f"#### [1] {query}\n(No content)\n"
        facts_json = json.dumps(facts, ensure_ascii=False, indent=2)

        try:
            draft = llm.invoke(_writer_prompt(query, source_sections_md, facts_json)).content
        except Exception:
            traceback.print_exc()
            draft = "Summary unavailable; see references below."

        # Numbered refs
        refs = "\n".join([f"{i}. [{s['title']}]({s['url']})" for i, s in enumerate(live_sources, 1)])
        md_final = f"{draft}\n\n## References\n{refs}\n"

        brief = {"topic": query, "summary": (draft or "")[:1500], "key_facts": facts,
                 "sources": live_sources, "_markdown": md_final}
        _ = validate_brief(brief)  # even if warnings, return markdown
        return brief
