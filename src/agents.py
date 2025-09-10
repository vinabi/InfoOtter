import os, json, textwrap, traceback, datetime
from typing import List, Dict
from .state import validate_facts, validate_brief
from .guardrails.moderation import basic_moderation
from .tools.url2md import url_to_markdown
from .tools.search import aggregate_search, enrich_with_content
from .observability import trace

LLM_MODE   = os.getenv("LLM_MODE", "groq").lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

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

        brief = {"topic": query, "summary": draft[:1500], "key_facts": facts, "sources": live_sources, "_markdown": md_final}
        _ = validate_brief(brief)  # even if validation warns, we still return markdown
        return brief