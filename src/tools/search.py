import os, re, time, requests
from typing import List, Dict, Tuple
from urllib.parse import urlencode
from bs4 import BeautifulSoup  # add to requirements if missing

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "15"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERP_API_KEY   = os.getenv("SERP_API_KEY")
NEWSAPI_KEY    = os.getenv("NEWSAPI_KEY")

# ---------- Utilities ----------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def _dedup(items: List[Dict], max_results: int) -> List[Dict]:
    seen = set()
    out = []
    for it in items:
        url = (it.get("url") or "").strip()
        if not url:
            continue
        key = (_norm(it.get("title",""))[:120], url.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= max_results:
            break
    return out

def expand_queries(topic: str) -> List[str]:
    """Generic expansion: quoted/unquoted, plus mix-in of common market modifiers."""
    t = topic.strip()
    if not t:
        return []
    base = [_norm(t)]
    # quoted version
    if " " in t:
        base.append(f"\"{t}\"")
    # split by commas/slashes and join variants
    parts = re.split(r"[,/|]+", t)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        base.append(" ".join(parts))
    # market modifiers (configurable)
    mod_env = os.getenv("QUERY_EXPANSION_MODIFIERS", "market size; vendors; tooling; spec; roadmap; 2024; 2025; news; open-source")
    modifiers = [m.strip() for m in mod_env.split(";") if m.strip()]
    for m in modifiers:
        base.append(f"{t} {m}")
    # unique + cap
    uniq = []
    seen = set()
    for q in base:
        nq = _norm(q)
        if nq not in seen:
            uniq.append(q)
            seen.add(nq)
    return uniq[:8]

# ---------- Providers ----------
def tavily_search(query: str, max_results: int = 10) -> List[Dict]:
    if not TAVILY_API_KEY:
        return []
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
        "topic": "general",
    }
    r = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json() or {}
    out = []
    for res in (data.get("results") or [])[:max_results]:
        out.append({
            "title": res.get("title",""),
            "url": res.get("url",""),
            "description": (res.get("content") or res.get("snippet") or "")[:600],
            "published_at": res.get("published_time"),
            "source": "tavily",
            "is_stub": False
        })
    return out

def serp_search(query: str, max_results: int = 10) -> List[Dict]:
    if not SERP_API_KEY:
        return []
    url = "https://serpapi.com/search.json"
    params = {"engine":"google","q":query,"num":max_results,"api_key":SERP_API_KEY,"hl":"en"}
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json() or {}
    out = []
    for item in (data.get("organic_results") or [])[:max_results]:
        out.append({
            "title": item.get("title") or "",
            "url": item.get("link") or "",
            "description": item.get("snippet") or "",
            "source": "serpapi",
            "is_stub": False
        })
    return out

def newsapi_search(query: str, max_results: int = 10) -> List[Dict]:
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "pageSize": max_results, "language": "en", "sortBy": "relevancy", "apiKey": NEWSAPI_KEY}
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json() or {}
    out = []
    for a in (data.get("articles") or [])[:max_results]:
        out.append({
            "title": a.get("title") or "",
            "url": a.get("url") or "",
            "description": (a.get("description") or "")[:500],
            "published_at": a.get("publishedAt"),
            "source": "newsapi",
            "is_stub": False,
        })
    return out

def ddg_api_search(query: str, max_results: int = 10) -> List[Dict]:
    try:
        from ddgs  import DDGS
    except Exception:
        return []
    try:
        out = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                out.append({
                    "title": r.get("title",""),
                    "url": r.get("href",""),
                    "description": r.get("body",""),
                    "source": "ddg",
                    "is_stub": False
                })
        return out
    except Exception:
        return []

def ddg_html_search(query: str, max_results: int = 10) -> List[Dict]:
    """HTML fallback that often works when the python package is blocked."""
    try:
        qs = urlencode({"q": query})
        url = f"https://duckduckgo.com/html/?{qs}"
        r = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        out = []
        for res in soup.select("a.result__a"):
            href = res.get("href")
            title = res.get_text(strip=True)
            if href and title:
                out.append({"title": title, "url": href, "description": "", "source":"ddg_html", "is_stub": False})
                if len(out) >= max_results:
                    break
        return out
    except Exception:
        return []

def wikipedia_search(query: str, max_results: int = 6) -> List[Dict]:
    api = "https://en.wikipedia.org/w/api.php"
    params = {"action":"opensearch","search":query,"limit":max_results,"namespace":0,"format":"json"}
    r = requests.get(api, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    titles, descs, urls = data[1], data[2], data[3]
    out = []
    for t, d, u in zip(titles, descs, urls):
        out.append({"title": t, "url": u, "description": d, "source":"wikipedia", "is_stub": False})
    return out

# ---------- Aggregate across expansions ----------
def aggregate_search(topic: str, max_results: int = 12) -> List[Dict]:
    queries = expand_queries(topic)
    collected: List[Dict] = []

    # Priority: Tavily → Serp → News → DDG API → DDG HTML → Wikipedia
    providers = [tavily_search, serp_search, newsapi_search, ddg_api_search, ddg_html_search, wikipedia_search]

    for q in queries:
        for fn in providers:
            try:
                res = fn(q, max_results=6)
                if res:
                    collected.extend(res)
            except Exception:
                continue

    # Deduplicate and cap
    return _dedup(collected, max_results=max_results)

# ---------- “Enrichment” placeholder ----------
def enrich_with_content(results: List[Dict]) -> List[Dict]:
    """We keep descriptions; full content extraction happens later in url2md."""
    return [{**r, "content": r.get("description","")} for r in results]
