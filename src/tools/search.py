# src/tools/search.py
import os
import requests
from typing import List, Dict
from ..fallbacks import with_retries

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "15"))

# ---- DuckDuckGo (ddgs, maintained) ------------------------------------------
def _ddg_search(query: str, max_results: int) -> List[Dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        return []
    out = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                out.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "description": r.get("body", ""),
                })
        return out
    except Exception:
        return []

# ---- Wikipedia search -------------------------------------------------------
@with_retries
def wikipedia_search(query: str, max_results: int = 8) -> List[Dict]:
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "opensearch",
        "search": query,
        "limit": max_results,
        "namespace": 0,
        "format": "json"
    }
    r = requests.get(api, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    titles, descs, urls = data[1], data[2], data[3]
    out = []
    for t, d, u in zip(titles, descs, urls):
        out.append({"title": t, "url": u, "description": d})
    return out

def wikipedia_summary_from_url(url: str) -> str:
    import urllib.parse as up
    if "/wiki/" in url:
        title = up.unquote(url.split("/wiki/", 1)[1])
    else:
        title = url
    api = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "prop": "extracts", "explaintext": 1,
              "titles": title, "format": "json"}
    r = requests.get(api, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "")[:4000]

# ---- Aggregator -------------------------------------------------------------
def aggregate_search(query: str, max_results: int = 8) -> List[Dict]:
    ddg = _ddg_search(query, max_results=max_results) or []
    if ddg:
        return ddg[:max_results]
    try:
        return wikipedia_search(query, max_results=max_results)
    except Exception:
        return []

def enrich_with_content(results: List[Dict]) -> List[Dict]:
    enriched = []
    for r in results:
        url = r.get("url") or ""
        content = ""
        if "wikipedia.org/wiki/" in url:
            try:
                content = wikipedia_summary_from_url(url)
            except Exception:
                content = ""
        enriched.append({**r, "content": content})
    return enriched