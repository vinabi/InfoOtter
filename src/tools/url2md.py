import os, requests
from html import unescape
try:
    from markdownify import markdownify as _to_md
except Exception:
    _to_md = None

RAPIDAPI_KEY   = os.getenv("RAPIDAPI_KEY")
URL2MD_HOST    = os.getenv("URL2MD_HOST") or "url-to-markdown-api.p.rapidapi.com"
URL2MD_BASE    = os.getenv("URL2MD_BASE") or "https://url-to-markdown-api.p.rapidapi.com"
URL2MD_ENDPOINT= os.getenv("URL2MD_ENDPOINT") or "/convert"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HTTP_TIMEOUT   = int(os.getenv("HTTP_TIMEOUT","15"))

def _rapidapi_convert(url: str) -> str:
    if not RAPIDAPI_KEY: raise RuntimeError("No RAPIDAPI_KEY")
    endpoint = f"{URL2MD_BASE}{URL2MD_ENDPOINT}"
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": URL2MD_HOST, "Content-Type": "application/json"}
    payload = {"url": url, "returnType": "markdown"}
    r = requests.post(endpoint, json=payload, headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "markdown" in data: return data["markdown"]
    if isinstance(data, str): return data
    return str(data)

def _tavily_extract(url: str) -> str:
    if not TAVILY_API_KEY: return ""
    endpoint = "https://api.tavily.com/extract"
    payload = {"api_key": TAVILY_API_KEY, "url": url}
    r = requests.post(endpoint, json=payload, timeout=HTTP_TIMEOUT)
    if r.status_code >= 400: return ""
    data = r.json()
    title = data.get("title") or url
    text  = data.get("content") or ""
    return f"# {title}\n\n{text}"

def _jina_reader(url: str) -> str:
    try:
        r = requests.get(f"https://r.jina.ai/http://{url.split('://',1)[-1]}", timeout=HTTP_TIMEOUT)
        if r.status_code < 400 and r.text.strip():
            return r.text
    except Exception:
        pass
    return ""

def url_to_markdown(url: str, timeout: int = None) -> str:
    # Priority: RapidAPI → Tavily Extract → Jina → local markdownify
    try:
        return _rapidapi_convert(url)
    except Exception:
        pass
    md = _tavily_extract(url)
    if md: return md
    md = _jina_reader(url)
    if md: return md
    try:
        resp = requests.get(url, headers={"User-Agent":"Mozilla/5.0 (MarketBriefBot/1.0)"}, timeout=timeout or HTTP_TIMEOUT)
        resp.raise_for_status()
        html = resp.text
        if _to_md: return _to_md(html, heading_style="ATX")
        return unescape(html)
    except Exception:
        return f"# Unable to convert\n\nFailed to fetch/convert: {url}\n"
        
    resp = requests.get(url, headers={"User-Agent":"Mozilla/5.0 (MarketBriefBot/1.0)"}, timeout=timeout or HTTP_TIMEOUT)
    resp.raise_for_status()
    html = resp.text
    if _to_md: return _to_md(html, heading_style="ATX")

