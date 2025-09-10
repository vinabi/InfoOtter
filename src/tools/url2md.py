import os, time, requests
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
HTTP_TIMEOUT   = int(os.getenv("HTTP_TIMEOUT","20"))

UA = {"User-Agent": "Mozilla/5.0 (MarketBrief/1.0; +https://example.com)"}

def _retry(fn, tries=3, backoff=0.75):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(backoff*(2**i))
    raise last

def _rapidapi_convert(url: str) -> str:
    if not RAPIDAPI_KEY:
        raise RuntimeError("No RAPIDAPI_KEY")
    endpoint = f"{URL2MD_BASE}{URL2MD_ENDPOINT}"
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": URL2MD_HOST, "Content-Type": "application/json", **UA}
    payload = {"url": url, "returnType": "markdown"}
    def go():
        r = requests.post(endpoint, json=payload, headers=headers, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "markdown" in data: return data["markdown"]
        if isinstance(data, str): return data
        return str(data)
    return _retry(go)

def _tavily_extract(url: str) -> str:
    if not TAVILY_API_KEY: return ""
    endpoint = "https://api.tavily.com/extract"
    payload = {"api_key": TAVILY_API_KEY, "url": url}
    def go():
        r = requests.post(endpoint, json=payload, timeout=HTTP_TIMEOUT)
        if r.status_code >= 400: return ""
        data = r.json()
        title = data.get("title") or url
        text  = data.get("content") or ""
        return f"# {title}\n\n{text}"
    try:
        return _retry(go)
    except Exception:
        return ""

def _jina_reader(url: str) -> str:
    # works without a key; good on Cloud
    try:
        r = requests.get(f"https://r.jina.ai/http://{url.split('://',1)[-1]}", headers=UA, timeout=HTTP_TIMEOUT)
        if r.status_code < 400 and r.text.strip(): return r.text
    except Exception:
        pass
    return ""

def _local_markdownify(url: str) -> str:
    try:
        r = requests.get(url, headers=UA, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        if _to_md: return _to_md(r.text, heading_style="ATX")
        return unescape(r.text)
    except Exception:
        return ""

def url_to_markdown(url: str, timeout: int | None = None) -> str:
    # Priority: RapidAPI → Tavily Extract → Jina → Local
    for fn in (
        lambda: _rapidapi_convert(url),
        lambda: _tavily_extract(url),
        lambda: _jina_reader(url),
        lambda: _local_markdownify(url),
    ):
        try:
            md = fn()
            if md and md.strip(): return md
        except Exception:
            continue
    return f"# Unable to convert\n\nFailed to fetch/convert: {url}\n"