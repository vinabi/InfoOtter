import requests
from bs4 import BeautifulSoup

def fetch_and_clean(url: str) -> str:
    try:
        response = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        for tag in soup.find_all(['script','style','noscript']):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines[:8000])
    except Exception:
        return ""
