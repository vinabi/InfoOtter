from bs4 import BeautifulSoup
def extract_tables(html: str) -> list[list[list[str]]]:
    soup = BeautifulSoup(html, "lxml")
    tables = []
    for t in soup.find_all("table"):
        rows = []
        for tr in t.find_all("tr"):
            cells = [c.get_text(strip=True) for c in tr.find_all(["th","td"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables
