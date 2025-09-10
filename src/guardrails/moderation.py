import re
TOXIC_PATTERNS = [r"(?i)\bkill\b", r"(?i)\bhate\b", r"(?i)\bslur\b"]
def basic_moderation(text: str) -> bool:
    if not text:
        return True
    for pat in TOXIC_PATTERNS:
        if re.search(pat, text):
            return False
    return True
