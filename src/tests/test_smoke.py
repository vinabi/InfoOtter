import os
os.environ["LLM_MODE"] = "groq"
from src.graph import compiled
from src.state import Brief
def test_end_to_end_groq():
    out = compiled.invoke({"query": "A2A and MCP in AI"})
    brief = out["brief"]
    Brief(**brief)
    assert brief["topic"]
    assert len(brief["sources"]) >= 1
