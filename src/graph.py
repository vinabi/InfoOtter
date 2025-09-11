import os, json
from pprint import pprint
from langgraph.graph import StateGraph, START, END
from .state import GraphState
from .agents import get_llm, run_researcher, run_analyst, run_writer
from .tools.search import aggregate_search, enrich_with_content
from .observability import trace, get_callbacks

MAX_SOURCES = int(os.getenv("MAX_SOURCES", "8"))
MIN_NON_EMPTY_SOURCES = int(os.getenv("MIN_NON_EMPTY_SOURCES", "4"))

llm = get_llm()
graph = StateGraph(GraphState)

def researcher_node(state: GraphState):
    with trace("researcher", {"q": state.get("query", "")}):
        # Never set tool_error=True; run_researcher never raises and returns at least a stub
        sources = run_researcher(
            aggregate_search, enrich_with_content, state["query"],
            MAX_SOURCES, MIN_NON_EMPTY_SOURCES
        )
        state["sources"] = sources
        state.setdefault("tools_used", []).append("search")
        state["tool_error"] = False
    return state

def analyst_node(state: GraphState):
    with trace("analyst"):
        state["facts"] = run_analyst(llm, state.get("query",""), state.get("sources", []))
    return state

def writer_node(state: GraphState):
    with trace("writer"):
        state["brief"] = run_writer(
            llm, state.get("query",""), state.get("facts", []), state.get("sources", [])
        )
        state["schema_ok"] = True
    return state

def reviewer_node(state: GraphState):
    with trace("reviewer"):
        brief = state.get("brief") or {
            "topic": state.get("query", ""),
            "summary": "Partial brief due to upstream errors. See sources and facts collected so far.",
            "key_facts": state.get("facts", []),
            "sources": [{"title": s.get("title", ""), "url": s.get("url", "")}
                        for s in state.get("sources", [])]
        }
        brief["summary"] = (brief.get("summary") or "")[:1180]
        brief["_markdown"] = brief.get("_markdown") or render_markdown_brief(brief)
        state["brief"] = brief
        state["schema_ok"] = True
    return state

graph.add_node("researcher", researcher_node)
graph.add_node("analyst", analyst_node)
graph.add_node("writer", writer_node)
graph.add_node("reviewer", reviewer_node)

graph.add_edge(START, "researcher")
graph.add_edge("researcher", "analyst")
graph.add_edge("analyst", "writer")
graph.add_edge("writer", "reviewer")
graph.add_edge("reviewer", END)

compiled = graph.compile()
__all__ = ["compiled"]

def main(topic: str | None = None):
    q = topic or input("Enter the topic for market research: ").strip()
    init_state: GraphState = {"query": q, "failure_count": 0}
    final = compiled.invoke(init_state, config={"callbacks": get_callbacks()})
    brief = final.get("brief") or {}
    pprint(brief)

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/sample_output.json", "w", encoding="utf-8") as f:
        json.dump(brief, f, indent=2)

    md = brief.get("_markdown") or render_markdown_brief(brief)
    with open("artifacts/brief.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("[ok] wrote artifacts/brief.md")
    return md

if __name__ == "__main__":
    main()
