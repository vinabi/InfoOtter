import os
from typing import List, Any
from contextlib import contextmanager
import json, time, pathlib

TRACE_DIR = pathlib.Path(os.getenv("TRACE_DIR", "artifacts"))

def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None: 
        return default
    return str(v).lower() in ("1","true","yes","on")

def langsmith_enabled() -> bool:
    # Enable if either switch is on AND a key exists (you have both)
    return (_bool_env("LANGSMITH_ENABLED", False) or _bool_env("LANGCHAIN_TRACING_V2", False)) and bool(os.getenv("LANGCHAIN_API_KEY"))

def get_callbacks() -> List[Any]:
    if not langsmith_enabled():
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ.pop("LANGCHAIN_ENDPOINT", None)
        return []
    try:
        from langchain.callbacks.tracers.langchain import LangChainTracerV2
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        # Respect user’s project/endpoint if provided
        return [LangChainTracerV2()]
    except Exception:
        return []

@contextmanager
def trace(name: str, meta: dict | None = None):
    t0 = time.time()
    try:
        yield
    finally:
        t1 = time.time()
        try:
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            with (TRACE_DIR / "trace.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"span": name, "meta": meta or {}, "start_s": t0, "end_s": t1, "dur_ms": int(1000*(t1-t0))}) + "\n")
        except Exception:
            pass
        print(f"[trace] {name} took {int(1000*(t1-t0))}ms")