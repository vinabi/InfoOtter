from __future__ import annotations
from typing import List, Dict, Optional, TypedDict
from pydantic import BaseModel, Field, HttpUrl, ValidationError

class GraphState(TypedDict, total=False):
    query: str
    sources: List[Dict]
    facts: List[Dict]
    brief: Dict
    tools_used: List[str]
    violations: List[str]
    needs_more_context: bool
    tool_error: bool
    schema_ok: bool
    failure_count: int

class Fact(BaseModel):
    fact: str = Field(..., min_length=3)
    evidence_url: Optional[HttpUrl] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

class Source(BaseModel):
    title: str = ""
    url: HttpUrl
    content: str = ""
    published_at: Optional[str] = None

class Brief(BaseModel):
    topic: str
    summary: str
    key_facts: List[Fact] = []
    sources: List[Source] = []
    _markdown: Optional[str] = None

def validate_facts(facts: List[Dict]) -> Optional[str]:
    try:
        [Fact(**f) for f in facts]
        return None
    except ValidationError as e:
        return str(e)

def validate_brief(brief: Dict) -> Optional[str]:
    try:
        Brief(**brief)
        return None
    except ValidationError as e:
        return str(e)
