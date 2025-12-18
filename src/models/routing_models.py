from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    content = "content"
    code = "code"
    business_discovery = "business_discovery"
    unknown = "unknown"


class RoutingDecision(BaseModel):
    destination: TaskType = Field(...)
    confidence: float | None = None
    rationale: str | None = None


