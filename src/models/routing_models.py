"""Pydantic models for routing decisions (PeerAgent)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    content = "content"
    code = "code"
    business_discovery = "business_discovery"
    diagnosis = "diagnosis"
    unknown = "unknown"


class RoutingDecision(BaseModel):
    destination: TaskType = Field(...)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str | None = None


