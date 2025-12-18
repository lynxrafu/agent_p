"""Pydantic models for task persistence and API responses."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.models.agent_content import ContentSource


class TaskStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class TaskResult(BaseModel):
    answer: str | None = None
    sources: list[ContentSource] = Field(default_factory=list)
    model: str | None = None
    error: str | None = None
    stage: Literal["search", "synthesis", "unknown"] | None = None
    debug: dict[str, Any] | None = None


class TaskReadResponse(BaseModel):
    task_id: str
    status: TaskStatus
    result: TaskResult | None = None
    # Optional observability fields (persisted on task doc)
    route: str | None = None
    route_confidence: float | None = None
    route_rationale: str | None = None


