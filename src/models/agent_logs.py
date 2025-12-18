"""Pydantic models for deep observability agent logs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


AgentName = Literal[
    "peer_agent",
    "content_agent",
    "code_agent",
    "business_discovery_agent",
    "diagnosis_agent",
]


class AgentLogEntry(BaseModel):
    """A single agent interaction log persisted to MongoDB."""

    task_id: str = Field(..., min_length=1)
    session_id: str | None = None
    agent: AgentName = Field(...)
    stage: str = Field(default="main")
    model: str | None = None

    # Observability payload
    prompt: str | None = None
    raw_output: str | None = None
    parsed_output: Any | None = None

    latency_ms: float | None = Field(default=None, ge=0.0)

    created_at: datetime | None = None


