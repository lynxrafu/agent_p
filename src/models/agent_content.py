from __future__ import annotations

from pydantic import BaseModel, Field


class ContentSource(BaseModel):
    title: str | None = None
    url: str
    score: float | None = None


class ContentAgentOutput(BaseModel):
    answer: str
    sources: list[ContentSource] = Field(default_factory=list)
    model: str


