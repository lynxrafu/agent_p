"""Pydantic input model shared by agents."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TaskInput(BaseModel):
    """Normalized task input passed to agents."""

    task: str = Field(..., description="User task text")


