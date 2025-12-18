from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Base interface for all agents."""

    @abstractmethod
    async def process(self, task: str) -> object:
        """Process a task and return an agent-specific output."""


