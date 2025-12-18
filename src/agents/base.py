"""Base interface for agent implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.task_input import TaskInput


class BaseAgent(ABC):
    """Base interface for all agents."""

    @abstractmethod
    async def process(self, input_data: TaskInput) -> object:
        """Process a task and return an agent-specific output."""


