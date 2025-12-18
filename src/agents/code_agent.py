"""CodeAgent: generates Python code snippets with brief explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agents.base import BaseAgent
from src.models.task_input import TaskInput


@dataclass(frozen=True)
class CodeAgentConfig:
    """Configuration for CodeAgent."""

    google_api_key: str
    model: str


class CodeAgent(BaseAgent):
    """CodeAgent: generates Python code in a fenced code block plus a brief explanation."""

    def __init__(self, config: CodeAgentConfig, *, llm: Any | None = None) -> None:
        if not config.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for CodeAgent")

        self._config = config
        temperature = 1.0 if config.model.startswith("gemini-3") else 0.2
        self._llm = llm or ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=config.google_api_key,
            temperature=temperature,
        )

    async def process(self, input_data: TaskInput) -> str:
        task = input_data.task.strip()
        if not task:
            raise ValueError("Task content cannot be empty")

        system_prompt = "\n".join(
            [
                "You are CodeAgent.",
                "Goal: generate clean, correct Python code for the user's request.",
                "",
                "Rules:",
                "- Output MUST be exactly two parts:",
                "  1) A Python fenced code block: ```python ...```",
                "  2) A short explanation (3-8 sentences) immediately after the code block.",
                "- Do not include any other markdown sections.",
                "- Prefer standard library where possible; if you use a third-party library, mention it in the explanation.",
                "- Keep the code runnable and self-contained when possible.",
            ]
        )
        user_prompt = f"Task:\n{task}"

        resp = await self._llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        content = getattr(resp, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM returned empty content")
        return content.strip()


