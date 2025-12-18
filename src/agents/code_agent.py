"""CodeAgent: generates Python code snippets with brief explanations."""

from __future__ import annotations

from dataclasses import dataclass
import re
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

    def _looks_destructive(self, task: str) -> bool:
        """Heuristic guardrail to block obviously destructive requests."""
        t = task.lower()
        patterns = [
            r"\bdelete\s+all\s+files\b",
            r"\brm\s+-rf\b",
            r"\bformat\s+c:\b",
            r"\bdel\s+/s\b",
            r"\bshutil\.rmtree\b",
            r"\bos\.remove\b",
            r"\bos\.unlink\b",
            r"\bfilesystem\s+wiper\b",
            r"\bransomware\b",
            r"\bkeylogger\b",
            r"\bddos\b",
            r"\bcredential\s+steal\b",
            r"\bsteal\s+passwords?\b",
        ]
        return any(re.search(p, t) for p in patterns)

    def _refusal_with_safe_alternative(self) -> str:
        """Return a safe alternative response in the required two-part format."""
        code = "\n".join(
            [
                "```python",
                "from pathlib import Path",
                "",
                "# SAFE ALTERNATIVE: dry-run listing only (no deletion).",
                "root = Path('.')",
                "paths = sorted(p for p in root.rglob('*') if p.is_file())",
                "print(f\"Found {len(paths)} files under {root.resolve()}\")",
                "for p in paths[:200]:",
                "    print(p)",
                "if len(paths) > 200:",
                "    print(f\"... and {len(paths) - 200} more\")",
                "```",
            ]
        )
        explanation = (
            "I can’t help generate destructive code that deletes data or harms systems. "
            "As a safe alternative, the script above performs a dry-run by listing files so you can review what would be affected. "
            "If you need cleanup, narrow the scope (specific directory, file types, and confirmation requirements) and I can help write a safer, scoped approach."
        )
        return f"{code}\n\n{explanation}"

    async def process(self, input_data: TaskInput) -> str:
        task = input_data.task.strip()
        if not task:
            raise ValueError("Task content cannot be empty")

        if self._looks_destructive(task):
            # Deterministic refusal (no network/LLM call).
            return self._refusal_with_safe_alternative()

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
                "",
                "Safety rules:",
                "- Refuse requests that enable malware, credential theft, data destruction, or abuse.",
                "- When refusing, provide a safe alternative when possible (e.g., dry-run, validation, or sandboxed example).",
            ]
        )
        user_prompt = f"Task:\n{task}"

        resp = await self._llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        content = getattr(resp, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM returned empty content")
        return content.strip()


