"""CodeAgent: generates code snippets (multi-language) with brief explanations."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
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
    """CodeAgent: generates code in a fenced code block plus a brief explanation."""

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
        self.last_trace: dict[str, Any] | None = None

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

    def _detect_language(self, task: str) -> str:
        """Best-effort language detection from the task text."""
        t = task.lower()
        # Keep simple + deterministic: prefer explicit user intent.
        if any(k in t for k in ["typescript", " ts ", "tsx", "tsc"]):
            return "ts"
        if any(k in t for k in ["javascript", " js ", "node", "nodejs", "npm"]):
            return "javascript"
        if any(k in t for k in ["bash", "shell", "sh ", "linux", "ubuntu"]):
            return "bash"
        if any(k in t for k in ["powershell", "pwsh"]):
            return "powershell"
        if "sql" in t:
            return "sql"
        if any(k in t for k in ["golang", " go "]):
            return "go"
        if "rust" in t:
            return "rust"
        if any(k in t for k in ["c#", "csharp", ".net"]):
            return "csharp"
        if "java" in t:
            return "java"
        if any(k in t for k in ["c++", "cpp"]):
            return "cpp"
        if any(k in t for k in ["c ", "c-language"]):
            return "c"
        # Default for this project: Python (also best supported in tests).
        return "python"

    def _refusal_with_safe_alternative(self, *, language: str) -> str:
        """Return a safe alternative response in the required two-part format."""
        # Always provide a safe "dry run" alternative; language-specific where reasonable.
        if language == "bash":
            code_lines = [
                "```bash",
                "# SAFE ALTERNATIVE: dry-run listing only (no deletion).",
                "find . -type f | head -n 200",
                "```",
            ]
        elif language == "powershell":
            code_lines = [
                "```powershell",
                "# SAFE ALTERNATIVE: dry-run listing only (no deletion).",
                "Get-ChildItem -Recurse -File | Select-Object -First 200 | ForEach-Object { $_.FullName }",
                "```",
            ]
        elif language == "javascript":
            code_lines = [
                "```javascript",
                "// SAFE ALTERNATIVE: dry-run listing only (no deletion).",
                "import { readdir } from 'node:fs/promises';",
                "import { join } from 'node:path';",
                "",
                "async function walk(dir, out) {",
                "  const entries = await readdir(dir, { withFileTypes: true });",
                "  for (const e of entries) {",
                "    const p = join(dir, e.name);",
                "    if (e.isDirectory()) await walk(p, out);",
                "    else out.push(p);",
                "    if (out.length >= 200) return;",
                "  }",
                "}",
                "",
                "const files = [];",
                "await walk('.', files);",
                "console.log(files.join('\\n'));",
                "```",
            ]
        else:
            code_lines = [
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
        code = "\n".join(code_lines)
        explanation = (
            "I can’t help generate destructive code that deletes data or harms systems. "
            "As a safe alternative, the snippet above performs a dry-run by listing files so you can review what would be affected. "
            "If you need cleanup, narrow the scope (specific directory, file types, and confirmation requirements) and I can help write a safer, scoped approach."
        )
        return f"{code}\n\n{explanation}"

    async def process(self, input_data: TaskInput) -> str:
        task = input_data.task.strip()
        if not task:
            raise ValueError("Task content cannot be empty")

        language = self._detect_language(task)

        if self._looks_destructive(task):
            # Deterministic refusal (no network/LLM call).
            out = self._refusal_with_safe_alternative(language=language)
            self.last_trace = {
                "agent": "code_agent",
                "stage": "refusal",
                "model": self._config.model,
                "prompt": None,
                "raw_output": None,
                "parsed_output": out,
                "latency_ms": 0.0,
            }
            return out

        system_prompt = "\n".join(
            [
                "You are CodeAgent.",
                "Goal: generate clean, correct code for the user's request.",
                "",
                "Rules:",
                "- Output MUST be exactly two parts:",
                "  1) A specific language fenced code block: ```{language} ...```",
                "  2) A short explanation (3-8 sentences) immediately after the code block.",
                "- Do not include any other markdown sections.",
                "- Prefer standard library where possible; if you use a third-party library, mention it in the explanation.",
                "- Keep the code runnable and self-contained when possible. {language} is the language of the code block.",
                "",
                f"Target language: {language}",
                "- The fenced code block language MUST match the target language above.",
            ]
        )
        user_prompt = f"Task:\n{task}"

        start = time.perf_counter()
        resp = await self._llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        latency_ms = (time.perf_counter() - start) * 1000.0
        content = getattr(resp, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM returned empty content")
        self.last_trace = {
            "agent": "code_agent",
            "stage": "generation",
            "model": self._config.model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "raw_output": content,
            "parsed_output": content.strip(),
            "latency_ms": latency_ms,
        }
        return content.strip()


