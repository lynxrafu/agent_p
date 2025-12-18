"""CodeAgent: Research-first code generation with web search and conversation memory.

Implements requirements from research_about_code_Agent.md:
- Web Search Tool integration (Tavily) for latest documentation
- Conversation history/memory for follow-up requests
- Research-first approach with Chain-of-Thought reasoning
- Complete, executable code with no placeholders
- Reference links to documentation sources
- Installation instructions for third-party libraries
- Error handling (try-except) in generated code

Architecture follows the "World-Class Senior Software Engineer" persona.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any

import httpx
import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient
from tavily.errors import InvalidAPIKeyError as TavilyInvalidAPIKeyError

from src.agents.base import BaseAgent


def _extract_llm_content(content: Any) -> str:
    """Extract text content from LLM response, handling Gemini 3's new format.
    
    Gemini 3 models return content as a list of dicts:
    [{'type': 'text', 'text': 'Hello', 'extras': {...}}]
    
    Older models return a simple string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Gemini 3 format: list of content blocks
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts)
    return ""
from src.models.task_input import TaskInput

log = structlog.get_logger(__name__)


# =============================================================================
# Pydantic Output Models
# =============================================================================

class CodeReference(BaseModel):
    """A documentation reference used in code generation."""
    title: str = Field(..., description="Title of the documentation source")
    url: str = Field(..., description="URL to the documentation")


class CodeAgentOutput(BaseModel):
    """Structured output from CodeAgent."""
    code: str = Field(..., description="The generated code block")
    explanation: str = Field(..., description="Brief explanation of the code")
    language: str = Field(..., description="Programming language used")
    installation: str | None = Field(None, description="Installation instructions if third-party libraries used")
    references: list[CodeReference] = Field(default_factory=list, description="Documentation references")


# =============================================================================
# Mongo Protocol (for dependency injection)
# =============================================================================

class MongoProtocol:
    """Protocol for Mongo dependency (avoids circular imports)."""

    async def list_tasks_by_session(
        self, session_id: str, *, limit: int = 50
    ) -> list[dict]:
        """List tasks for a session."""
        raise NotImplementedError  # Protocol stub


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class CodeAgentConfig:
    """Configuration for CodeAgent."""
    google_api_key: str
    model: str
    tavily_api_key: str | None = None
    max_search_results: int = 3
    search_depth: str = "advanced"


# =============================================================================
# Conversation History Manager
# =============================================================================

class ConversationHistory:
    """Simple in-memory conversation history for session continuity.

    Implements requirement: "The agent must maintain conversation history.
    If the user says 'Fix the error in this code,' the agent must recall
    the code generated in the previous turn."
    """

    def __init__(self, max_turns: int = 10) -> None:
        self._history: dict[str, list[dict[str, str]]] = {}
        self._max_turns = max_turns

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Add a conversation turn."""
        if session_id not in self._history:
            self._history[session_id] = []

        self._history[session_id].append({"role": role, "content": content})

        # Keep only last N turns to prevent context overflow
        if len(self._history[session_id]) > self._max_turns * 2:
            self._history[session_id] = self._history[session_id][-self._max_turns * 2:]

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Get conversation history for a session."""
        return self._history.get(session_id, [])

    def format_for_prompt(self, session_id: str) -> str:
        """Format history as prompt context."""
        history = self.get_history(session_id)
        if not history:
            return ""

        lines = ["## Previous Conversation:"]
        for turn in history[-6:]:  # Last 3 exchanges
            role = "User" if turn["role"] == "user" else "Assistant"
            # Truncate long content
            content = turn["content"][:500] + "..." if len(turn["content"]) > 500 else turn["content"]
            lines.append(f"**{role}:** {content}")

        return "\n".join(lines)


# Global conversation history (in production, use Redis/MongoDB)
_conversation_history = ConversationHistory()


# =============================================================================
# System Prompt with Few-Shot Examples (from research_about_code_Agent.md)
# =============================================================================

CODE_AGENT_SYSTEM_PROMPT = """## Role
You are a **World-Class Senior Software Engineer**. You are an expert in Python, JavaScript, TypeScript, Go, Rust, and System Architecture.

## Task
Generate clean, correct, production-ready code for the user's request.

## Constraints & Rules (MUST FOLLOW)

1. **Research First:** If the user asks about a specific library or framework, use the provided web search context to ensure you're using the latest API and syntax. Do NOT rely solely on training data for rapidly-evolving libraries.

2. **Complete Code Only:** NEVER leave placeholders like `# code goes here`, `// TODO`, or `pass`. Generate FULL, FUNCTIONAL, EXECUTABLE code.

3. **Error Handling:** Include appropriate error handling (try-except/try-catch blocks) in your code. Handle edge cases gracefully.

4. **Comments:** Add brief explanatory comments to complex logic. Don't over-comment obvious code.

5. **Installation Instructions:** If you use third-party libraries, specify exactly how to install them (e.g., `pip install requests`).

6. **References:** At the end of your response, cite the documentation sources you used from the web search context.

7. **Format:** Structure your response as:
   - Code block with correct language tag
   - Brief explanation (3-8 sentences)
   - Installation instructions (if applicable)
   - Source references

## Target Language: {language}

{history_context}

{search_context}

## Few-Shot Example

**User:** "Write a function to make an HTTP GET request"

**Assistant:**
```python
import requests
from typing import Any

def fetch_data(url: str, timeout: int = 30) -> dict[str, Any] | None:
    \"\"\"Fetch JSON data from a URL with error handling.

    Args:
        url: The URL to fetch data from
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response or None if request fails
    \"\"\"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Raise exception for 4xx/5xx status codes
        return response.json()
    except requests.exceptions.Timeout:
        print(f"Request timed out after {{timeout}} seconds")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {{e}}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {{e}}")
        return None

# Usage example
if __name__ == "__main__":
    data = fetch_data("https://api.example.com/data")
    if data:
        print(data)
```

This function makes a GET request to the specified URL and returns the JSON response. It includes comprehensive error handling for timeouts, HTTP errors, and general request failures. The function uses type hints for better code clarity and IDE support.

**Installation:** `pip install requests`

**Source:** Python Requests Library Documentation - https://docs.python-requests.org/

---

Now generate code for the user's request following the same format and quality standards."""


# =============================================================================
# CodeAgent Class
# =============================================================================

class CodeAgent(BaseAgent):
    """CodeAgent: Research-first code generation with web search and memory.

    Implements all requirements from research_about_code_Agent.md:
    - Web search for latest documentation (Tavily)
    - Conversation history for follow-up requests (Mongo-backed for persistence)
    - Chain-of-Thought reasoning
    - Complete, executable code
    - Reference links and installation instructions
    """

    def __init__(
        self,
        config: CodeAgentConfig,
        *,
        llm: Any | None = None,
        tavily_client: AsyncTavilyClient | None = None,
        mongo: MongoProtocol | None = None,
    ) -> None:
        if not config.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for CodeAgent")

        self._config = config
        self._mongo = mongo

        # Initialize LLM (Gemini)
        temperature = 1.0 if config.model.startswith("gemini-3") else 0.2
        self._llm = llm or ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=config.google_api_key,
            temperature=temperature,
        )

        # Initialize Tavily for web search (optional but recommended)
        self._tavily: AsyncTavilyClient | None = None
        if config.tavily_api_key:
            self._tavily = tavily_client or AsyncTavilyClient(api_key=config.tavily_api_key)

        self.last_trace: dict[str, Any] | None = None

    # -------------------------------------------------------------------------
    # Session History (Mongo-backed for persistence across workers)
    # -------------------------------------------------------------------------
    async def _load_session_history_from_mongo(
        self, session_id: str
    ) -> list[dict[str, str]]:
        """Load conversation history from Mongo for session continuity.
        
        Returns list of {"role": "user"|"assistant", "content": str} dicts.
        """
        if not self._mongo or not session_id:
            return []
        
        try:
            docs = await self._mongo.list_tasks_by_session(session_id, limit=20)
        except (ConnectionError, TimeoutError, OSError, RuntimeError):
            log.warning("code_agent_mongo_history_load_failed", session_id=session_id)
            return []
        
        history: list[dict[str, str]] = []
        for doc in docs:
            task_text = (doc.get("task") or "").strip()
            if task_text:
                history.append({"role": "user", "content": task_text})
            
            result = doc.get("result")
            if isinstance(result, dict):
                answer = (result.get("answer") or "").strip()
                if answer:
                    # Truncate long code responses for context window efficiency
                    if len(answer) > 1500:
                        answer = answer[:1500] + "\n... [truncated]"
                    history.append({"role": "assistant", "content": answer})
        
        return history
    
    def _format_history_for_prompt(self, history: list[dict[str, str]]) -> str:
        """Format session history as prompt context."""
        if not history:
            return ""
        
        lines = ["## Previous Conversation:"]
        # Use last 6 messages (3 exchanges) for context
        for turn in history[-6:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            content = turn["content"]
            # Truncate long content
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"**{role}:** {content}")
        
        return "\n".join(lines)

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
        if any(k in t for k in ["typescript", " ts ", "tsx", "tsc"]):
            return "typescript"
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
        if "java" in t and "javascript" not in t:
            return "java"
        if any(k in t for k in ["c++", "cpp"]):
            return "cpp"
        if any(k in t for k in [" c ", "c-language"]):
            return "c"
        return "python"

    def _refusal_with_safe_alternative(self, *, language: str) -> str:
        """Return a safe alternative response for destructive requests."""
        if language == "bash":
            code = """```bash
# SAFE ALTERNATIVE: dry-run listing only (no deletion).
find . -type f | head -n 200
```"""
        elif language == "powershell":
            code = """```powershell
# SAFE ALTERNATIVE: dry-run listing only (no deletion).
Get-ChildItem -Recurse -File | Select-Object -First 200 | ForEach-Object { $_.FullName }
```"""
        elif language == "javascript":
            code = """```javascript
// SAFE ALTERNATIVE: dry-run listing only (no deletion).
import { readdir } from 'node:fs/promises';
import { join } from 'node:path';

async function walk(dir, out = []) {
  const entries = await readdir(dir, { withFileTypes: true });
  for (const e of entries) {
    const p = join(dir, e.name);
    if (e.isDirectory()) await walk(p, out);
    else out.push(p);
    if (out.length >= 200) return out;
  }
  return out;
}

const files = await walk('.');
console.log(files.join('\\n'));
```"""
        else:
            code = """```python
from pathlib import Path

# SAFE ALTERNATIVE: dry-run listing only (no deletion).
root = Path('.')
paths = sorted(p for p in root.rglob('*') if p.is_file())
print(f"Found {len(paths)} files under {root.resolve()}")
for p in paths[:200]:
    print(p)
if len(paths) > 200:
    print(f"... and {len(paths) - 200} more")
```"""

        explanation = (
            "I can't help generate destructive code that deletes data or harms systems. "
            "As a safe alternative, the snippet above performs a dry-run by listing files "
            "so you can review what would be affected. If you need cleanup, narrow the scope "
            "(specific directory, file types, and confirmation requirements) and I can help "
            "write a safer, scoped approach."
        )
        return f"{code}\n\n{explanation}"

    def _build_search_query(self, task: str, language: str) -> str:
        """Build an optimized search query for documentation lookup."""
        # Extract library/framework mentions for targeted search
        libraries = []
        task_lower = task.lower()

        # Common libraries/frameworks to detect
        lib_patterns = [
            "langchain", "fastapi", "flask", "django", "requests", "pandas",
            "numpy", "pytorch", "tensorflow", "react", "vue", "angular",
            "express", "nestjs", "spring", "axios", "fetch", "sqlalchemy",
            "pydantic", "asyncio", "aiohttp", "httpx", "selenium", "playwright",
        ]

        for lib in lib_patterns:
            if lib in task_lower:
                libraries.append(lib)

        if libraries:
            return f"{' '.join(libraries)} {language} documentation example 2025"

        # Generic query based on task
        keywords = task.split()[:5]  # First 5 words
        return f"{language} {' '.join(keywords)} documentation example"

    async def _search_documentation(self, task: str, language: str) -> tuple[str, list[CodeReference]]:
        """Search for relevant documentation using Tavily.

        Returns:
            Tuple of (context_string, list_of_references)
        """
        if not self._tavily:
            return "", []

        query = self._build_search_query(task, language)

        try:
            response = await self._tavily.search(
                query=query,
                max_results=self._config.max_search_results,
                search_depth=self._config.search_depth,
                include_answer=False,
                include_images=False,
            )
        except (httpx.HTTPError, TimeoutError, OSError, ValueError, RuntimeError, TavilyInvalidAPIKeyError) as e:
            log.warning("code_agent_search_failed", error=str(e), query=query)
            return "", []

        results = (response or {}).get("results") or []
        if not results:
            return "", []

        context_parts: list[str] = []
        references: list[CodeReference] = []

        for r in results:
            url = r.get("url")
            if not url:
                continue

            title = r.get("title") or "Documentation"
            snippet = r.get("content") or r.get("raw_content") or ""
            if len(snippet) > 800:
                snippet = snippet[:800] + "..."

            references.append(CodeReference(title=title, url=url))
            context_parts.append(f"**{title}**\nURL: {url}\n{snippet}")

        context = "\n\n---\n\n".join(context_parts) if context_parts else ""
        return context, references

    async def process(self, input_data: TaskInput) -> str:
        """Process a code generation request with research-first approach."""
        task = input_data.task.strip()
        if not task:
            raise ValueError("Task content cannot be empty")

        session_id = input_data.session_id or "default"
        language = self._detect_language(task)

        # Safety check for destructive requests
        if self._looks_destructive(task):
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

        # Load session history - prefer Mongo (persistent), fall back to in-memory
        if self._mongo and session_id != "default":
            history = await self._load_session_history_from_mongo(session_id)
            history_context = self._format_history_for_prompt(history)
        else:
            # Fallback to in-memory for tests or when Mongo unavailable
            _conversation_history.add_turn(session_id, "user", task)
            history_context = _conversation_history.format_for_prompt(session_id)

        # Research-first: Search for relevant documentation
        search_context, references = await self._search_documentation(task, language)

        # Format search context
        search_context_section = ""
        if search_context:
            search_context_section = f"""## Web Search Results (Latest Documentation)
Use this context to ensure you're using the latest API syntax:

{search_context}"""

        # Build system prompt with all context
        system_prompt = CODE_AGENT_SYSTEM_PROMPT.format(
            language=language,
            history_context=history_context if history_context else "## No previous conversation",
            search_context=search_context_section if search_context_section else "## No web search results available",
        )

        user_prompt = f"**User Request:** {task}"

        # Generate code
        start = time.perf_counter()
        try:
            resp = await self._llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
        except Exception as e:
            log.error("code_agent_llm_failed", error=str(e))
            raise ValueError(f"LLM generation failed: {e}") from e

        latency_ms = (time.perf_counter() - start) * 1000.0

        raw_content = getattr(resp, "content", None)
        content = _extract_llm_content(raw_content)
        if not content.strip():
            raise ValueError("LLM returned empty content")

        response = content.strip()

        # Append references if we have them and they're not already in response
        if references and "Source:" not in response and "Reference:" not in response:
            ref_section = "\n\n**Sources:**\n" + "\n".join(
                f"- [{ref.title}]({ref.url})" for ref in references[:3]
            )
            response += ref_section

        # Add assistant response to in-memory history (only when not using Mongo)
        if not self._mongo or session_id == "default":
            _conversation_history.add_turn(session_id, "assistant", response)

        self.last_trace = {
            "agent": "code_agent",
            "stage": "generation",
            "model": self._config.model,
            "prompt": f"{system_prompt[:500]}...\n\n{user_prompt}",
            "raw_output": content,
            "parsed_output": response,
            "latency_ms": latency_ms,
            "search_query": self._build_search_query(task, language) if self._tavily else None,
            "references": [ref.model_dump() for ref in references],
        }

        return response
