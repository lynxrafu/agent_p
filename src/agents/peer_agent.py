from __future__ import annotations

# ruff: noqa: BLE001
# pylint: disable=broad-exception-caught

from dataclasses import dataclass
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.settings import Settings
from src.models.routing_models import RoutingDecision, TaskType


@dataclass(frozen=True)
class PeerAgentConfig:
    google_api_key: str | None
    model: str


class PeerAgent:
    """PeerAgent: routes tasks to specialist agents."""

    def __init__(self, settings: Settings, *, llm: Any | None = None) -> None:
        self._settings = settings
        self._config = PeerAgentConfig(google_api_key=settings.GOOGLE_API_KEY, model=settings.GEMINI_MODEL)

        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "\n".join(
                        [
                            "You are a PeerAgent that classifies the user's task into one of:",
                            "- content",
                            "- code",
                            "- business_discovery",
                            "- unknown",
                            "",
                            "Rules:",
                            "- Prefer 'unknown' only when truly ambiguous.",
                            "- If ambiguous, default to 'content'.",
                            "- Do not output anything except the JSON fields required by the schema.",
                            "",
                            "Examples:",
                            'User: "kimi k2 modeli hakkında beni bilgilendir" -> content',
                            'User: "LangChain metot örneğini bana göster" -> code',
                            'User: "Satışlarımız düşüyor, kök nedeni bulmama yardım et" -> business_discovery',
                        ]
                    ),
                ),
                ("human", "Task: {task}"),
            ]
        )

        # LLM is optional. If not configured or fails, we fall back to keyword routing.
        self._llm = llm
        if self._llm is None and self._config.google_api_key:
            self._llm = ChatGoogleGenerativeAI(
                model=self._config.model,
                google_api_key=self._config.google_api_key,
                temperature=0,
            )

        self._chain = None
        if self._llm is not None and hasattr(self._llm, "with_structured_output"):
            self._chain = self._prompt | self._llm.with_structured_output(RoutingDecision)

    async def route(self, task: str) -> RoutingDecision:
        """Route a task to an agent destination with fallback behavior."""

        task = task.strip()
        if not task:
            return RoutingDecision(destination=TaskType.content, confidence=0.0, rationale="empty_task_defaults_to_content")

        # Attempt LLM routing first (if available).
        if self._chain is not None:
            try:
                decision: RoutingDecision = await self._chain.ainvoke({"task": task})
                # Enforce default-to-content on unknown/ambiguous.
                if decision.destination == TaskType.unknown:
                    return RoutingDecision(destination=TaskType.content, confidence=decision.confidence, rationale=decision.rationale)
                return decision
            except Exception as _e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                # Fall back to deterministic routing.
                pass

        return self._keyword_route(task)

    def _keyword_route(self, task: str) -> RoutingDecision:
        t = task.lower()

        # Code intent keywords (Turkish + English; keep minimal and deterministic).
        if any(k in t for k in ["code", "kod", "örnek", "example", "snippet", "write a", "yaz", "implement", "langchain"]):
            return RoutingDecision(destination=TaskType.code, confidence=0.55, rationale="keyword_match_code")

        # Business discovery intent keywords.
        if any(k in t for k in ["sales", "satış", "kök", "root cause", "problem", "şikayet", "complaint", "operasyon", "depo"]):
            return RoutingDecision(destination=TaskType.business_discovery, confidence=0.55, rationale="keyword_match_business")

        # Default: content.
        return RoutingDecision(destination=TaskType.content, confidence=0.4, rationale="default_content")


