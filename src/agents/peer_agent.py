"""PeerAgent: routes tasks to specialist agent types."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from src.core.settings import Settings
from src.models.routing_models import RoutingDecision, TaskType
from src.models.task_input import TaskInput

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PeerAgentConfig:
    """Configuration for PeerAgent LLM routing."""

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
                            "You are PeerAgent. Your job is to classify the user's task into exactly one destination:",
                            "  - content",
                            "  - code",
                            "  - business_discovery",
                            "  - diagnosis",
                            "  - unknown",
                            "",
                            "Output requirements (STRICT):",
                            "- Output only a JSON object matching the schema fields: destination, confidence, rationale.",
                            "- Do not include markdown. Do not include extra keys.",
                            "- confidence: a float in [0, 1].",
                            "",
                            "Decision rules:",
                            "- Prefer 'unknown' only when the task is truly ambiguous after applying the rules below.",
                            "- If ambiguous, default to 'content'.",
                            "- If the user asks for code snippets, implementation, debugging, or API usage → code.",
                            "- If the user asks to diagnose business performance, root causes, ops issues, complaints → business_discovery.",
                            "- If the user asks to structure findings into a problem tree / issue tree / causes and sub-causes → diagnosis.",
                            "- Otherwise → content.",
                            "",
                            "Examples:",
                            'User: "kimi k2 modeli hakkında beni bilgilendir" -> content',
                            'User: "LangChain metot örneğini bana göster" -> code',
                            'User: "Satışlarımız düşüyor, kök nedeni bulmama yardım et" -> business_discovery',
                            'User: "Konuşmayı problem ağacına dönüştür ve nedenleri grupla" -> diagnosis',
                            'User: "Write a summary of this article" -> content',
                        ]
                    ),
                ),
                ("human", "Task: {task}"),
            ]
        )

        # LLM is optional. If not configured or fails, we fall back to keyword routing.
        self._llm = llm
        if self._llm is None and self._config.google_api_key:
            # Vendor guidance: Gemini 3 models recommend keeping temperature at default 1.0.
            # For non-Gemini-3 models, we prefer low variance for routing/classification.
            temperature = 1.0 if self._config.model.startswith("gemini-3") else 0.0
            self._llm = ChatGoogleGenerativeAI(
                model=self._config.model,
                google_api_key=self._config.google_api_key,
                temperature=temperature,
            )

        self._chain = None
        if self._llm is not None and hasattr(self._llm, "with_structured_output"):
            self._chain = self._prompt | self._llm.with_structured_output(RoutingDecision)

    async def route(self, input_data: TaskInput) -> RoutingDecision:
        """Route a task to an agent destination with fallback behavior."""

        task = input_data.task.strip()
        if not task:
            return RoutingDecision(destination=TaskType.content, confidence=0.0, rationale="empty_task_defaults_to_content")

        # Attempt LLM routing first (if available).
        if self._chain is not None:
            with suppress(ValidationError, ValueError, TypeError, RuntimeError, TimeoutError, OSError):
                decision: RoutingDecision = await self._chain.ainvoke({"task": task})
                # Enforce default-to-content on unknown/ambiguous.
                if decision.destination == TaskType.unknown:
                    return RoutingDecision(destination=TaskType.content, confidence=decision.confidence, rationale=decision.rationale)
                return decision

            # Fall back to deterministic routing.
            log.warning("peer_agent_llm_routing_failed_fallback_to_keyword", task=task)

        return self._keyword_route(task)

    def _keyword_route(self, task: str) -> RoutingDecision:
        t = task.lower()

        # Code intent keywords (Turkish + English; keep minimal and deterministic).
        if any(k in t for k in ["code", "kod", "örnek", "example", "snippet", "yaz", "implement", "langchain"]):
            return RoutingDecision(destination=TaskType.code, confidence=0.55, rationale="keyword_match_code")

        # Diagnosis intent keywords.
        if any(k in t for k in ["problem tree", "issue tree", "problem ağacı", "ağaç", "root causes", "kök nedenler", "causes", "sub-causes", "diagnosis", "diagnose", "structure"]):
            return RoutingDecision(destination=TaskType.diagnosis, confidence=0.55, rationale="keyword_match_diagnosis")

        # Business discovery intent keywords.
        if any(k in t for k in ["sales", "satış", "kök", "root cause", "problem", "şikayet", "complaint", "operasyon", "depo"]):
            return RoutingDecision(destination=TaskType.business_discovery, confidence=0.55, rationale="keyword_match_business")

        # Default: content.
        return RoutingDecision(destination=TaskType.content, confidence=0.4, rationale="default_content")


