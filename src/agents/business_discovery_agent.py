"""BusinessDiscoveryAgent: stateful interview that asks clarifying questions (no solutions)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib
from typing import Any, TypedDict

import structlog
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.models.task_input import TaskInput

log = structlog.get_logger(__name__)


class DiscoveryPhase(str):
    """Discovery phase labels."""

    problem_definition = "problem_definition"
    impact_priority = "impact_priority"
    needs_analysis = "needs_analysis"


class DiscoveryTurn(BaseModel):
    """Structured output for a single discovery turn."""

    phase: str = Field(...)
    question: str = Field(..., min_length=1)
    rationale: str | None = None


class DiscoveryState(TypedDict, total=False):
    """LangGraph state stored per session."""

    history: list[str]
    phase: str
    last_question: str
    message: str


@dataclass(frozen=True)
class BusinessDiscoveryAgentConfig:
    """Configuration for BusinessDiscoveryAgent."""

    google_api_key: str
    model: str
    mongo_url: str


class BusinessDiscoveryAgent:
    """Stateful business discovery interview agent."""

    def __init__(self, config: BusinessDiscoveryAgentConfig, *, llm: Any | None = None, checkpointer: Any | None = None) -> None:
        self._config = config

        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "\n".join(
                        [
                            "You are BusinessDiscoveryAgent.",
                            "You interview the user to clarify the real business problem.",
                            "You MUST NOT propose solutions. Ask clarifying questions only.",
                            "",
                            "Phases:",
                            f"- {DiscoveryPhase.problem_definition}: clarify what the problem is and context",
                            f"- {DiscoveryPhase.impact_priority}: quantify impact and urgency",
                            f"- {DiscoveryPhase.needs_analysis}: distinguish requested solution vs underlying need",
                            "",
                            "Rules:",
                            "- Output ONLY a JSON object matching the schema.",
                            "- Ask exactly ONE question per turn.",
                            "- Keep it professional and concise.",
                        ]
                    ),
                ),
                (
                    "human",
                    "\n".join(
                        [
                            "Session state:",
                            "{state}",
                            "",
                            "User message:",
                            "{message}",
                        ]
                    ),
                ),
            ]
        )

        self._llm = llm
        if self._llm is None and config.google_api_key:
            temperature = 1.0 if config.model.startswith("gemini-3") else 0.7
            self._llm = ChatGoogleGenerativeAI(
                model=config.model,
                google_api_key=config.google_api_key,
                temperature=temperature,
            )

        self._chain = None
        if self._llm is not None and hasattr(self._llm, "with_structured_output"):
            self._chain = self._prompt | self._llm.with_structured_output(DiscoveryTurn)

        # Lazy import to keep base install light; required by Story 3.1 requirements.txt
        self._checkpointer = checkpointer

    async def process(self, input_data: TaskInput) -> str:
        """Process a user message and return the next clarifying question."""
        session_id = input_data.session_id or "default"
        message = input_data.task.strip()
        if not message:
            return "What business problem are you facing right now, and which team is most affected?"

        # Build/compile graph and load state via checkpointer.
        state = await self._run_graph(session_id=session_id, message=message)
        return state.get("last_question") or "What business problem are you facing right now, and which team is most affected?"

    async def _run_graph(self, *, session_id: str, message: str) -> DiscoveryState:
        # Dynamic imports keep static analysis resilient across environments.
        langgraph_graph = importlib.import_module("langgraph.graph")
        langgraph_mongo = importlib.import_module("langgraph.checkpoint.mongodb")
        END = getattr(langgraph_graph, "END")
        StateGraph = getattr(langgraph_graph, "StateGraph")
        MongoDBSaver = getattr(langgraph_mongo, "MongoDBSaver")

        def node(state: DiscoveryState) -> DiscoveryState:
            history = state.get("history") or []
            phase = state.get("phase") or DiscoveryPhase.problem_definition
            msg = (state.get("message") or "").strip()

            # If LLM chain unavailable, use deterministic fallbacks.
            if self._chain is None:
                q = self._fallback_question(phase)
                history2 = history + [f"USER: {msg}", f"Q: {q}"]
                next_phase = self._advance_phase(phase)
                return {"history": history2, "phase": next_phase, "last_question": q}

            turn: DiscoveryTurn = self._chain.invoke({"state": {"phase": phase, "history": history}, "message": msg})
            q = turn.question.strip()
            if not q.endswith("?"):
                q = q + "?"
            history2 = history + [f"USER: {msg}", f"Q: {q}"]
            next_phase = turn.phase if turn.phase in (DiscoveryPhase.problem_definition, DiscoveryPhase.impact_priority, DiscoveryPhase.needs_analysis) else phase
            return {"history": history2, "phase": next_phase, "last_question": q}

        # Graph definition: single step per invocation, persisted by checkpointer.
        graph = StateGraph(DiscoveryState)
        graph.add_node("step", node)
        graph.set_entry_point("step")
        graph.add_edge("step", END)

        checkpointer = self._checkpointer or MongoDBSaver(self._config.mongo_url, db_name="agent_p", collection_name="langgraph_checkpoints")
        compiled = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": session_id}}
        # Run in a thread because MongoDBSaver is sync-oriented.
        return await asyncio.to_thread(compiled.invoke, {"message": message}, config)

    def _fallback_question(self, phase: str) -> str:
        if phase == DiscoveryPhase.impact_priority:
            return "What is the estimated monthly cost/impact of this problem, and is it a top-3 priority?"
        if phase == DiscoveryPhase.needs_analysis:
            return "Do you want a solution now, or do you want to understand the root cause first—and what have you tried so far?"
        return "What is the main problem you are facing, which department is most affected, and when did it start?"

    def _advance_phase(self, phase: str) -> str:
        # Simple deterministic phase progression: after each question, move to next phase.
        if phase == DiscoveryPhase.problem_definition:
            return DiscoveryPhase.impact_priority
        if phase == DiscoveryPhase.impact_priority:
            return DiscoveryPhase.needs_analysis
        return DiscoveryPhase.needs_analysis


