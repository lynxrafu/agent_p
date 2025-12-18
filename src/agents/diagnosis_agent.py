"""DiagnosisAgent: converts a business discovery conversation into a structured problem tree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.db.mongo import Mongo
from src.models.task_input import TaskInput


ProblemType = Literal["growth", "cost", "ops", "product", "support", "unknown"]


class ProblemTreeNode(BaseModel):
    """A node in a problem tree."""

    label: str = Field(..., min_length=1)
    children: list["ProblemTreeNode"] = Field(default_factory=list)


ProblemTreeNode.model_rebuild()


class ProblemDiagnosisOutput(BaseModel):
    """Structured output from DiagnosisAgent."""

    problem_type: ProblemType = Field(...)
    main_problem: str = Field(..., min_length=1)
    tree: ProblemTreeNode = Field(...)
    key_unknowns: list[str] = Field(default_factory=list)
    next_questions: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class DiagnosisAgentConfig:
    """Configuration for DiagnosisAgent."""

    google_api_key: str
    model: str
    mongo_url: str


class DiagnosisAgent:
    """DiagnosisAgent: outputs a structured 'problem tree' from conversation history."""

    def __init__(self, config: DiagnosisAgentConfig, *, llm: Any | None = None, mongo: Mongo | None = None) -> None:
        self._config = config
        self._mongo = mongo or Mongo(config.mongo_url)

        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "\n".join(
                        [
                            "You are DiagnosisAgent (strategy consultant).",
                            "You must structure the conversation into a problem tree: Main Problem -> Causes -> Sub-causes.",
                            "",
                            "Rules:",
                            "- Output ONLY JSON matching the schema.",
                            "- Do NOT propose solutions; focus on causes, unknowns, and questions.",
                            "- Keep labels concise; avoid long paragraphs.",
                            "",
                            "Problem type options:",
                            "- growth, cost, ops, product, support, unknown",
                        ]
                    ),
                ),
                ("human", "Conversation:\n{conversation}"),
            ]
        )

        self._llm = llm
        if self._llm is None and config.google_api_key:
            temperature = 1.0 if config.model.startswith("gemini-3") else 0.3
            self._llm = ChatGoogleGenerativeAI(
                model=config.model,
                google_api_key=config.google_api_key,
                temperature=temperature,
            )

        self._chain = None
        if self._llm is not None and hasattr(self._llm, "with_structured_output"):
            self._chain = self._prompt | self._llm.with_structured_output(ProblemDiagnosisOutput)

    async def process(self, input_data: TaskInput) -> ProblemDiagnosisOutput:
        """Generate a structured diagnosis output."""
        session_id = input_data.session_id
        conversation = input_data.task.strip()
        if session_id:
            conversation = await self._build_conversation_from_session(session_id=session_id) or conversation

        if not conversation:
            conversation = "User provided no details."

        if self._chain is None:
            return self._fallback(conversation)

        return await self._chain.ainvoke({"conversation": conversation})

    async def _build_conversation_from_session(self, *, session_id: str) -> str:
        docs = await self._mongo.list_tasks_by_session(session_id, limit=50)
        lines: list[str] = []
        for d in docs:
            task = (d.get("task") or "").strip()
            if task:
                lines.append(f"USER: {task}")
            result = d.get("result") or {}
            answer = (result.get("answer") or "").strip() if isinstance(result, dict) else ""
            if answer:
                lines.append(f"AGENT: {answer}")
        return "\n".join(lines).strip()

    def _fallback(self, conversation: str) -> ProblemDiagnosisOutput:
        # Minimal deterministic structuring for testability / no-LLM environments.
        main = "Business performance issue"
        if "sales" in conversation.lower() or "satış" in conversation.lower():
            main = "Sales are down"
        tree = ProblemTreeNode(
            label=main,
            children=[
                ProblemTreeNode(label="Demand (market/customer)"),
                ProblemTreeNode(label="Supply (inventory/ops)"),
                ProblemTreeNode(label="Pricing and positioning"),
                ProblemTreeNode(label="Channel execution"),
            ],
        )
        return ProblemDiagnosisOutput(
            problem_type="unknown",
            main_problem=main,
            tree=tree,
            key_unknowns=["Insufficient details to confirm root causes"],
            next_questions=["Which segment/channel is most impacted, and when did the change start?"],
        )

    @staticmethod
    def render_markdown(output: ProblemDiagnosisOutput) -> str:
        """Render the structured problem tree as a readable markdown string."""

        def walk(node: ProblemTreeNode, depth: int) -> list[str]:
            prefix = "  " * depth + "- "
            lines = [f"{prefix}{node.label}"]
            for c in node.children:
                lines.extend(walk(c, depth + 1))
            return lines

        parts = [
            f"Problem type: **{output.problem_type}**",
            "",
            "**Problem tree**",
            *walk(output.tree, 0),
        ]
        if output.key_unknowns:
            parts.extend(["", "**Key unknowns**", *[f"- {u}" for u in output.key_unknowns]])
        if output.next_questions:
            parts.extend(["", "**Next questions**", *[f"- {q}" for q in output.next_questions]])
        return "\n".join(parts).strip()


