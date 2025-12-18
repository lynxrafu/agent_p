"""BusinessDiscoveryAgent: stateful interview that asks clarifying questions (no solutions).

Implements the Business Analyst methodology from the specification:
- "5 Whys Analysis" & "Business Impact Analysis"
- Iterative questioning until sufficient information gathered
- Outputs structured DiscoveryAnalysis with 4 required fields

Stages:
1. Problem Definition (4 questions)
2. Impact & Priority Test (3 questions)  
3. Needs vs Wants Analysis (3 questions)
4. Final Summary Generation
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib
import time
from typing import Any, Literal, TypedDict

import structlog
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from contextlib import suppress
from pymongo import MongoClient

from src.models.task_input import TaskInput

log = structlog.get_logger(__name__)


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


# =============================================================================
# Pydantic Schemas (from Research_about_prompting.md §3.1)
# =============================================================================

class DiscoveryAnalysis(BaseModel):
    """Final structured output from Business Discovery Agent."""
    
    customer_stated_problem: str = Field(
        ..., 
        description="The problem as initially stated by the customer"
    )
    identified_business_problem: str = Field(
        ..., 
        description="The actual business problem identified after analysis"
    )
    hidden_root_risk: str = Field(
        ..., 
        description="A hidden risk that is not visible yet but could be critical"
    )
    business_urgency: Literal["Low", "Medium", "Critical"] = Field(
        ..., 
        description="The urgency level of the problem"
    )


class DiscoveryTurn(BaseModel):
    """Structured output for a single discovery turn."""
    
    question: str = Field(..., min_length=1, description="The next question to ask")
    reasoning: str | None = Field(None, description="Why this question is important")
    information_gathered: list[str] = Field(
        default_factory=list, 
        description="Key facts extracted from user's response"
    )
    ready_for_summary: bool = Field(
        False, 
        description="True if enough information gathered to generate final analysis"
    )


# =============================================================================
# Phase Configuration
# =============================================================================

class DiscoveryPhase:
    """Discovery phase labels with question counts."""
    
    problem_definition = "problem_definition"  # 4 questions
    impact_priority = "impact_priority"        # 3 questions
    needs_analysis = "needs_analysis"          # 3 questions
    summary = "summary"                        # Final output


# Questions per phase (from claude.md + Research_about_prompting.md)
PHASE_QUESTIONS = {
    DiscoveryPhase.problem_definition: [
        "What is the main problem disrupting your business operations right now?",
        "Which department is most directly affected by this problem?",
        "When did this problem first emerge? How long has it existed?",
        "How are you currently managing or working around this problem?",
    ],
    DiscoveryPhase.impact_priority: [
        "What is the estimated monthly or yearly cost/impact of this problem?",
        "If this problem is not solved, what will happen in 12 months?",
        "Is this problem currently in your company's TOP 3 priorities?",
    ],
    DiscoveryPhase.needs_analysis: [
        "Do you want a solution now, or do you first want to understand the root cause?",
        "What solutions have you tried so far? Why did they fail?",
        "Do you actually need a solution, or is visibility/understanding what's missing?",
    ],
}

PHASE_ORDER = [
    DiscoveryPhase.problem_definition,
    DiscoveryPhase.impact_priority,
    DiscoveryPhase.needs_analysis,
    DiscoveryPhase.summary,
]


# =============================================================================
# State Definition
# =============================================================================

class DiscoveryState(TypedDict, total=False):
    """LangGraph state stored per session."""
    
    history: list[str]              # Conversation transcript
    phase: str                      # Current phase
    phase_question_index: int       # Question index within phase
    last_question: str              # Last question asked
    message: str                    # Current user message
    is_complete: bool               # Interview complete?
    analysis_data: dict             # Final DiscoveryAnalysis output
    information_gathered: list[str] # Facts extracted during interview


# =============================================================================
# Agent Configuration
# =============================================================================

@dataclass(frozen=True)
class BusinessDiscoveryAgentConfig:
    """Configuration for BusinessDiscoveryAgent."""
    
    google_api_key: str
    model: str
    mongo_url: str


# =============================================================================
# System Prompts (from Research_about_prompting.md §1.2)
# =============================================================================

DISCOVERY_SYSTEM_PROMPT = """ROLE: You are an experienced Business Analyst. Your task is to deeply analyze the problem the user is facing and uncover the root cause.

BEHAVIOR GUIDELINES:
1. **Investigation Only:** Never propose a solution immediately. Your job is to diagnose.
2. **Iterative Questioning:** Proceed in a "Question-Answer" format. Ask only 1 focused question at a time.
3. **Technique:** Apply the "5 Whys" technique based on the user's responses.
4. **Metric Extraction:** Attempt to uncover:
   - Duration: How long has this problem existed?
   - Cost: What is the financial impact (Monthly/Yearly loss)?
   - Urgency: Is this currently within the company's TOP 3 priorities?
   - History: What solutions have been tried so far, and why did they fail?

CURRENT PHASE: {phase}
BASE QUESTION TO ADAPT: {base_question}

CONVERSATION HISTORY:
{history}

USER'S LATEST MESSAGE:
{message}

OUTPUT FORMAT:
Return ONLY a JSON object with these fields:
- question: Your next question (adapt the base question to flow naturally)
- reasoning: Why you're asking this (1 sentence)
- information_gathered: Array of key facts from user's response
- ready_for_summary: true if you have enough info about Duration, Cost, Urgency, and History

EXAMPLES:
User: "Satışlarımız düşüyor, nedenini bilmiyorum."
Output: {{"question": "Bu düşüş ne zamandır devam ediyor ve aylık cironuza tahmini etkisi nedir?", "reasoning": "Need to establish duration and financial impact", "information_gathered": ["Sales are declining", "Root cause unknown"], "ready_for_summary": false}}

User: "6 aydır var, ayda 50.000$ kaybediyoruz. Reklamları değiştirdik ama işe yaramadı."
Output: {{"question": "Reklam stratejisini değiştirdiğinizde beklediğiniz sonucu alamamanızın sebebi sizce neydi?", "reasoning": "Understanding why previous solution failed", "information_gathered": ["Problem duration: 6 months", "Monthly loss: $50,000", "Tried: Changed ads - failed"], "ready_for_summary": false}}"""


SUMMARY_SYSTEM_PROMPT = """ROLE: You are an experienced Business Analyst completing a discovery interview.

Based on the complete conversation below, generate a final structured analysis.

CONVERSATION TRANSCRIPT:
{history}

INFORMATION GATHERED:
{facts}

OUTPUT FORMAT:
Return ONLY a JSON object with these exact fields:
- customer_stated_problem: What the customer originally described (their words)
- identified_business_problem: The actual underlying business issue you identified through analysis
- hidden_root_risk: A deeper risk or root cause that may not be obvious to the customer
- business_urgency: One of "Low", "Medium", or "Critical" based on impact and timeline

Be specific and actionable. Avoid generic statements. Reference specific facts from the conversation."""


# =============================================================================
# BusinessDiscoveryAgent
# =============================================================================

class BusinessDiscoveryAgent:
    """Stateful business discovery interview agent.
    
    Conducts a structured interview using the "5 Whys" methodology,
    then generates a DiscoveryAnalysis with 4 required output fields.
    """

    def __init__(
        self,
        config: BusinessDiscoveryAgentConfig,
        *,
        llm: Any | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        
        if self._llm is None and config.google_api_key:
            temperature = 0.7
            self._llm = ChatGoogleGenerativeAI(
                model=config.model,
                google_api_key=config.google_api_key,
                temperature=temperature,
            )

        self.last_trace: dict[str, Any] | None = None
        self._checkpointer = checkpointer

    async def process(self, input_data: TaskInput) -> str:
        """Process a user message and return the next question or final analysis."""
        session_id = input_data.session_id or "default"
        message = input_data.task.strip()

        if not message:
            # First question of the interview
            return PHASE_QUESTIONS[DiscoveryPhase.problem_definition][0]

        # Run the state graph
        state = await self._run_graph(session_id=session_id, message=message)

        # If complete, format and return the analysis
        if state.get("is_complete") and state.get("analysis_data"):
            return self._format_analysis(state["analysis_data"])

        return state.get("last_question") or PHASE_QUESTIONS[DiscoveryPhase.problem_definition][0]

    async def _run_graph(self, *, session_id: str, message: str) -> DiscoveryState:
        """Execute one step of the discovery graph."""
        langgraph_graph = importlib.import_module("langgraph.graph")
        langgraph_mongo = importlib.import_module("langgraph.checkpoint.mongodb")
        END = getattr(langgraph_graph, "END")
        StateGraph = getattr(langgraph_graph, "StateGraph")
        MongoDBSaver = getattr(langgraph_mongo, "MongoDBSaver")

        def node(state: DiscoveryState) -> DiscoveryState:
            history = list(state.get("history") or [])
            phase = state.get("phase") or DiscoveryPhase.problem_definition
            question_idx = state.get("phase_question_index") or 0
            msg = (state.get("message") or "").strip()
            facts = list(state.get("information_gathered") or [])

            # Add user message to history
            if msg:
                history.append(f"USER: {msg}")

            # Check if we're in summary phase
            if phase == DiscoveryPhase.summary:
                analysis = self._generate_analysis(history, facts)
                return {
                    "history": history,
                    "phase": phase,
                    "phase_question_index": 0,
                    "is_complete": True,
                    "analysis_data": analysis,
                    "information_gathered": facts,
                    "last_question": "",
                }

            # Get questions for current phase
            phase_questions = PHASE_QUESTIONS.get(phase, [])

            # Check if we've asked all questions in this phase
            if question_idx >= len(phase_questions):
                # Move to next phase
                current_phase_idx = PHASE_ORDER.index(phase)
                if current_phase_idx + 1 < len(PHASE_ORDER):
                    next_phase = PHASE_ORDER[current_phase_idx + 1]
                    
                    # If next phase is summary, generate analysis
                    if next_phase == DiscoveryPhase.summary:
                        analysis = self._generate_analysis(history, facts)
                        return {
                            "history": history,
                            "phase": next_phase,
                            "phase_question_index": 0,
                            "is_complete": True,
                            "analysis_data": analysis,
                            "information_gathered": facts,
                            "last_question": "",
                        }
                    
                    # Otherwise, get first question of next phase
                    question_idx = 0
                    phase = next_phase
                    phase_questions = PHASE_QUESTIONS.get(phase, [])

            # Get the base question for this turn
            base_question = phase_questions[question_idx] if question_idx < len(phase_questions) else phase_questions[-1]

            # Generate contextual question using LLM
            question, new_facts, ready = self._generate_question(
                phase, base_question, history, msg, facts
            )
            
            # Merge new facts
            facts = facts + new_facts

            # Check if LLM says we're ready for summary
            if ready and len(facts) >= 4:  # Minimum facts threshold
                analysis = self._generate_analysis(history, facts)
                return {
                    "history": history + [f"AGENT: {question}"],
                    "phase": DiscoveryPhase.summary,
                    "phase_question_index": 0,
                    "is_complete": True,
                    "analysis_data": analysis,
                    "information_gathered": facts,
                    "last_question": "",
                }

            # Add question to history
            history.append(f"AGENT: {question}")

            return {
                "history": history,
                "phase": phase,
                "phase_question_index": question_idx + 1,
                "last_question": question,
                "is_complete": False,
                "information_gathered": facts,
            }

        # Build graph
        graph = StateGraph(DiscoveryState)
        graph.add_node("step", node)
        graph.set_entry_point("step")
        graph.add_edge("step", END)

        if self._checkpointer:
            checkpointer = self._checkpointer
        else:
            # MongoDBSaver requires a MongoClient object, not a connection string
            mongo_client = MongoClient(self._config.mongo_url)
            checkpointer = MongoDBSaver(
                client=mongo_client,
                db_name="agent_p",
                checkpoint_collection_name="langgraph_checkpoints",
            )
        compiled = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": session_id}}
        return await asyncio.to_thread(compiled.invoke, {"message": message}, config)

    def _generate_question(
        self, 
        phase: str, 
        base_question: str, 
        history: list[str], 
        message: str,
        existing_facts: list[str],
    ) -> tuple[str, list[str], bool]:
        """Generate a contextual question using LLM. Returns (question, new_facts, ready_for_summary)."""
        if self._llm is None:
            return base_question, [], False

        try:
            history_text = "\n".join(history[-12:]) if history else "(No prior conversation)"
            
            prompt = DISCOVERY_SYSTEM_PROMPT.format(
                phase=phase,
                base_question=base_question,
                history=history_text,
                message=message,
            )
            
            start = time.perf_counter()
            resp = self._llm.invoke(prompt)
            latency_ms = (time.perf_counter() - start) * 1000.0
            
            raw_content = getattr(resp, "content", None)
            raw = _extract_llm_content(raw_content)
            if not raw.strip():
                return base_question, [], False

            raw = raw.strip()
            # Clean potential markdown code blocks
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            
            with suppress(ValueError, TypeError):
                turn = DiscoveryTurn.model_validate_json(raw)
                self.last_trace = {
                    "agent": "business_discovery_agent",
                    "stage": "question",
                    "model": self._config.model,
                    "prompt": prompt[:500],
                    "raw_output": raw,
                    "parsed_output": turn.model_dump(),
                    "latency_ms": latency_ms,
                }
                q = turn.question.strip()
                q = q if q.endswith("?") else q + "?"
                return q, turn.information_gathered, turn.ready_for_summary

        except Exception as e:
            log.warning("discovery_question_generation_failed", error=str(e))

        return base_question, [], False

    def _generate_analysis(self, history: list[str], facts: list[str]) -> dict:
        """Generate the final DiscoveryAnalysis using LLM."""
        if self._llm is None:
            return self._fallback_analysis(history)

        try:
            history_text = "\n".join(history)
            facts_text = "\n".join(f"- {f}" for f in facts) if facts else "No specific facts extracted"
            
            prompt = SUMMARY_SYSTEM_PROMPT.format(
                history=history_text,
                facts=facts_text,
            )
            
            start = time.perf_counter()
            resp = self._llm.invoke(prompt)
            latency_ms = (time.perf_counter() - start) * 1000.0
            
            raw_content = getattr(resp, "content", None)
            raw = _extract_llm_content(raw_content)
            if not raw.strip():
                return self._fallback_analysis(history)

            raw = raw.strip()
            # Clean potential markdown code blocks
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            
            with suppress(ValueError, TypeError):
                analysis = DiscoveryAnalysis.model_validate_json(raw)
                self.last_trace = {
                    "agent": "business_discovery_agent",
                    "stage": "analysis",
                    "model": self._config.model,
                    "prompt": prompt[:500],
                    "raw_output": raw,
                    "parsed_output": analysis.model_dump(),
                    "latency_ms": latency_ms,
                }
                return analysis.model_dump()

        except Exception as e:
            log.warning("discovery_analysis_generation_failed", error=str(e))

        return self._fallback_analysis(history)

    def _fallback_analysis(self, history: list[str]) -> dict:
        """Generate a basic analysis when LLM is unavailable."""
        stated = "Not specified"
        for h in history:
            if h.startswith("USER:"):
                stated = h.replace("USER:", "").strip()
                break

        return {
            "customer_stated_problem": stated,
            "identified_business_problem": "Further analysis required to identify root business problem",
            "hidden_root_risk": "Insufficient information to determine hidden risks",
            "business_urgency": "Medium",
        }

    def _format_analysis(self, analysis: dict) -> str:
        """Format the analysis as a readable string."""
        return "\n".join([
            "## Business Discovery Analysis",
            "",
            f"**Customer Stated Problem:** {analysis.get('customer_stated_problem', 'N/A')}",
            "",
            f"**Identified Business Problem:** {analysis.get('identified_business_problem', 'N/A')}",
            "",
            f"**Hidden Root Risk:** {analysis.get('hidden_root_risk', 'N/A')}",
            "",
            f"**Business Urgency Level:** {analysis.get('business_urgency', 'N/A')}",
        ])

    def get_analysis_data(self) -> dict | None:
        """Return the last analysis data for handoff to StructuringAgent."""
        if self.last_trace and self.last_trace.get("stage") == "analysis":
            return self.last_trace.get("parsed_output")
        return None
