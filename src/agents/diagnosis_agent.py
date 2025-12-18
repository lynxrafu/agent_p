"""DiagnosisAgent (Problem Structuring Agent): converts discovery analysis into a structured problem tree.

Implements the Strategic Consultant methodology from Research_about_prompting.md:
- MECE (Mutually Exclusive, Collectively Exhaustive) principle
- Issue Tree / Decomposition methodology
- Takes DiscoveryAnalysis as input, outputs structured ProblemTree

Output Structure:
- problem_type: Growth, Cost, Operational, Tech, Regulation, Organizational
- root_problem: The main problem at the root
- branches: 3+ main causes, each with 2-3 sub-causes
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Literal, List

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from contextlib import suppress

from src.db.mongo import Mongo
from src.models.task_input import TaskInput

import structlog

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
# Pydantic Schemas (from Research_about_prompting.md §3.2)
# =============================================================================

class SubCause(BaseModel):
    """A sub-cause in the problem tree (leaf node)."""
    description: str = Field(..., description="Description of the sub-cause")


class MainCause(BaseModel):
    """A main cause category with its sub-causes (branch)."""
    category: str = Field(..., description="Main cause category (e.g., Marketing, Competition)")
    sub_causes: List[SubCause] = Field(..., description="List of sub-causes under this category")


# Problem type classification (from Research_about_prompting.md)
ProblemType = Literal["Growth", "Cost", "Operational", "Tech", "Regulation", "Organizational"]


class ProblemTree(BaseModel):
    """Structured problem tree output (from Research_about_prompting.md §3.2)."""
    problem_type: ProblemType = Field(..., description="Classification of the problem")
    root_problem: str = Field(..., description="The main problem at the root of the tree")
    branches: List[MainCause] = Field(..., description="The main branches of the problem tree (3+ required)")


# Legacy compatibility - keep for existing code
class ProblemTreeNode(BaseModel):
    """A node in a problem tree (legacy format)."""
    label: str = Field(..., min_length=1)
    children: list["ProblemTreeNode"] = Field(default_factory=list)


ProblemTreeNode.model_rebuild()


class ProblemDiagnosisOutput(BaseModel):
    """Structured output from DiagnosisAgent (combines new + legacy formats)."""
    
    # New format (from spec)
    problem_type: str = Field(...)  # Growth, Cost, Operational, Tech, Regulation, Organizational
    main_problem: str = Field(..., min_length=1)
    
    # Tree can be either format
    tree: ProblemTreeNode = Field(...)
    branches: List[MainCause] = Field(default_factory=list)
    
    # Additional analysis
    key_unknowns: list[str] = Field(default_factory=list)
    next_questions: list[str] = Field(default_factory=list)


# =============================================================================
# Agent Configuration
# =============================================================================

@dataclass(frozen=True)
class DiagnosisAgentConfig:
    """Configuration for DiagnosisAgent."""
    google_api_key: str
    model: str
    mongo_url: str


# =============================================================================
# System Prompt (from Research_about_prompting.md §2.2)
# =============================================================================

STRUCTURING_SYSTEM_PROMPT = """ROLE: You are a Strategic Consultant. Your task is to break down a complex business problem into manageable components (Problem Structuring).

CRITICAL LANGUAGE RULE: You MUST respond in the SAME LANGUAGE as the conversation/input. If the input is in Turkish, write all text fields in Turkish. If in English, use English.

INPUT CONTEXT: You will receive a diagnosed problem and conversation details.

TASK INSTRUCTIONS:
1. **Classification:** Classify the problem into one of these categories:
   - Growth (revenue/sales decline, market share)
   - Cost (expenses, efficiency)
   - Operational (processes, delivery, quality)
   - Tech (systems, infrastructure)
   - Regulation (compliance, legal)
   - Organizational (team, culture, structure)

2. **Tree Construction:** Create an "Issue Tree" using MECE principle where:
   - **Root:** The main problem
   - **Branches:** At least 3 main causes
   - **Leaves:** Each main cause must have 2-3 specific sub-causes

3. **Format:** Output MUST be a valid JSON object. No markdown formatting, no conversational text.

CONVERSATION/ANALYSIS TO STRUCTURE:
{conversation}

OUTPUT FORMAT:
Return ONLY a JSON object with these exact fields:
{{
  "problem_type": "Growth" | "Cost" | "Operational" | "Tech" | "Regulation" | "Organizational",
  "root_problem": "The main problem statement",
  "branches": [
    {{
      "category": "Main Cause Category",
      "sub_causes": [
        {{"description": "Specific sub-cause 1"}},
        {{"description": "Specific sub-cause 2"}}
      ]
    }}
  ]
}}

EXAMPLE:
Input: "Satışlar düşüyor, pazarlama bütçesi verimsiz, rakipler fiyat kırdı."
Output:
{{
  "problem_type": "Growth",
  "root_problem": "Satışların Düşmesi",
  "branches": [
    {{
      "category": "Pazarlama Etkisizliği",
      "sub_causes": [
        {{"description": "Hedefleme yanlış yapılıyor"}},
        {{"description": "Reklam dönüşüm oranları (CTR) düştü"}}
      ]
    }},
    {{
      "category": "Rekabet Baskısı",
      "sub_causes": [
        {{"description": "Rakipler daha düşük fiyat sunuyor"}},
        {{"description": "Rakip ürün özellikleri bizden önde"}}
      ]
    }},
    {{
      "category": "Ürün-Pazar Uyumu",
      "sub_causes": [
        {{"description": "Müşteri ihtiyaçları değişti"}},
        {{"description": "Fiyat/Değer algısı zayıfladı"}}
      ]
    }}
  ]
}}"""


# =============================================================================
# DiagnosisAgent
# =============================================================================

class DiagnosisAgent:
    """DiagnosisAgent (Problem Structuring Agent): outputs a structured problem tree.
    
    Takes the output from BusinessDiscoveryAgent or conversation history
    and structures it into a MECE-compliant issue tree.
    """

    def __init__(
        self, 
        config: DiagnosisAgentConfig, 
        *, 
        llm: Any | None = None, 
        mongo: Mongo | None = None
    ) -> None:
        self._config = config
        self._mongo = mongo or Mongo(config.mongo_url)

        self._llm = llm
        if self._llm is None and config.google_api_key:
            temperature = 0.3  # Lower temperature for structured output
            self._llm = ChatGoogleGenerativeAI(
                model=config.model,
                google_api_key=config.google_api_key,
                temperature=temperature,
            )

        self.last_trace: dict[str, Any] | None = None

    async def process(self, input_data: TaskInput) -> ProblemDiagnosisOutput:
        """Generate a structured diagnosis output."""
        session_id = input_data.session_id
        conversation = input_data.task.strip()
        
        # Try to get conversation from session history
        if session_id:
            session_conv = await self._build_conversation_from_session(session_id=session_id)
            if session_conv:
                conversation = session_conv

        if not conversation:
            conversation = "User provided no details."

        if self._llm is None:
            return self._fallback(conversation)

        # Generate structured output
        prompt = STRUCTURING_SYSTEM_PROMPT.format(conversation=conversation)
        
        start = time.perf_counter()
        try:
            resp = await self._llm.ainvoke(prompt)
        except Exception as e:
            log.warning("diagnosis_llm_call_failed", error=str(e))
            return self._fallback(conversation)
            
        latency_ms = (time.perf_counter() - start) * 1000.0
        
        raw_content = getattr(resp, "content", None)
        raw = _extract_llm_content(raw_content)
        if not raw.strip():
            return self._fallback(conversation)

        raw = raw.strip()
        
        # Clean potential markdown code blocks
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        # Try to parse as ProblemTree first (new format)
        with suppress(ValueError, TypeError):
            tree_data = ProblemTree.model_validate_json(raw)
            
            # Convert to ProblemDiagnosisOutput
            legacy_tree = self._convert_to_legacy_tree(tree_data)
            
            output = ProblemDiagnosisOutput(
                problem_type=tree_data.problem_type,
                main_problem=tree_data.root_problem,
                tree=legacy_tree,
                branches=tree_data.branches,
                key_unknowns=[],
                next_questions=[],
            )
            
            self.last_trace = {
                "agent": "diagnosis_agent",
                "stage": "structuring",
                "model": self._config.model,
                "prompt": prompt[:500],
                "raw_output": raw,
                "parsed_output": output.model_dump(),
                "latency_ms": latency_ms,
            }
            return output

        # Fallback to legacy format parsing
        with suppress(ValueError, TypeError):
            output = ProblemDiagnosisOutput.model_validate_json(raw)
            self.last_trace = {
                "agent": "diagnosis_agent",
                "stage": "structuring",
                "model": self._config.model,
                "prompt": prompt[:500],
                "raw_output": raw,
                "parsed_output": output.model_dump(),
                "latency_ms": latency_ms,
            }
            return output

        return self._fallback(conversation)

    def _convert_to_legacy_tree(self, tree_data: ProblemTree) -> ProblemTreeNode:
        """Convert new ProblemTree format to legacy ProblemTreeNode format."""
        children = []
        for branch in tree_data.branches:
            branch_children = [
                ProblemTreeNode(label=sc.description)
                for sc in branch.sub_causes
            ]
            children.append(ProblemTreeNode(label=branch.category, children=branch_children))
        
        return ProblemTreeNode(label=tree_data.root_problem, children=children)

    async def _build_conversation_from_session(self, *, session_id: str) -> str:
        """Build conversation transcript from session history."""
        try:
            docs = await self._mongo.list_tasks_by_session(session_id, limit=50)
        except Exception:
            return ""
            
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
        """Generate basic structure when LLM is unavailable."""
        # Detect problem type from keywords
        conv_lower = conversation.lower()
        problem_type: str = "Operational"
        main = "Business performance issue"
        
        if any(k in conv_lower for k in ["sales", "satış", "revenue", "growth", "market"]):
            problem_type = "Growth"
            main = "Sales/Revenue Decline"
        elif any(k in conv_lower for k in ["cost", "maliyet", "expense", "budget"]):
            problem_type = "Cost"
            main = "Cost Management Issue"
        elif any(k in conv_lower for k in ["tech", "system", "software", "infrastructure"]):
            problem_type = "Tech"
            main = "Technology/Infrastructure Issue"
        elif any(k in conv_lower for k in ["compliance", "regulation", "legal"]):
            problem_type = "Regulation"
            main = "Compliance/Regulatory Issue"
        elif any(k in conv_lower for k in ["team", "culture", "organization", "hr"]):
            problem_type = "Organizational"
            main = "Organizational Issue"

        # Build default tree
        branches = [
            MainCause(
                category="External Factors",
                sub_causes=[
                    SubCause(description="Market conditions changing"),
                    SubCause(description="Competitive pressure"),
                ]
            ),
            MainCause(
                category="Internal Processes",
                sub_causes=[
                    SubCause(description="Process inefficiencies"),
                    SubCause(description="Resource allocation issues"),
                ]
            ),
            MainCause(
                category="Strategic Alignment",
                sub_causes=[
                    SubCause(description="Goals not clearly defined"),
                    SubCause(description="Execution gaps"),
                ]
            ),
        ]

        legacy_tree = ProblemTreeNode(
            label=main,
            children=[
                ProblemTreeNode(
                    label=b.category,
                    children=[ProblemTreeNode(label=sc.description) for sc in b.sub_causes]
                )
                for b in branches
            ],
        )

        return ProblemDiagnosisOutput(
            problem_type=problem_type,
            main_problem=main,
            tree=legacy_tree,
            branches=branches,
            key_unknowns=["Insufficient details to confirm root causes"],
            next_questions=["Which specific area is most impacted, and when did the issue begin?"],
        )

    @staticmethod
    def render_markdown(output: ProblemDiagnosisOutput) -> str:
        """Render the structured problem tree as a readable markdown string."""
        parts = [
            f"## Problem Type: **{output.problem_type}**",
            "",
            f"### Main Problem: {output.main_problem}",
            "",
            "### Problem Tree (Issue Tree)",
        ]
        
        # Use branches if available (new format), otherwise use legacy tree
        if output.branches:
            for branch in output.branches:
                parts.append(f"- **{branch.category}**")
                for sc in branch.sub_causes:
                    parts.append(f"  - {sc.description}")
        else:
            # Legacy tree rendering
            def walk(node: ProblemTreeNode, depth: int) -> list[str]:
                prefix = "  " * depth + "- "
                lines = [f"{prefix}{node.label}"]
                for c in node.children:
                    lines.extend(walk(c, depth + 1))
                return lines
            parts.extend(walk(output.tree, 0))

        if output.key_unknowns:
            parts.extend(["", "### Key Unknowns", *[f"- {u}" for u in output.key_unknowns]])
        
        if output.next_questions:
            parts.extend(["", "### Recommended Next Questions", *[f"- {q}" for q in output.next_questions]])
        
        return "\n".join(parts).strip()
