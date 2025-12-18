"""PeerAgent: intelligent task router using Chain-of-Thought reasoning.

Routes incoming tasks to the appropriate specialist agent:
- ContentAgent: Information queries, summaries, explanations
- CodeAgent: Code generation, debugging, implementation help
- BusinessDiscoveryAgent: Business problem diagnosis, root cause analysis
- DiagnosisAgent: Problem structuring, issue tree generation

Features:
- LLM-based routing with Chain-of-Thought prompting
- Keyword-based fallback routing
- Dynamic Agent Registry (Open/Closed Principle)
- Session context awareness for multi-turn conversations

Architecture follows SOLID principles:
- Open for extension (add new agents via registry)
- Closed for modification (no core changes needed)
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol, runtime_checkable

import structlog
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from pydantic import ValidationError

from src.core.settings import Settings
from src.models.routing_models import RoutingDecision, TaskType
from src.models.task_input import TaskInput


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

log = structlog.get_logger(__name__)


# =============================================================================
# Agent Registry System (Requirement 2.2: Dynamic Registration)
# =============================================================================

@dataclass(frozen=True)
class AgentMetadata:
    """Metadata for a registered agent.
    
    Implements Requirement 2.2:
    - Agent Identity (ID/Name)
    - Agent Capability Description  
    - Trigger Keywords
    """
    agent_id: str
    name: str
    description: str
    capabilities: list[str]
    trigger_keywords: list[str]
    examples: list[str] = field(default_factory=list)
    priority: int = 0  # Higher priority = checked first in keyword routing


@runtime_checkable
class SpecialistAgent(Protocol):
    """Protocol for specialist agents (for type checking)."""
    
    async def run(self, input_data: TaskInput) -> dict[str, Any]:
        """Execute the agent's task."""
        raise NotImplementedError


@runtime_checkable
class SessionStore(Protocol):
    """Session state store for system-level continuity (active agent per session_id)."""

    async def get_active_agent(self, session_id: str) -> dict[str, Any] | None:
        raise NotImplementedError


class AgentRegistry:
    """Dynamic Agent Registry for hot-pluggable agent management.
    
    Implements Requirement 2.2:
    - Dynamic Registration: New agents can be added at runtime
    - Hot-Pluggability: No core codebase modifications needed
    
    Implements Requirement 3.1 (Open/Closed Principle):
    - Open for extension: Register new agents via `register()`
    - Closed for modification: Core routing logic unchanged
    """
    
    def __init__(self) -> None:
        self._agents: dict[str, AgentMetadata] = {}
    
    def register(self, metadata: AgentMetadata) -> None:
        """Register a new agent with the router.
        
        Args:
            metadata: Agent metadata including ID, description, and keywords
        """
        self._agents[metadata.agent_id] = metadata
        log.info(
            "agent_registered",
            agent_id=metadata.agent_id,
            name=metadata.name,
            keywords_count=len(metadata.trigger_keywords),
        )
    
    def unregister(self, agent_id: str) -> bool:
        """Remove an agent from the registry.
        
        Args:
            agent_id: The agent's unique identifier
            
        Returns:
            True if agent was removed, False if not found
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            log.info("agent_unregistered", agent_id=agent_id)
            return True
        return False
    
    def get(self, agent_id: str) -> AgentMetadata | None:
        """Get agent metadata by ID."""
        return self._agents.get(agent_id)
    
    def get_all(self) -> list[AgentMetadata]:
        """Get all registered agents sorted by priority (highest first)."""
        return sorted(self._agents.values(), key=lambda a: -a.priority)
    
    def find_by_keyword(self, text: str) -> AgentMetadata | None:
        """Find agent matching keywords in text (priority order).
        
        Args:
            text: The input text to match against
            
        Returns:
            The first matching agent's metadata, or None
            
        Note: Short keywords (<=3 chars like 'go', 'sql') use word boundary
        matching to avoid false positives (e.g., 'go' in 'categories').
        """
        import re
        t = text.lower()
        for agent in self.get_all():
            for keyword in agent.trigger_keywords:
                kw = keyword.lower()
                # For short keywords, use word boundary to avoid substring false positives
                if len(kw) <= 3:
                    # Match whole word only
                    if re.search(r'\b' + re.escape(kw) + r'\b', t):
                        return agent
                elif kw in t:
                    return agent
        return None
    
    def build_prompt_section(self) -> str:
        """Generate the 'Available Agents' section for LLM prompt.
        
        Dynamically builds prompt from registered agents.
        """
        lines = ["## Available Agents\n"]
        for i, agent in enumerate(self.get_all(), 1):
            lines.append(f"{i}. **{agent.agent_id}** - {agent.name}")
            lines.append(f"   - For: {agent.description}")
            lines.append(f"   - Capabilities: {', '.join(agent.capabilities)}")
            if agent.examples:
                lines.append(f"   - Examples: {', '.join(agent.examples[:3])}")
            lines.append("")
        return "\n".join(lines)


# =============================================================================
# Default Agent Configurations
# =============================================================================

def create_default_registry() -> AgentRegistry:
    """Create registry with default agent configurations.
    
    Satisfies Requirement 2.1 keyword examples:
    - Code: "Code", "implementation", "Python", "script"
    - Content: "What is", "info", "history", "explain"
    """
    registry = AgentRegistry()
    
    # ContentAgent - lowest priority (default fallback)
    registry.register(AgentMetadata(
        agent_id="content",
        name="ContentAgent",
        description="Information queries, explanations, summaries, research questions",
        capabilities=["Web search", "Synthesizing information", "Explanations"],
        trigger_keywords=[
            # English - from PRD requirements
            "what is", "info", "information", "history", "explain", "tell me about",
            "describe", "summarize", "summary", "research", "learn about",
            "how does", "why does", "overview", "definition",
            # Turkish
            "nedir", "bilgi", "açıkla", "anlat", "tarih", "özetle",
            "hakkında", "araştır", "tanım",
        ],
        examples=[
            '"Give me information about X"',
            '"What is machine learning?"',
            '"Explain how Y works"',
        ],
        priority=0,  # Lowest - default fallback
    ))
    
    # CodeAgent - high priority
    registry.register(AgentMetadata(
        agent_id="code",
        name="CodeAgent",
        description="Code generation, debugging, implementation, API usage examples",
        capabilities=["Multi-language code generation", "Debugging", "Code explanations"],
        trigger_keywords=[
            # English - from PRD requirements (explicit code-related)
            "code", "coding", "implementation", "implement", "script",
            "program", "function", "class", "method", "debug", "fix bug",
            "api", "sdk", "library", "snippet",
            "write code", "write function", "write script",
            # Turkish
            "kod", "kodla", "programla", "fonksiyon", "metot",
            "kod yaz", "hata ayıkla",
            # Specific technologies (these are strong signals for code)
            "python", "javascript", "typescript", "java", "langchain", "fastapi",
            "react", "node", "sql", "bash", "powershell", "rust", "go",
        ],
        examples=[
            '"Write a Python function that..."',
            '"Debug this code..."',
            '"Show me a LangChain example"',
        ],
        priority=100,  # Highest - explicit code requests
    ))
    
    # BusinessDiscoveryAgent
    registry.register(AgentMetadata(
        agent_id="business_discovery",
        name="BusinessDiscoveryAgent",
        description="Business problem diagnosis, root cause analysis, interview-style discovery",
        capabilities=["Structured questioning", "5 Whys analysis", "Business impact assessment"],
        trigger_keywords=[
            # English
            "sales declining", "revenue drop", "customer complaints",
            "diagnose", "root cause", "why is", "help me understand",
            "business problem", "operational issue", "process problem",
            "interview", "discovery", "problem diagnosis",
            # Turkish
            "satış düşüyor", "gelir azalıyor", "müşteri şikayet",
            "kök neden", "neden oluyor", "anlamamı yardım",
            "iş problemi", "operasyon sorunu", "süreç problemi",
            "satışlarımız", "maliyetimiz", "verimlilik", "şikayet",
        ],
        examples=[
            '"Our sales are declining, help me understand why"',
            '"Diagnose our customer complaint issue"',
        ],
        priority=50,
    ))
    
    # DiagnosisAgent (Problem Structuring)
    registry.register(AgentMetadata(
        agent_id="diagnosis",
        name="DiagnosisAgent",
        description="Structuring identified problems into issue trees, MECE analysis",
        capabilities=["Problem tree generation", "MECE breakdown", "Root cause hierarchy"],
        trigger_keywords=[
            # English
            "problem tree", "issue tree", "cause tree", "root causes",
            "structure the problem", "break down", "categorize causes",
            "mece", "sub-causes", "hierarchy", "organize findings",
            # Turkish
            "problem ağacı", "sorun ağacı", "kök nedenler", "alt nedenler",
            "yapılandır", "kategorize", "nedenleri grupla", "ağaç oluştur",
        ],
        examples=[
            '"Structure this problem into causes"',
            '"Create an issue tree from our findings"',
        ],
        priority=75,
    ))
    
    return registry


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class PeerAgentConfig:
    """Configuration for PeerAgent LLM routing."""
    google_api_key: str | None
    model: str


# =============================================================================
# Chain-of-Thought System Prompt Template
# =============================================================================

ROUTING_SYSTEM_PROMPT_TEMPLATE = """You are PeerAgent, an intelligent task router. Your job is to analyze the user's request and route it to the most appropriate specialist agent.

{available_agents}

## Routing Rules (Apply in Order)

1. If the request mentions code, programming, API, implementation, debugging → **code**
2. If the request asks to structure/organize findings into a tree or causes → **diagnosis**
3. If the request describes a business problem and asks for help understanding it → **business_discovery**
4. For general information, explanations, or unclear requests → **content**

## Chain-of-Thought Process

Before deciding, reason through:
1. What is the user actually asking for?
2. What type of output do they expect?
3. Which agent is best equipped to help?

## Output Format

Return ONLY a JSON object (no markdown):
{{
  "destination": "<agent_id>",
  "confidence": 0.0-1.0,
  "rationale": "Brief explanation of your reasoning"
}}

## Examples with Reasoning

### Example 1
User: "kimi k2 modeli hakkında beni bilgilendir"
Reasoning: User wants information about the Kimi K2 model. This is a knowledge/research question.
Output: {{"destination": "content", "confidence": 0.9, "rationale": "Information request about a specific topic"}}

### Example 2
User: "Python ile bir dosyayı okuyup yazan kod yaz"
Reasoning: User explicitly asks for code ("kod yaz"). This is a code generation task.
Output: {{"destination": "code", "confidence": 0.95, "rationale": "Explicit code generation request"}}

### Example 3
User: "LangChain metot örneğini bana göster"
Reasoning: User wants to see a method example ("örnek göster"). This requires code demonstration.
Output: {{"destination": "code", "confidence": 0.9, "rationale": "Request for code example/demonstration"}}

### Example 4
User: "Satışlarımız yılda %20 düşüyor, kök nedeni bulmam için bana sorular sorar mısın?"
Reasoning: User has a business problem (declining sales) and wants diagnostic questions. This is business discovery.
Output: {{"destination": "business_discovery", "confidence": 0.95, "rationale": "Business problem requiring root cause discovery"}}

### Example 5
User: "What is machine learning?"
Reasoning: User asks "what is" - this is an information/explanation request.
Output: {{"destination": "content", "confidence": 0.95, "rationale": "Explanation request matching 'what is' pattern"}}

### Example 6
User: "Konuşmayı problem ağacına dönüştür ve nedenleri grupla"
Reasoning: User wants to structure findings into a problem tree with grouped causes.
Output: {{"destination": "diagnosis", "confidence": 0.95, "rationale": "Request to structure into problem tree"}}

### Example 7 (Ambiguous Request)
User: "Hello"
Reasoning: Simple greeting with no clear intent. Default to content agent.
Output: {{"destination": "content", "confidence": 0.3, "rationale": "Ambiguous request, defaulting to content"}}"""


# =============================================================================
# PeerAgent Class
# =============================================================================

class PeerAgent:
    """PeerAgent: intelligent task router with CoT reasoning and fallback.
    
    Implements:
    - 2.1 Intent Recognition & Routing (LLM + keyword fallback)
    - 2.2 Agent Management via Registry (dynamic, hot-pluggable)
    - 2.3 Session Continuity (session context awareness)
    - 3.1 Open/Closed Principle (extend via registry)
    - 3.2 Modularity (decoupled from agent internals)
    - 3.3 Fault Tolerance (error handling with fallback)
    - 3.4 Performance (O(n) keyword matching, async LLM)
    """

    def __init__(
        self, 
        settings: Settings, 
        *, 
        llm: Any | None = None,
        registry: AgentRegistry | None = None,
        session_store: SessionStore | None = None,
        session_stickiness_ttl_s: int = 30 * 60,
    ) -> None:
        self._settings = settings
        self._config = PeerAgentConfig(
            google_api_key=settings.GOOGLE_API_KEY, 
            model=settings.GEMINI_MODEL
        )
        
        # Agent Registry (Requirement 2.2)
        self._registry = registry or create_default_registry()

        # LLM is optional. If not configured or fails, fall back to keyword routing.
        self._llm = llm
        if self._llm is None and self._config.google_api_key:
            temperature = 0.3
            self._llm = ChatGoogleGenerativeAI(
                model=self._config.model,
                google_api_key=self._config.google_api_key,
                temperature=temperature,
            )

        self.last_trace: dict[str, Any] | None = None
        self._session_store = session_store
        self._session_stickiness_ttl_s = max(0, int(session_stickiness_ttl_s))
        
        # Build system prompt from registry
        self._system_prompt = ROUTING_SYSTEM_PROMPT_TEMPLATE.format(
            available_agents=self._registry.build_prompt_section()
        )

    @property
    def registry(self) -> AgentRegistry:
        """Access the agent registry for dynamic registration."""
        return self._registry

    def register_agent(self, metadata: AgentMetadata) -> None:
        """Register a new agent (hot-plug capability).
        
        Requirement 2.2: Hot-Pluggability
        """
        self._registry.register(metadata)
        # Rebuild prompt with new agent
        self._system_prompt = ROUTING_SYSTEM_PROMPT_TEMPLATE.format(
            available_agents=self._registry.build_prompt_section()
        )

    async def route(self, input_data: TaskInput) -> RoutingDecision:
        """Route a task to an agent destination with fallback behavior.
        
        Implements Requirement 2.1:
        1. Try LLM routing with Chain-of-Thought reasoning
        2. Fall back to keyword-based routing if LLM fails
        3. Default to content agent for ambiguous requests (Scenario C)
        
        Implements Requirement 2.3:
        - Session context is checked for continuation
        """
        task = input_data.task.strip()
        
        if not task:
            return RoutingDecision(
                destination=TaskType.content, 
                confidence=0.0, 
                rationale="empty_task_defaults_to_content"
            )

        # Requirement 2.3: Session Continuity
        if input_data.session_id:
            session_route = await self._check_session_context(input_data.session_id, task)
            if session_route:
                return session_route

        # Attempt LLM routing with CoT
        if self._llm is not None and hasattr(self._llm, "ainvoke"):
            llm_result = await self._llm_route(task)
            if llm_result:
                return llm_result
            log.warning("peer_agent_llm_routing_failed_fallback_to_keyword", task=task[:100])

        # Requirement 2.1: Keyword-Based Routing
        return self._keyword_route(task)

    async def _llm_route(self, task: str) -> RoutingDecision | None:
        """Route using LLM with Chain-of-Thought prompting.
        
        Implements Requirement 3.3: Fault Tolerance
        - Catches exceptions and returns None to trigger fallback
        """
        with suppress(ValidationError, ValueError, TypeError, RuntimeError, TimeoutError, OSError, ChatGoogleGenerativeAIError):
            prompt = f"{self._system_prompt}\n\n## Current Task\nUser: \"{task}\"\n\nAnalyze and route:"
            
            start = time.perf_counter()
            resp = await self._llm.ainvoke(prompt)
            latency_ms = (time.perf_counter() - start) * 1000.0

            raw_content = getattr(resp, "content", None)
            raw = _extract_llm_content(raw_content)
            if not raw.strip():
                return None
            
            raw = raw.strip()
            
            # Clean potential markdown code blocks
            if raw.startswith("```"):
                parts = raw.split("```")
                if len(parts) >= 2:
                    raw = parts[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()

            decision = RoutingDecision.model_validate_json(raw)

            self.last_trace = {
                "agent": "peer_agent",
                "stage": "routing",
                "model": self._config.model,
                "prompt": prompt[:500],
                "raw_output": raw,
                "parsed_output": decision.model_dump(),
                "latency_ms": latency_ms,
            }

            # Requirement 2.1 Fallback: unknown → content
            if decision.destination == TaskType.unknown:
                return RoutingDecision(
                    destination=TaskType.content, 
                    confidence=decision.confidence, 
                    rationale=decision.rationale
                )
            
            return decision

        return None

    async def _check_session_context(self, session_id: str, task: str) -> RoutingDecision | None:
        """Check session context for conversation continuity.
        
        Implements Requirement 2.3: Session Continuity
        - Preserves conversation context across follow-up questions
        - Routes continuation responses to the active agent
        """
        # Prefer persisted session state (if available) for true multi-worker continuity.
        if self._session_store is not None:
            try:
                doc = await self._session_store.get_active_agent(session_id)
            except (TimeoutError, OSError, ValueError, TypeError, RuntimeError):
                doc = None
            if isinstance(doc, dict):
                active = doc.get("active_agent")
                updated_at = doc.get("updated_at")

                # Honor TTL to avoid "sticky forever" sessions.
                if active and self._session_stickiness_ttl_s > 0 and isinstance(updated_at, datetime):
                    # Handle both timezone-aware and naive datetimes from MongoDB
                    now = datetime.now(timezone.utc)
                    if updated_at.tzinfo is None:
                        # MongoDB returned naive datetime, assume UTC
                        updated_at = updated_at.replace(tzinfo=timezone.utc)
                    age = now - updated_at
                    if age > timedelta(seconds=self._session_stickiness_ttl_s):
                        active = None

                # Only stick if the new message looks like a continuation and does not
                # explicitly request a different agent type.
                if isinstance(active, str) and active:
                    active_tt = None
                    with suppress(ValueError):
                        active_tt = TaskType(active)
                    if active_tt is not None and self._looks_like_continuation(task):
                        explicit = self._keyword_route(task).destination
                        # If keyword routing suggests a *different* strong intent (code/diagnosis/business),
                        # don't override the user.
                        if explicit in {TaskType.code, TaskType.diagnosis, TaskType.business_discovery} and explicit != active_tt:
                            return None
                        return RoutingDecision(
                            destination=active_tt,
                            confidence=0.85,
                            rationale="session_active_agent_stickiness",
                        )
        
        return None

    @staticmethod
    def _looks_like_continuation(task: str) -> bool:
        """Heuristic: short, answer-like messages are likely continuation turns."""
        task_lower = task.lower()
        continuation_patterns = [
            # Simple affirmative/negative
            "evet", "hayır", "yes", "no", "ok", "okay", "tamam",
            # Numeric/quantitative responses
            "ayda", "yılda", "monthly", "yearly", "%", "tl", "$",
            # Department/team responses
            "departman", "department", "team", "ekip",
            # Continuation signals
            "daha fazla", "more", "continue", "devam", "sonraki",
        ]
        is_short = len(task.split()) <= 12
        is_cont = any(p in task_lower for p in continuation_patterns)
        # Treat "short generic answer" as continuation even without explicit pattern.
        return is_short or is_cont

    def _keyword_route(self, task: str) -> RoutingDecision:
        """Route based on keyword matching from registry.
        
        Implements Requirement 2.1: Keyword-Based Routing
        Uses agent priority order (highest first)
        
        Implements Requirement 3.4: Performance
        - O(n*k) where n=agents, k=keywords per agent
        - Returns on first match for efficiency
        """
        agent = self._registry.find_by_keyword(task)
        
        if agent:
            # Map agent_id to TaskType
            try:
                task_type = TaskType(agent.agent_id)
            except ValueError:
                task_type = TaskType.content
            
            return RoutingDecision(
                destination=task_type, 
                confidence=0.6, 
                rationale=f"keyword_match_agent:{agent.agent_id}"
            )

        # Requirement 2.1 Fallback: Default to content agent
        return RoutingDecision(
            destination=TaskType.content, 
            confidence=0.4, 
            rationale="default_content_no_keyword_match"
        )

    def get_supported_agents(self) -> list[str]:
        """Return list of supported agent destinations from registry."""
        return [agent.agent_id for agent in self._registry.get_all()]
