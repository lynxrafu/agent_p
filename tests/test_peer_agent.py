"""Tests for PeerAgent routing and Agent Registry.

Tests requirements from research_about_peer_agent.md:
- 2.1: Intent Recognition & Routing
- 2.2: Agent Management & Registry System
- 2.3: Context & Session Management
- 3.1: Extensibility (Open/Closed Principle)
- 3.3: Fault Tolerance
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os

import pytest

from src.agents.peer_agent import (
    AgentMetadata,
    AgentRegistry,
    PeerAgent,
    create_default_registry,
)
from src.core.settings import get_settings
from src.models.routing_models import TaskType
from src.models.task_input import TaskInput


# =============================================================================
# Requirement 2.1: Intent Recognition & Routing Tests
# =============================================================================

@pytest.mark.asyncio
async def test_peer_agent_defaults_unknown_to_content_when_llm_returns_unknown():
    """Req 2.1 Fallback: Unknown/ambiguous routes to ContentAgent."""
    os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
    os.environ["REDIS_URL"] = "redis://localhost:6379"
    os.environ["GOOGLE_API_KEY"] = "x"
    os.environ["GEMINI_MODEL"] = "gemini-3-flash-preview"

    # Use object() as LLM to force keyword fallback (no ainvoke method)
    agent = PeerAgent(get_settings(), llm=object())
    
    decision = await agent.route(TaskInput(task="Hello"))
    # "Hello" doesn't match any keywords, defaults to content
    assert decision.destination == TaskType.content


@pytest.mark.asyncio
async def test_peer_agent_falls_back_to_keyword_routing_when_llm_throws():
    """Req 3.3 Fault Tolerance: LLM failure triggers keyword fallback."""
    os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
    os.environ["REDIS_URL"] = "redis://localhost:6379"
    os.environ["GOOGLE_API_KEY"] = "x"
    os.environ["GEMINI_MODEL"] = "gemini-3-flash-preview"

    class FailingLLM:
        async def ainvoke(self, _):
            raise RuntimeError("LLM unavailable")

    agent = PeerAgent(get_settings(), llm=FailingLLM())
    
    # "langchain" and "örnek" match code keywords
    decision = await agent.route(TaskInput(task="LangChain metot örneğini bana göster"))
    assert decision.destination == TaskType.code


@pytest.mark.asyncio
async def test_peer_agent_session_stickiness_routes_to_active_agent_on_continuation():
    os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
    os.environ["REDIS_URL"] = "redis://localhost:6379"

    class DummySessionStore:
        async def get_active_agent(self, session_id: str):
            assert session_id == "s1"
            return {
                "session_id": session_id,
                "active_agent": "business_discovery",
                "updated_at": datetime.now(timezone.utc),
            }

    agent = PeerAgent(get_settings(), llm=object(), session_store=DummySessionStore())
    decision = await agent.route(TaskInput(task="It started 3 months ago", session_id="s1"))
    assert decision.destination == TaskType.business_discovery
    assert decision.rationale == "session_active_agent_stickiness"


@pytest.mark.asyncio
async def test_peer_agent_session_stickiness_does_not_override_explicit_new_intent():
    os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
    os.environ["REDIS_URL"] = "redis://localhost:6379"

    class DummySessionStore:
        async def get_active_agent(self, session_id: str):
            assert session_id == "s1"
            return {
                "session_id": session_id,
                "active_agent": "business_discovery",
                "updated_at": datetime.now(timezone.utc),
            }

    agent = PeerAgent(get_settings(), llm=object(), session_store=DummySessionStore())
    decision = await agent.route(TaskInput(task="Write python code to parse JSON", session_id="s1"))
    assert decision.destination == TaskType.code


@pytest.mark.asyncio
async def test_peer_agent_session_stickiness_honors_ttl():
    os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
    os.environ["REDIS_URL"] = "redis://localhost:6379"

    class DummySessionStore:
        async def get_active_agent(self, session_id: str):
            assert session_id == "s1"
            return {
                "session_id": session_id,
                "active_agent": "business_discovery",
                "updated_at": datetime.now(timezone.utc) - timedelta(hours=2),
            }

    # Force keyword routing by passing an LLM without ainvoke
    agent = PeerAgent(
        get_settings(),
        llm=object(),
        session_store=DummySessionStore(),
        session_stickiness_ttl_s=60,
    )
    decision = await agent.route(TaskInput(task="yes", session_id="s1"))
    # With expired stickiness, we fall back to keyword routing ("yes" matches nothing => content).
    assert decision.destination == TaskType.content


@pytest.mark.asyncio
async def test_keyword_routing_code_agent():
    """Req 2.1: Code keywords route to CodeAgent."""
    os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
    os.environ["REDIS_URL"] = "redis://localhost:6379"

    agent = PeerAgent(get_settings(), llm=object())
    
    test_cases = [
        "Write code to sort a list in Python",
        "Show me a code snippet for API calls",
        "Debug this JavaScript error",
        "Python ile kod yaz",  # Turkish: Write code with Python
    ]
    
    for task in test_cases:
        decision = await agent.route(TaskInput(task=task))
        assert decision.destination == TaskType.code, f"Failed for: {task}"


@pytest.mark.asyncio
async def test_keyword_routing_content_agent():
    """Req 2.1: Info/explanation keywords route to ContentAgent."""
    os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
    os.environ["REDIS_URL"] = "redis://localhost:6379"

    agent = PeerAgent(get_settings(), llm=object())
    
    test_cases = [
        "What is machine learning?",
        "Give me information about K2 model",
        "Explain how neural networks work",
        "Yapay zeka nedir?",  # Turkish: What is AI?
    ]
    
    for task in test_cases:
        decision = await agent.route(TaskInput(task=task))
        assert decision.destination == TaskType.content, f"Failed for: {task}"


@pytest.mark.asyncio
async def test_keyword_routing_business_discovery():
    """Req 2.1: Business problem keywords route to BusinessDiscoveryAgent."""
    os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
    os.environ["REDIS_URL"] = "redis://localhost:6379"

    agent = PeerAgent(get_settings(), llm=object())
    
    test_cases = [
        "Our sales declining, help me understand why",
        "Diagnose our customer complaints issue",
        "Satışlarımız düşüyor",  # Turkish: Our sales are declining
    ]
    
    for task in test_cases:
        decision = await agent.route(TaskInput(task=task))
        assert decision.destination == TaskType.business_discovery, f"Failed for: {task}"


@pytest.mark.asyncio
async def test_keyword_routing_diagnosis_agent():
    """Req 2.1: Problem structuring keywords route to DiagnosisAgent."""
    os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
    os.environ["REDIS_URL"] = "redis://localhost:6379"

    agent = PeerAgent(get_settings(), llm=object())
    
    test_cases = [
        "Build a problem tree from our findings",
        "Organize this into an issue tree",
        "Structure the root causes into MECE categories",
        "Problem ağacı çiz",  # Turkish: Draw a problem tree
    ]
    
    for task in test_cases:
        decision = await agent.route(TaskInput(task=task))
        assert decision.destination == TaskType.diagnosis, f"Failed for: {task}"


# =============================================================================
# Requirement 2.2: Agent Registry System Tests
# =============================================================================

def test_agent_registry_register_and_get():
    """Req 2.2: Dynamic agent registration."""
    registry = AgentRegistry()
    
    metadata = AgentMetadata(
        agent_id="test_agent",
        name="TestAgent",
        description="A test agent",
        capabilities=["testing"],
        trigger_keywords=["test", "testing"],
        priority=50,
    )
    
    registry.register(metadata)
    
    retrieved = registry.get("test_agent")
    assert retrieved is not None
    assert retrieved.agent_id == "test_agent"
    assert retrieved.name == "TestAgent"


def test_agent_registry_unregister():
    """Req 2.2: Agent can be removed from registry."""
    registry = AgentRegistry()
    
    metadata = AgentMetadata(
        agent_id="temp_agent",
        name="TempAgent",
        description="Temporary",
        capabilities=[],
        trigger_keywords=[],
    )
    
    registry.register(metadata)
    assert registry.get("temp_agent") is not None
    
    result = registry.unregister("temp_agent")
    assert result is True
    assert registry.get("temp_agent") is None


def test_agent_registry_priority_ordering():
    """Req 2.2: Agents returned in priority order (highest first)."""
    registry = AgentRegistry()
    
    registry.register(AgentMetadata(
        agent_id="low", name="Low", description="", 
        capabilities=[], trigger_keywords=[], priority=10,
    ))
    registry.register(AgentMetadata(
        agent_id="high", name="High", description="",
        capabilities=[], trigger_keywords=[], priority=100,
    ))
    registry.register(AgentMetadata(
        agent_id="medium", name="Medium", description="",
        capabilities=[], trigger_keywords=[], priority=50,
    ))
    
    agents = registry.get_all()
    assert [a.agent_id for a in agents] == ["high", "medium", "low"]


def test_agent_registry_find_by_keyword():
    """Req 2.2: Find agent by keyword matching."""
    registry = AgentRegistry()
    
    registry.register(AgentMetadata(
        agent_id="coder", name="Coder", description="",
        capabilities=[], trigger_keywords=["code", "python"],
        priority=100,
    ))
    registry.register(AgentMetadata(
        agent_id="researcher", name="Researcher", description="",
        capabilities=[], trigger_keywords=["info", "explain"],
        priority=50,
    ))
    
    # Should find coder (has "code" keyword)
    result = registry.find_by_keyword("Write me some code")
    assert result is not None
    assert result.agent_id == "coder"
    
    # Should find researcher (has "explain" keyword)
    result = registry.find_by_keyword("Please explain this concept")
    assert result is not None
    assert result.agent_id == "researcher"
    
    # Should return None (no matching keywords)
    result = registry.find_by_keyword("Hello world")
    assert result is None


def test_default_registry_has_all_agents():
    """Req 2.2: Default registry includes all expected agents."""
    registry = create_default_registry()
    
    expected_agents = {"content", "code", "business_discovery", "diagnosis"}
    actual_agents = {a.agent_id for a in registry.get_all()}
    
    assert expected_agents == actual_agents


# =============================================================================
# Requirement 3.1: Extensibility (Open/Closed Principle) Tests
# =============================================================================

@pytest.mark.asyncio
async def test_hot_plug_new_agent():
    """Req 2.2 & 3.1: Hot-plug a new agent without modifying core code."""
    os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
    os.environ["REDIS_URL"] = "redis://localhost:6379"

    agent = PeerAgent(get_settings(), llm=object())
    
    # Verify supported agents before
    initial_agents = agent.get_supported_agents()
    assert "custom_agent" not in initial_agents
    
    # Hot-plug a new agent
    agent.register_agent(AgentMetadata(
        agent_id="custom_agent",
        name="CustomAgent",
        description="A custom agent",
        capabilities=["custom tasks"],
        trigger_keywords=["custom", "special"],
        priority=200,  # Highest priority
    ))
    
    # Verify it's now supported
    updated_agents = agent.get_supported_agents()
    assert "custom_agent" in updated_agents


def test_registry_prompt_generation():
    """Req 3.1: Registry dynamically generates LLM prompt section."""
    registry = AgentRegistry()
    
    registry.register(AgentMetadata(
        agent_id="demo",
        name="DemoAgent",
        description="Demonstration agent",
        capabilities=["demo", "test"],
        trigger_keywords=["demo"],
        examples=['"Run a demo"'],
    ))
    
    prompt = registry.build_prompt_section()
    
    assert "demo" in prompt
    assert "DemoAgent" in prompt
    assert "Demonstration agent" in prompt
    assert "demo, test" in prompt


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

@pytest.mark.asyncio
async def test_empty_task_routes_to_content():
    """Empty task should default to content agent."""
    os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
    os.environ["REDIS_URL"] = "redis://localhost:6379"

    agent = PeerAgent(get_settings(), llm=object())
    
    decision = await agent.route(TaskInput(task=""))
    assert decision.destination == TaskType.content
    assert decision.confidence == 0.0


@pytest.mark.asyncio
async def test_whitespace_only_task_routes_to_content():
    """Whitespace-only task should default to content agent."""
    os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
    os.environ["REDIS_URL"] = "redis://localhost:6379"

    agent = PeerAgent(get_settings(), llm=object())
    
    decision = await agent.route(TaskInput(task="   \n\t  "))
    assert decision.destination == TaskType.content
