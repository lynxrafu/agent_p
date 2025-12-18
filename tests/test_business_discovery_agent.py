from __future__ import annotations

import pytest

from src.agents.business_discovery_agent import BusinessDiscoveryAgent, BusinessDiscoveryAgentConfig
from src.models.task_input import TaskInput


@pytest.mark.asyncio
async def test_business_discovery_agent_resumes_session_state_with_injected_checkpointer():
    # Use LangGraph's in-memory saver so the test doesn't touch network or Mongo.
    import importlib

    InMemorySaver = importlib.import_module("langgraph.checkpoint.memory").InMemorySaver

    saver = InMemorySaver()

    # Force deterministic path by disabling LLM chain.
    agent = BusinessDiscoveryAgent(
        BusinessDiscoveryAgentConfig(google_api_key="", model="gemini-3-pro", mongo_url="mongodb://x"),
        llm=None,
        checkpointer=saver,
    )

    q1 = await agent.process(TaskInput(task="Sales are down", session_id="sess1"))
    q2 = await agent.process(TaskInput(task="It started 3 months ago", session_id="sess1"))

    assert q1 != q2
    assert q1.endswith("?")
    assert q2.endswith("?")


