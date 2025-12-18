from __future__ import annotations

import os

import pytest

from src.agents.peer_agent import PeerAgent
from src.core.settings import get_settings
from src.models.routing_models import RoutingDecision, TaskType
from src.models.task_input import TaskInput


@pytest.mark.asyncio
async def test_peer_agent_defaults_unknown_to_content_when_llm_returns_unknown():
    os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
    os.environ["REDIS_URL"] = "redis://localhost:6379"
    os.environ["GOOGLE_API_KEY"] = "x"
    os.environ["GEMINI_MODEL"] = "gemini-3-pro"

    class DummyChain:
        async def ainvoke(self, _):
            return RoutingDecision(destination=TaskType.unknown, confidence=0.2, rationale="ambiguous")

    agent = PeerAgent(get_settings(), llm=object())
    # Force chain injection without calling real LLM.
    agent._chain = DummyChain()  # type: ignore[attr-defined]  # noqa: SLF001  # pylint: disable=protected-access

    decision = await agent.route(TaskInput(task="some ambiguous task"))
    assert decision.destination == TaskType.content


@pytest.mark.asyncio
async def test_peer_agent_falls_back_to_keyword_routing_when_llm_throws():
    os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
    os.environ["REDIS_URL"] = "redis://localhost:6379"
    os.environ["GOOGLE_API_KEY"] = "x"
    os.environ["GEMINI_MODEL"] = "gemini-3-pro"

    class DummyChain:
        async def ainvoke(self, _):
            raise RuntimeError("boom")

    agent = PeerAgent(get_settings(), llm=object())
    agent._chain = DummyChain()  # type: ignore[attr-defined]  # noqa: SLF001  # pylint: disable=protected-access

    decision = await agent.route(TaskInput(task="LangChain metot örneğini bana göster"))
    assert decision.destination == TaskType.code


