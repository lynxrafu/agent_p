from __future__ import annotations

import os

import pytest


@pytest.mark.asyncio
async def test_worker_writes_agent_logs_when_agent_trace_available(monkeypatch):
    os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
    os.environ.setdefault("GOOGLE_API_KEY", "x")
    os.environ.setdefault("TAVILY_API_KEY", "y")
    os.environ.setdefault("GEMINI_MODEL", "gemini-3-pro")

    from src.core import jobs as jobs_mod
    from src.models.routing_models import RoutingDecision, TaskType

    agent_logs = []

    class DummyMongo:
        def __init__(self, mongo_url: str):
            _ = mongo_url

        async def update_task(self, task_id: str, status: str, result=None):
            _ = (task_id, status, result)

        async def set_task_route(self, task_id: str, route: str, route_confidence, route_rationale):
            _ = (task_id, route, route_confidence, route_rationale)

        async def create_agent_log(self, entry):
            agent_logs.append(entry.model_dump())

        async def close(self):
            return None

    class DummyPeer:
        def __init__(self):
            self.last_trace = {
                "stage": "routing",
                "model": "gemini-3-pro",
                "prompt": "PROMPT",
                "raw_output": "{\"destination\":\"code\",\"confidence\":0.9,\"rationale\":\"x\"}",
                "parsed_output": {"destination": "code", "confidence": 0.9, "rationale": "x"},
                "latency_ms": 12.3,
            }

        async def route(self, _input_data):
            return RoutingDecision(destination=TaskType.code, confidence=0.9, rationale="x")

    class DummyCode:
        def __init__(self):
            self.last_trace = {
                "stage": "generation",
                "model": "gemini-3-pro",
                "prompt": "CODE_PROMPT",
                "raw_output": "RAW",
                "parsed_output": "```python\nprint(1)\n```",
                "latency_ms": 34.5,
            }

        async def process(self, _input_data):
            return "```python\nprint(1)\n```"

    monkeypatch.setattr(jobs_mod, "Mongo", DummyMongo)
    monkeypatch.setattr(jobs_mod, "PeerAgent", lambda *_args, **_kwargs: DummyPeer())
    monkeypatch.setattr(jobs_mod, "CodeAgent", lambda *_args, **_kwargs: DummyCode())

    await jobs_mod._process_task(task_id="t1", task="write code", mongo_url="mongodb://x", session_id="s1")

    agents = {e["agent"] for e in agent_logs}
    assert "peer_agent" in agents
    assert "code_agent" in agents
    for e in agent_logs:
        assert e["task_id"] == "t1"
        assert e["session_id"] == "s1"


