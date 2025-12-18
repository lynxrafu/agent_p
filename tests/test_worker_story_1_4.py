from __future__ import annotations

import os

import pytest


@pytest.mark.asyncio
async def test_worker_marks_failed_and_does_not_crash_on_unexpected_agent_error(monkeypatch):
    os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
    os.environ.setdefault("GOOGLE_API_KEY", "x")
    os.environ.setdefault("TAVILY_API_KEY", "y")
    os.environ.setdefault("GEMINI_MODEL", "gemini-3-pro")

    from src.core import jobs as jobs_mod
    from src.models.routing_models import RoutingDecision, TaskType

    updates = []

    class DummyMongo:
        def __init__(self, mongo_url: str):
            _ = mongo_url

        async def update_task(self, task_id: str, status: str, result=None):
            updates.append({"task_id": task_id, "status": status, "result": result})

        async def set_task_route(self, task_id: str, route: str, route_confidence, route_rationale):
            _ = (task_id, route, route_confidence, route_rationale)

        async def close(self):
            return None

    class DummyPeer:
        async def route(self, input_data):
            _ = input_data
            return RoutingDecision(destination=TaskType.content, confidence=0.9, rationale="test")

    class DummyContent:
        async def process(self, input_data):
            _ = input_data
            raise RuntimeError("boom")

    monkeypatch.setattr(jobs_mod, "Mongo", DummyMongo)
    monkeypatch.setattr(jobs_mod, "PeerAgent", lambda *_args, **_kwargs: DummyPeer())
    monkeypatch.setattr(jobs_mod, "ContentAgent", lambda *_args, **_kwargs: DummyContent())

    # Should not raise.
    await jobs_mod._process_task(task_id="t1", task="q", mongo_url="mongodb://x")

    assert updates[0]["status"] == "processing"
    assert updates[-1]["status"] == "failed"
    assert updates[-1]["result"]["stage"] == "unknown"


