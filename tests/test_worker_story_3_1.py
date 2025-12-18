from __future__ import annotations

import os

import pytest


@pytest.mark.asyncio
async def test_worker_business_discovery_route_runs_agent_and_completes(monkeypatch):
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
            return RoutingDecision(destination=TaskType.business_discovery, confidence=0.9, rationale="test")

    class DummyBiz:
        async def process(self, input_data):
            assert getattr(input_data, "session_id", None) == "s1"
            return "What is the main problem and when did it start?"

    monkeypatch.setattr(jobs_mod, "Mongo", DummyMongo)
    monkeypatch.setattr(jobs_mod, "PeerAgent", lambda *_args, **_kwargs: DummyPeer())
    monkeypatch.setattr(jobs_mod, "BusinessDiscoveryAgent", lambda *_args, **_kwargs: DummyBiz())

    # Use the public worker entrypoint to avoid testing protected internals.
    import asyncio

    await asyncio.to_thread(jobs_mod.process_task_job, task_id="t1", task="Sales are down", mongo_url="mongodb://x", session_id="s1")

    assert updates[0]["status"] == "processing"
    assert updates[-1]["status"] == "completed"
    assert "problem" in (updates[-1]["result"] or {}).get("answer", "").lower()


