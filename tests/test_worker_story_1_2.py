from __future__ import annotations

import os

import pytest


@pytest.mark.asyncio
async def test_worker_marks_failed_with_stage_search_on_zero_results(monkeypatch):
    # Ensure required settings exist for get_settings()
    os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
    os.environ.setdefault("GOOGLE_API_KEY", "x")
    os.environ.setdefault("TAVILY_API_KEY", "y")
    os.environ.setdefault("GEMINI_MODEL", "gemini-3-pro")

    from src.core import jobs as jobs_mod
    from src.agents.content_agent import ContentSearchError
    from src.models.routing_models import RoutingDecision, TaskType

    updates = []

    class DummyMongo:
        def __init__(self, mongo_url: str):
            _ = mongo_url

        async def update_task(self, task_id: str, status: str, result=None):
            updates.append({"task_id": task_id, "status": status, "result": result})

        async def set_task_route(self, task_id: str, route: str, route_confidence, route_rationale):
            updates.append(
                {
                    "task_id": task_id,
                    "status": "route_set",
                    "result": {"route": route, "route_confidence": route_confidence, "route_rationale": route_rationale},
                }
            )

        async def close(self):
            return None

    class DummyAgent:
        async def process(self, task: str):
            _ = task
            raise ContentSearchError("Tavily returned 0 results")

    class DummyPeer:
        async def route(self, task: str):
            _ = task
            return RoutingDecision(destination=TaskType.content, confidence=0.9, rationale="test")

    monkeypatch.setattr(jobs_mod, "Mongo", DummyMongo)
    monkeypatch.setattr(jobs_mod, "ContentAgent", lambda *_args, **_kwargs: DummyAgent())
    monkeypatch.setattr(jobs_mod, "PeerAgent", lambda *_args, **_kwargs: DummyPeer())

    await jobs_mod._process_task(task_id="t1", task="q", mongo_url="mongodb://x")

    assert updates[0]["status"] == "processing"
    assert updates[-1]["status"] == "failed"
    assert updates[-1]["result"]["stage"] == "search"


