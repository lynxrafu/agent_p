from __future__ import annotations

import os

import pytest


@pytest.mark.asyncio
async def test_worker_code_route_marks_failed_and_persists_route(monkeypatch):
    os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
    os.environ.setdefault("GOOGLE_API_KEY", "x")
    os.environ.setdefault("TAVILY_API_KEY", "y")
    os.environ.setdefault("GEMINI_MODEL", "gemini-3-pro")

    from src.core import jobs as jobs_mod
    from src.models.routing_models import RoutingDecision, TaskType

    calls = []

    class DummyMongo:
        def __init__(self, mongo_url: str):
            _ = mongo_url

        async def update_task(self, task_id: str, status: str, result=None):
            calls.append(("update_task", task_id, status, result))

        async def set_task_route(self, task_id: str, route: str, route_confidence, route_rationale):
            calls.append(("set_task_route", task_id, route, route_confidence, route_rationale))

        async def close(self):
            return None

    class DummyPeer:
        async def route(self, task: str):
            _ = task
            return RoutingDecision(destination=TaskType.code, confidence=0.8, rationale="test-code")

    class DummyContent:
        async def process(self, task: str):
            _ = task
            raise AssertionError("ContentAgent should not be called for code route")

    monkeypatch.setattr(jobs_mod, "Mongo", DummyMongo)
    monkeypatch.setattr(jobs_mod, "PeerAgent", lambda *_args, **_kwargs: DummyPeer())
    monkeypatch.setattr(jobs_mod, "ContentAgent", lambda *_args, **_kwargs: DummyContent())

    await jobs_mod._process_task(task_id="t1", task="write code", mongo_url="mongodb://x")

    assert ("set_task_route", "t1", "code", 0.8, "test-code") in calls
    # Final state should be failed with a clear error.
    final = [c for c in calls if c[0] == "update_task"][-1]
    assert final[2] == "failed"
    assert "CodeAgent" in (final[3] or {}).get("error", "")


@pytest.mark.asyncio
async def test_worker_content_route_calls_content_agent(monkeypatch):
    os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
    os.environ.setdefault("GOOGLE_API_KEY", "x")
    os.environ.setdefault("TAVILY_API_KEY", "y")
    os.environ.setdefault("GEMINI_MODEL", "gemini-3-pro")

    from src.core import jobs as jobs_mod
    from src.models.routing_models import RoutingDecision, TaskType

    class DummyMongo:
        def __init__(self, mongo_url: str):
            _ = mongo_url

        async def update_task(self, task_id: str, status: str, result=None):
            _ = (task_id, status, result)

        async def set_task_route(self, task_id: str, route: str, route_confidence, route_rationale):
            _ = (task_id, route, route_confidence, route_rationale)

        async def close(self):
            return None

    class DummyPeer:
        async def route(self, task: str):
            _ = task
            return RoutingDecision(destination=TaskType.content, confidence=0.9, rationale="test-content")

    called = {"content": 0}

    class DummyContent:
        async def process(self, task: str):
            _ = task
            called["content"] += 1

            class Out:
                answer = "Answer:\nX\n\nSources:\n- https://example.com"
                sources = [{"url": "https://example.com"}]
                model = "gemini-3-pro"

            return Out()

    monkeypatch.setattr(jobs_mod, "Mongo", DummyMongo)
    monkeypatch.setattr(jobs_mod, "PeerAgent", lambda *_args, **_kwargs: DummyPeer())
    monkeypatch.setattr(jobs_mod, "ContentAgent", lambda *_args, **_kwargs: DummyContent())

    await jobs_mod._process_task(task_id="t2", task="what is x", mongo_url="mongodb://x")
    assert called["content"] == 1


