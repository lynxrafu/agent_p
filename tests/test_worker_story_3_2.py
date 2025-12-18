from __future__ import annotations

import os

import pytest


@pytest.mark.asyncio
async def test_worker_diagnosis_route_completes_and_persists_structured_debug(monkeypatch):
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
            return RoutingDecision(destination=TaskType.diagnosis, confidence=0.9, rationale="test")

    class DummyDiagOut:
        def model_dump(self):
            return {"problem_type": "unknown", "main_problem": "x", "tree": {"label": "x", "children": []}}

    class DummyDiag:
        async def process(self, _input_data):
            return DummyDiagOut()

        @staticmethod
        def render_markdown(_out):
            return "Problem type: **unknown**\n\n**Problem tree**\n- x"

    monkeypatch.setattr(jobs_mod, "Mongo", DummyMongo)
    monkeypatch.setattr(jobs_mod, "PeerAgent", lambda *_args, **_kwargs: DummyPeer())
    monkeypatch.setattr(jobs_mod, "DiagnosisAgent", lambda *_args, **_kwargs: DummyDiag())

    await jobs_mod._process_task(task_id="t1", task="structure this", mongo_url="mongodb://x", session_id="s1")  # noqa: SLF001

    assert updates[-1]["status"] == "completed"
    result = updates[-1]["result"] or {}
    assert "Problem tree" in (result.get("answer") or "")
    assert "diagnosis" in (result.get("debug") or {})


