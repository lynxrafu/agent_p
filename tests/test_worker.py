"""Worker integration tests - verifies task processing flow with mocked agents.

Tests the complete flow from task reception to result persistence:
- Routing to correct agents
- Error handling and graceful failures
- Agent log persistence for observability
"""
from __future__ import annotations

import os

import pytest


# Ensure required settings exist for get_settings()
@pytest.fixture(autouse=True)
def setup_env():
    os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
    os.environ.setdefault("GOOGLE_API_KEY", "x")
    os.environ.setdefault("TAVILY_API_KEY", "y")
    os.environ.setdefault("GEMINI_MODEL", "gemini-3-flash-preview")


# =============================================================================
# Happy Path Tests
# =============================================================================

@pytest.mark.asyncio
async def test_worker_processes_code_task_successfully(monkeypatch):
    """Happy path: code task is routed to CodeAgent and completes successfully."""
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
        async def route(self, input_data):
            _ = input_data
            return RoutingDecision(destination=TaskType.code, confidence=0.95, rationale="code_keyword_match")

    class DummyCode:
        async def process(self, input_data):
            _ = input_data
            return "```python\ndef sort_list(arr):\n    return sorted(arr)\n```\n\nSorts a list in ascending order."

    monkeypatch.setattr(jobs_mod, "Mongo", DummyMongo)
    monkeypatch.setattr(jobs_mod, "PeerAgent", lambda *_args, **_kwargs: DummyPeer())
    monkeypatch.setattr(jobs_mod, "CodeAgent", lambda *_args, **_kwargs: DummyCode())

    await jobs_mod._process_task(task_id="t1", task="write python code to sort a list", mongo_url="mongodb://x")

    # Verify routing was recorded
    assert ("set_task_route", "t1", "code", 0.95, "code_keyword_match") in calls

    # Verify final state is completed with code output
    final = [c for c in calls if c[0] == "update_task"][-1]
    assert final[2] == "completed"
    assert "```python" in (final[3] or {}).get("answer", "")


@pytest.mark.asyncio
async def test_worker_processes_content_task_successfully(monkeypatch):
    """Happy path: content task is routed to ContentAgent and completes successfully."""
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
        async def route(self, task):
            _ = task
            return RoutingDecision(destination=TaskType.content, confidence=0.9, rationale="info_request")

    called = {"content": 0}

    class DummyContent:
        async def process(self, task):
            _ = task
            called["content"] += 1

            class Out:
                answer = "Python is a high-level programming language.\n\n## References\n[1] Python.org - https://python.org"
                sources = [{"url": "https://python.org", "title": "Python.org", "score": 0.9}]
                model = "gemini-3-flash-preview"

            return Out()

    monkeypatch.setattr(jobs_mod, "Mongo", DummyMongo)
    monkeypatch.setattr(jobs_mod, "PeerAgent", lambda *_args, **_kwargs: DummyPeer())
    monkeypatch.setattr(jobs_mod, "ContentAgent", lambda *_args, **_kwargs: DummyContent())

    await jobs_mod._process_task(task_id="t2", task="What is Python?", mongo_url="mongodb://x")
    assert called["content"] == 1


@pytest.mark.asyncio
async def test_worker_processes_business_discovery_task_successfully(monkeypatch):
    """Happy path: business problem is routed to BusinessDiscoveryAgent."""
    from src.core import jobs as jobs_mod
    from src.models.routing_models import RoutingDecision, TaskType
    import asyncio

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
            return RoutingDecision(destination=TaskType.business_discovery, confidence=0.9, rationale="business_problem")

    class DummyBiz:
        async def process(self, input_data):
            assert getattr(input_data, "session_id", None) == "s1"
            return "What is the main problem and when did it start?"

    monkeypatch.setattr(jobs_mod, "Mongo", DummyMongo)
    monkeypatch.setattr(jobs_mod, "PeerAgent", lambda *_args, **_kwargs: DummyPeer())
    monkeypatch.setattr(jobs_mod, "BusinessDiscoveryAgent", lambda *_args, **_kwargs: DummyBiz())

    await asyncio.to_thread(jobs_mod.process_task_job, task_id="t1", task="Sales are down", mongo_url="mongodb://x", session_id="s1")

    assert updates[0]["status"] == "processing"
    assert updates[-1]["status"] == "completed"
    assert "problem" in (updates[-1]["result"] or {}).get("answer", "").lower()


@pytest.mark.asyncio
async def test_worker_processes_diagnosis_task_successfully(monkeypatch):
    """Happy path: diagnosis request is routed to DiagnosisAgent."""
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
            return RoutingDecision(destination=TaskType.diagnosis, confidence=0.9, rationale="problem_structuring")

    class DummyDiagOut:
        def model_dump(self):
            return {"problem_type": "growth", "main_problem": "Sales decline", "tree": {"label": "Sales", "children": []}}

    class DummyDiag:
        async def process(self, _input_data):
            return DummyDiagOut()

        @staticmethod
        def render_markdown(_out):
            return "## Problem Type: **Growth**\n\n### Problem Tree (Issue Tree)\n- Sales decline"

    monkeypatch.setattr(jobs_mod, "Mongo", DummyMongo)
    monkeypatch.setattr(jobs_mod, "PeerAgent", lambda *_args, **_kwargs: DummyPeer())
    monkeypatch.setattr(jobs_mod, "DiagnosisAgent", lambda *_args, **_kwargs: DummyDiag())

    await jobs_mod._process_task(task_id="t1", task="Create problem tree", mongo_url="mongodb://x", session_id="s1")

    assert updates[-1]["status"] == "completed"
    result = updates[-1]["result"] or {}
    assert "Problem Tree" in (result.get("answer") or "")
    assert "diagnosis" in (result.get("debug") or {})


# =============================================================================
# Error Handling Tests
# =============================================================================

@pytest.mark.asyncio
async def test_worker_handles_agent_error_gracefully(monkeypatch):
    """Test that unexpected agent errors result in failed status without crashing."""
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
            raise RuntimeError("LLM service unavailable")

    monkeypatch.setattr(jobs_mod, "Mongo", DummyMongo)
    monkeypatch.setattr(jobs_mod, "PeerAgent", lambda *_args, **_kwargs: DummyPeer())
    monkeypatch.setattr(jobs_mod, "ContentAgent", lambda *_args, **_kwargs: DummyContent())

    # Should not raise - should fail gracefully
    await jobs_mod._process_task(task_id="t1", task="question", mongo_url="mongodb://x")

    assert updates[0]["status"] == "processing"
    assert updates[-1]["status"] == "failed"
    assert updates[-1]["result"]["stage"] == "unknown"


@pytest.mark.asyncio
async def test_worker_handles_search_error_with_proper_stage(monkeypatch):
    """Test that search errors are recorded with correct stage."""
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
            _ = (task_id, route, route_confidence, route_rationale)

        async def close(self):
            return None

    class DummyAgent:
        async def process(self, task):
            _ = task
            raise ContentSearchError("Tavily returned 0 results")

    class DummyPeer:
        async def route(self, task):
            _ = task
            return RoutingDecision(destination=TaskType.content, confidence=0.9, rationale="test")

    monkeypatch.setattr(jobs_mod, "Mongo", DummyMongo)
    monkeypatch.setattr(jobs_mod, "ContentAgent", lambda *_args, **_kwargs: DummyAgent())
    monkeypatch.setattr(jobs_mod, "PeerAgent", lambda *_args, **_kwargs: DummyPeer())

    await jobs_mod._process_task(task_id="t1", task="obscure topic", mongo_url="mongodb://x")

    assert updates[0]["status"] == "processing"
    assert updates[-1]["status"] == "failed"
    assert updates[-1]["result"]["stage"] == "search"


# =============================================================================
# Observability Tests
# =============================================================================

@pytest.mark.asyncio
async def test_worker_writes_agent_logs_for_observability(monkeypatch):
    """Test that agent interactions are logged to MongoDB for observability."""
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
                "model": "gemini-3-flash-preview",
                "prompt": "ROUTING_PROMPT",
                "raw_output": '{"destination":"code"}',
                "parsed_output": {"destination": "code"},
                "latency_ms": 15.0,
            }

        async def route(self, _input_data):
            return RoutingDecision(destination=TaskType.code, confidence=0.9, rationale="test")

    class DummyCode:
        def __init__(self):
            self.last_trace = {
                "stage": "generation",
                "model": "gemini-3-flash-preview",
                "prompt": "CODE_PROMPT",
                "raw_output": "```python\nprint(1)\n```",
                "parsed_output": "code_output",
                "latency_ms": 25.0,
            }

        async def process(self, _input_data):
            return "```python\nprint(1)\n```"

    monkeypatch.setattr(jobs_mod, "Mongo", DummyMongo)
    monkeypatch.setattr(jobs_mod, "PeerAgent", lambda *_args, **_kwargs: DummyPeer())
    monkeypatch.setattr(jobs_mod, "CodeAgent", lambda *_args, **_kwargs: DummyCode())

    await jobs_mod._process_task(task_id="t1", task="write code", mongo_url="mongodb://x", session_id="s1")

    # Verify both agents were logged
    agents = {e["agent"] for e in agent_logs}
    assert "peer_agent" in agents
    assert "code_agent" in agents

    # Verify log entries have correct metadata
    for entry in agent_logs:
        assert entry["task_id"] == "t1"
        assert entry["session_id"] == "s1"

