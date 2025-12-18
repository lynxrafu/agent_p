from __future__ import annotations

import os

import pytest
import httpx

# Ensure required settings exist during import-time settings validation.
os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

from src.api.main import app
from src.core.settings import get_settings


@pytest.mark.asyncio
async def test_health_ok(monkeypatch):
    class DummyMongo:
        async def ping(self):
            return None

    class DummyConn:
        def ping(self):
            return True

    class DummyQueue:
        connection = DummyConn()

    app.state.mongo = DummyMongo()

    monkeypatch.setattr("src.api.main.get_task_queue", DummyQueue)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    assert body["mongo"]["ok"] is True
    assert body["redis"]["ok"] is True


@pytest.mark.asyncio
async def test_health_degraded_when_mongo_fails(monkeypatch):
    class DummyMongo:
        async def ping(self):
            raise RuntimeError("mongo down")

    class DummyConn:
        def ping(self):
            return True

    class DummyQueue:
        connection = DummyConn()

    app.state.mongo = DummyMongo()
    monkeypatch.setattr("src.api.main.get_task_queue", DummyQueue)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
    assert res.status_code == 503
    body = res.json()
    assert body["status"] == "degraded"
    assert body["mongo"]["ok"] is False
    assert body["redis"]["ok"] is True


@pytest.mark.asyncio
async def test_health_degraded_when_redis_fails(monkeypatch):
    class DummyMongo:
        async def ping(self):
            return None

    class DummyConn:
        def ping(self):
            raise RuntimeError("redis down")

    class DummyQueue:
        connection = DummyConn()

    app.state.mongo = DummyMongo()
    monkeypatch.setattr("src.api.main.get_task_queue", DummyQueue)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
    assert res.status_code == 503
    body = res.json()
    assert body["status"] == "degraded"
    assert body["mongo"]["ok"] is True
    assert body["redis"]["ok"] is False


@pytest.mark.asyncio
async def test_execute_empty_task_400(monkeypatch):
    class DummyMongo:
        async def create_task(self, task_id: str, task: str):
            _ = task_id
            _ = task
            return None
        async def set_task_session(self, task_id: str, session_id: str):
            _ = (task_id, session_id)
            return None

    app.state.mongo = DummyMongo()

    class DummyQueue:
        def enqueue(self, *args, **kwargs):
            raise AssertionError("enqueue should not be called for empty task")

    monkeypatch.setattr("src.api.main.get_task_queue", DummyQueue)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/v1/agent/execute", json={"task": ""})
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_execute_happy_path_returns_202(monkeypatch):
    created = {}

    class DummyMongo:
        async def create_task(self, task_id: str, task: str):
            created["task_id"] = task_id
            created["task"] = task
        async def set_task_session(self, task_id: str, session_id: str):
            created["session_id"] = session_id

    app.state.mongo = DummyMongo()

    enqueued = {}

    class DummyQueue:
        def enqueue(self, func, **kwargs):
            enqueued["func"] = func
            enqueued.update(kwargs)

    monkeypatch.setattr("src.api.main.get_task_queue", DummyQueue)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/v1/agent/execute", json={"task": "hello"})

    assert res.status_code == 202
    body = res.json()
    assert "task_id" in body
    assert body["status"] == "queued"
    assert body["session_id"] == body["task_id"]
    assert created["task"] == "hello"
    assert enqueued["task"] == "hello"

@pytest.mark.asyncio
async def test_rate_limit_returns_429_when_limiter_denies(monkeypatch):
    class DummyLimiter:
        async def allow(self, identity: str):
            _ = identity
            class R:
                allowed = False
                retry_after_s = 1.1
                limit = 60
            return R()

    # Inject a limiter without enabling Redis connections.
    app.state.rate_limiter = DummyLimiter()

    class DummyMongo:
        async def create_task(self, task_id: str, task: str):
            _ = (task_id, task)
        async def set_task_session(self, task_id: str, session_id: str):
            _ = (task_id, session_id)

    app.state.mongo = DummyMongo()

    class DummyQueue:
        def enqueue(self, *_args, **_kwargs):
            raise AssertionError("enqueue should not be called when rate limited")

    monkeypatch.setattr("src.api.main.get_task_queue", DummyQueue)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/v1/agent/execute", json={"task": "hello"})
    assert res.status_code == 429
    assert res.json()["detail"] == "Too Many Requests"


@pytest.mark.asyncio
async def test_api_key_required_for_v1_when_configured(monkeypatch):
    # Configure API key.
    os.environ["API_KEY"] = "secret"
    os.environ["API_KEY_HEADER"] = "X-API-Key"
    get_settings.cache_clear()

    app.state.rate_limiter = None

    class DummyMongo:
        async def create_task(self, task_id: str, task: str):
            _ = (task_id, task)
        async def set_task_session(self, task_id: str, session_id: str):
            _ = (task_id, session_id)

    app.state.mongo = DummyMongo()

    class DummyQueue:
        def enqueue(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr("src.api.main.get_task_queue", DummyQueue)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res1 = await client.post("/v1/agent/execute", json={"task": "hello"})
        res2 = await client.post("/v1/agent/execute", json={"task": "hello"}, headers={"X-API-Key": "secret"})

    assert res1.status_code == 401
    assert res2.status_code == 202

    # Cleanup: avoid leaking env to other tests.
    os.environ.pop("API_KEY", None)
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_get_task_404_when_missing():
    class DummyMongo:
        async def get_task(self, task_id: str):
            _ = task_id
            return None

    app.state.mongo = DummyMongo()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/v1/agent/tasks/does-not-exist")

    assert res.status_code == 404


@pytest.mark.asyncio
async def test_get_task_returns_status_and_result():
    class DummyMongo:
        async def get_task(self, task_id: str):
            return {
                "task_id": task_id,
                "status": "completed",
                "route": "content",
                "route_confidence": 0.9,
                "route_rationale": "test",
                "result": {
                    "answer": "Answer:\nHello\n\nSources:\n- https://example.com",
                    "sources": [{"title": "Example", "url": "https://example.com", "score": 0.9}],
                    "model": "gemini-3-flash-preview",
                },
            }

    app.state.mongo = DummyMongo()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/v1/agent/tasks/abc")

    assert res.status_code == 200
    body = res.json()
    assert body["task_id"] == "abc"
    assert body["status"] == "completed"
    assert body["result"]["model"] == "gemini-3-flash-preview"
    assert body["route"] == "content"
    assert body["route_confidence"] == 0.9
    assert body["route_rationale"] == "test"


@pytest.mark.asyncio
async def test_get_task_returns_failed_status_with_error():
    """Verify that tasks with model errors return proper error information."""
    class DummyMongo:
        async def get_task(self, task_id: str):
            return {
                "task_id": task_id,
                "status": "failed",
                "route": "code",
                "route_confidence": 0.8,
                "route_rationale": "keyword_match",
                "result": {
                    "answer": None,
                    "sources": [],
                    "model": "gemini-3-flash-preview",
                    "error": "LLM generation failed: API key invalid",
                    "stage": "unknown",  # Valid stage value
                },
            }

    app.state.mongo = DummyMongo()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/v1/agent/tasks/failed-task")

    assert res.status_code == 200
    body = res.json()
    assert body["task_id"] == "failed-task"
    assert body["status"] == "failed"
    assert body["result"]["error"] == "LLM generation failed: API key invalid"
    assert body["result"]["answer"] is None


@pytest.mark.asyncio
async def test_execute_whitespace_only_task_400(monkeypatch):
    """Verify that whitespace-only tasks are rejected."""
    class DummyMongo:
        async def create_task(self, task_id: str, task: str):
            raise AssertionError("create_task should not be called for whitespace task")
        async def set_task_session(self, task_id: str, session_id: str):
            raise AssertionError("set_task_session should not be called for whitespace task")

    app.state.mongo = DummyMongo()
    app.state.rate_limiter = None

    class DummyQueue:
        def enqueue(self, *args, **kwargs):
            raise AssertionError("enqueue should not be called for whitespace task")

    monkeypatch.setattr("src.api.main.get_task_queue", DummyQueue)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/v1/agent/execute", json={"task": "   \n\t  "})
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


@pytest.mark.asyncio
async def test_execute_preserves_custom_session_id(monkeypatch):
    """Verify that provided session_id is used instead of generating a new one."""
    created = {}

    class DummyMongo:
        async def create_task(self, task_id: str, task: str):
            created["task_id"] = task_id
        async def set_task_session(self, task_id: str, session_id: str):
            created["session_id"] = session_id

    app.state.mongo = DummyMongo()
    app.state.rate_limiter = None

    class DummyQueue:
        def enqueue(self, func, **kwargs):
            pass

    monkeypatch.setattr("src.api.main.get_task_queue", DummyQueue)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/agent/execute",
            json={"task": "Hello", "session_id": "my-custom-session"}
        )

    assert res.status_code == 202
    body = res.json()
    assert body["session_id"] == "my-custom-session"
    assert created["session_id"] == "my-custom-session"


@pytest.mark.asyncio
async def test_get_task_handles_unknown_status_gracefully():
    """Verify API doesn't crash if MongoDB contains an unexpected status value."""
    class DummyMongo:
        async def get_task(self, task_id: str):
            return {
                "task_id": task_id,
                "status": "weird_unknown_status",  # Unexpected status
                "route": None,
            }

    app.state.mongo = DummyMongo()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/v1/agent/tasks/strange-task")

    # Should not 500; should default to "queued"
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "queued"


@pytest.mark.asyncio
async def test_execute_with_valid_task_triggers_queue_enqueue(monkeypatch):
    """Happy path: valid task creates MongoDB record and enqueues job."""
    mongo_calls = []
    queue_calls = []

    class DummyMongo:
        async def create_task(self, task_id: str, task: str):
            mongo_calls.append(("create_task", task_id, task))
        async def set_task_session(self, task_id: str, session_id: str):
            mongo_calls.append(("set_task_session", task_id, session_id))

    app.state.mongo = DummyMongo()
    app.state.rate_limiter = None

    class DummyQueue:
        def enqueue(self, func, **kwargs):
            queue_calls.append(kwargs)

    monkeypatch.setattr("src.api.main.get_task_queue", DummyQueue)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/agent/execute",
            json={"task": "Write a Python function to sort a list"}
        )

    assert res.status_code == 202
    body = res.json()
    
    # Verify MongoDB was called
    assert len(mongo_calls) == 2
    assert mongo_calls[0][0] == "create_task"
    assert mongo_calls[0][2] == "Write a Python function to sort a list"
    
    # Verify queue was called
    assert len(queue_calls) == 1
    assert queue_calls[0]["task"] == "Write a Python function to sort a list"


