from __future__ import annotations

import os

import pytest
import httpx

# Ensure required settings exist during import-time settings validation.
os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

from src.api.main import app


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
    assert created["task"] == "hello"
    assert enqueued["task"] == "hello"


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
                    "model": "gemini-3-pro",
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
    assert body["result"]["model"] == "gemini-3-pro"
    assert body["route"] == "content"
    assert body["route_confidence"] == 0.9
    assert body["route_rationale"] == "test"


