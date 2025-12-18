"""FastAPI application entrypoint and HTTP endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import structlog
import redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pymongo.errors import PyMongoError
from pydantic import BaseModel, Field

from src.core.logging import configure_logging
from src.core.queue import get_task_queue
from src.core.rate_limiter import RateLimiter, RateLimiterConfig
from src.core.settings import get_settings
from src.core.jobs import process_task_job
from src.db.mongo import Mongo
from src.models.task_models import TaskReadResponse, TaskResult, TaskStatus

log = structlog.get_logger(__name__)


class ExecuteRequest(BaseModel):
    task: str = Field(...)
    session_id: str | None = None


class ExecuteResponse(BaseModel):
    task_id: str
    status: str
    session_id: str | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    """FastAPI lifespan hook: configure logging and establish Mongo connection."""
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    application.state.mongo = Mongo(settings.MONGODB_URL)
    application.state.rate_limiter = None
    if settings.RATE_LIMIT_ENABLED:
        application.state.rate_limiter = RateLimiter(
            RateLimiterConfig(
                redis_url=settings.REDIS_URL,
                requests_per_min=settings.RATE_LIMIT_REQUESTS_PER_MIN,
                burst=settings.RATE_LIMIT_BURST,
            )
        )
    await application.state.mongo.ping()
    log.info("api_startup_complete")
    yield
    if getattr(application.state, "rate_limiter", None) is not None:
        await application.state.rate_limiter.close()
    await application.state.mongo.close()
    log.info("api_shutdown_complete")


app = FastAPI(title="Peer Agent System API", version="1.0.0", lifespan=lifespan)


def _client_identity(request: Request, api_key_header: str) -> str:
    api_key = request.headers.get(api_key_header)
    if api_key:
        return f"key:{api_key}"
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return f"ip:{xff.split(',')[0].strip()}"
    host = getattr(getattr(request, "client", None), "host", None)
    return f"ip:{host or 'unknown'}"


@app.middleware("http")
async def security_and_rate_limit(request: Request, call_next):
    # Allow unauthenticated/unlimited health and docs endpoints.
    if request.url.path in {"/health", "/docs", "/openapi.json", "/redoc"}:
        return await call_next(request)

    settings = get_settings()

    # Optional API key enforcement (Story 4.2).
    if settings.API_KEY and request.url.path.startswith("/v1/"):
        provided = request.headers.get(settings.API_KEY_HEADER)
        if provided != settings.API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    # Optional rate limiting (Story 4.2).
    limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is not None:
        identity = _client_identity(request, settings.API_KEY_HEADER)
        result = await limiter.allow(identity)
        if not result.allowed:
            headers = {
                "Retry-After": str(int(result.retry_after_s) + 1),
                "X-RateLimit-Limit": str(result.limit),
                "X-RateLimit-Remaining": "0",
            }
            return JSONResponse(status_code=429, content={"detail": "Too Many Requests"}, headers=headers)

    return await call_next(request)


@app.post("/v1/agent/execute", status_code=202, response_model=ExecuteResponse)
async def execute_agent(request: ExecuteRequest) -> ExecuteResponse:
    """Create a task record and enqueue it for worker processing."""
    settings = get_settings()
    task = request.task.strip()
    if not task:
        raise HTTPException(status_code=400, detail="Task content cannot be empty")

    task_id = str(uuid4())
    session_id = request.session_id or task_id

    # Persist task (MongoDB preferred logging/persistence per CLAUDE.md)
    mongo: Mongo = app.state.mongo
    await mongo.create_task(task_id=task_id, task=task)
    await mongo.set_task_session(task_id=task_id, session_id=session_id)

    # Enqueue task for worker processing
    queue = get_task_queue()
    queue.enqueue(
        process_task_job,
        task_id=task_id,
        task=task,
        mongo_url=settings.MONGODB_URL,
        log_level=settings.LOG_LEVEL,
        session_id=session_id,
    )

    return ExecuteResponse(task_id=task_id, status="queued", session_id=session_id)


@app.get("/v1/agent/tasks/{task_id}", response_model=TaskReadResponse)
async def get_task(task_id: str) -> TaskReadResponse:
    """Fetch a task and its result (if present) from MongoDB."""
    mongo: Mongo = app.state.mongo
    doc = await mongo.get_task(task_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Task not found")

    raw_status = doc.get("status") or "queued"
    try:
        status = TaskStatus(raw_status)
    except ValueError:
        # Defensive: don't 500 if DB contains unexpected status.
        status = TaskStatus.queued
    raw_result = doc.get("result")
    result = TaskResult.model_validate(raw_result) if isinstance(raw_result, dict) else None
    return TaskReadResponse(
        task_id=task_id,
        status=status,
        result=result,
        route=doc.get("route"),
        route_confidence=doc.get("route_confidence"),
        route_rationale=doc.get("route_rationale"),
        session_id=doc.get("session_id"),
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check: verifies Mongo ping and Redis connectivity."""
    import asyncio

    mongo: Mongo = app.state.mongo
    mongo_ok = True
    mongo_error: str | None = None
    try:
        await mongo.ping()
    except (PyMongoError, TimeoutError, OSError, ConnectionError, RuntimeError) as e:
        mongo_ok = False
        mongo_error = str(e)

    q = get_task_queue()
    redis_ok = True
    redis_error: str | None = None
    try:
        # Avoid blocking the event loop with a sync ping.
        await asyncio.to_thread(q.connection.ping)
    except (redis.exceptions.RedisError, TimeoutError, OSError, ConnectionError, RuntimeError) as e:
        redis_ok = False
        redis_error = str(e)

    overall = "healthy" if mongo_ok and redis_ok else "degraded"
    payload: dict[str, Any] = {
        "status": overall,
        "mongo": {"ok": mongo_ok, "error": mongo_error},
        "redis": {"ok": redis_ok, "error": redis_error},
    }
    status_code = 200 if overall == "healthy" else 503
    return JSONResponse(status_code=status_code, content=payload)


