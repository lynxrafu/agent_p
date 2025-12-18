from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.core.logging import configure_logging
from src.core.queue import get_task_queue
from src.core.settings import get_settings
from src.core.jobs import process_task_job
from src.db.mongo import Mongo
from src.models.task_models import TaskReadResponse, TaskResult, TaskStatus

log = structlog.get_logger(__name__)


class ExecuteRequest(BaseModel):
    task: str = Field(...)


class ExecuteResponse(BaseModel):
    task_id: str
    status: str


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    application.state.mongo = Mongo(settings.MONGODB_URL)
    await application.state.mongo.ping()
    log.info("api_startup_complete")
    yield
    await application.state.mongo.close()
    log.info("api_shutdown_complete")


app = FastAPI(title="Peer Agent System API", version="1.0.0", lifespan=lifespan)


@app.post("/v1/agent/execute", status_code=202, response_model=ExecuteResponse)
async def execute_agent(request: ExecuteRequest) -> ExecuteResponse:
    settings = get_settings()
    task = request.task.strip()
    if not task:
        raise HTTPException(status_code=400, detail="Task content cannot be empty")

    task_id = str(uuid4())

    # Persist task (MongoDB preferred logging/persistence per CLAUDE.md)
    mongo: Mongo = app.state.mongo
    await mongo.create_task(task_id=task_id, task=task)

    # Enqueue task for worker processing
    queue = get_task_queue()
    queue.enqueue(process_task_job, task_id=task_id, task=task, mongo_url=settings.MONGODB_URL, log_level=settings.LOG_LEVEL)

    return ExecuteResponse(task_id=task_id, status="queued")


@app.get("/v1/agent/tasks/{task_id}", response_model=TaskReadResponse)
async def get_task(task_id: str) -> TaskReadResponse:
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
    return TaskReadResponse(task_id=task_id, status=status, result=result)


@app.get("/health")
async def health() -> dict[str, Any]:
    # Mongo
    mongo: Mongo = app.state.mongo
    await mongo.ping()

    # Redis (queue backend)
    q = get_task_queue()
    # Avoid blocking the event loop with a sync ping.
    import asyncio

    await asyncio.to_thread(q.connection.ping)

    return {"status": "healthy"}


