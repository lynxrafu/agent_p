"""Worker job execution: route tasks and persist results."""

from __future__ import annotations

import asyncio
from contextlib import suppress

import structlog
from pydantic import ValidationError
from pymongo.errors import PyMongoError

from src.core.logging import configure_logging
from src.core.settings import get_settings
from src.agents.code_agent import CodeAgent, CodeAgentConfig
from src.agents.business_discovery_agent import BusinessDiscoveryAgent, BusinessDiscoveryAgentConfig
from src.agents.content_agent import ContentAgent, ContentAgentConfig, ContentAgentError
from src.agents.peer_agent import PeerAgent
from src.db.mongo import Mongo
from src.models.task_models import TaskResult
from src.models.routing_models import TaskType
from src.models.task_input import TaskInput

log = structlog.get_logger(__name__)


def process_task_job(task_id: str, task: str, mongo_url: str, log_level: str = "INFO", session_id: str | None = None) -> None:
    """RQ worker entrypoint (sync function)."""

    # Ensure worker logging is structured JSON (story/CLAUDE alignment).
    configure_logging(log_level)
    asyncio.run(_process_task(task_id=task_id, task=task, mongo_url=mongo_url, session_id=session_id))


async def _process_task(task_id: str, task: str, mongo_url: str, session_id: str | None = None) -> None:
    """Process a single task: persist status, route, and final result."""
    db = Mongo(mongo_url)
    settings = None
    try:
        settings = get_settings()

        log.info("worker_received_task", task_id=task_id)
        await db.update_task(task_id, status="processing")

        # Story 1.3: route-first execution via PeerAgent (LLM + keyword fallback).
        peer = PeerAgent(settings)
        task_input = TaskInput(task=task, session_id=session_id)
        routing = await peer.route(task_input)

        await db.set_task_route(
            task_id=task_id,
            route=routing.destination.value,
            route_confidence=routing.confidence,
            route_rationale=routing.rationale,
        )

        debug = {
            "route": routing.destination.value,
            "route_confidence": routing.confidence,
            "route_rationale": routing.rationale,
        }

        if routing.destination == TaskType.content:
            agent = ContentAgent(
                ContentAgentConfig(
                    google_api_key=settings.GOOGLE_API_KEY or "",
                    tavily_api_key=settings.TAVILY_API_KEY or "",
                    model=settings.GEMINI_MODEL,
                )
            )
            output = await agent.process(task_input)
            result = TaskResult(answer=output.answer, sources=output.sources, model=output.model, debug=debug).model_dump()
            await db.update_task(task_id, status="completed", result=result)
            log.info("worker_completed_task", task_id=task_id, route=routing.destination.value)
            return

        if routing.destination == TaskType.code:
            agent = CodeAgent(
                CodeAgentConfig(
                    google_api_key=settings.GOOGLE_API_KEY or "",
                    model=settings.GEMINI_MODEL,
                )
            )
            answer = await agent.process(task_input)
            result = TaskResult(answer=answer, model=settings.GEMINI_MODEL, debug=debug).model_dump()
            await db.update_task(task_id, status="completed", result=result)
            log.info("worker_completed_task", task_id=task_id, route=routing.destination.value)
            return

        if routing.destination == TaskType.business_discovery:
            agent = BusinessDiscoveryAgent(
                BusinessDiscoveryAgentConfig(
                    google_api_key=settings.GOOGLE_API_KEY or "",
                    model=settings.GEMINI_MODEL,
                    mongo_url=mongo_url,
                )
            )
            question = await agent.process(task_input)
            result = TaskResult(answer=question, model=settings.GEMINI_MODEL, debug=debug).model_dump()
            await db.update_task(task_id, status="completed", result=result)
            log.info("worker_completed_task", task_id=task_id, route=routing.destination.value)
            return

        # Any remaining case defaults to content (AC2).
        agent = ContentAgent(
            ContentAgentConfig(
                google_api_key=settings.GOOGLE_API_KEY or "",
                tavily_api_key=settings.TAVILY_API_KEY or "",
                model=settings.GEMINI_MODEL,
            )
        )
        output = await agent.process(task_input)
        result = TaskResult(answer=output.answer, sources=output.sources, model=output.model, debug=debug).model_dump()
        await db.update_task(task_id, status="completed", result=result)
        log.info("worker_completed_task", task_id=task_id, route="content_default")
    except ContentAgentError as e:
        log.error("worker_failed_task", task_id=task_id, error=str(e), stage=getattr(e, "stage", "unknown"), exc_info=True)
        result = TaskResult(error=str(e), stage=getattr(e, "stage", "unknown"), model=settings.GEMINI_MODEL).model_dump()
        await db.update_task(task_id, status="failed", result=result)
    except (ValidationError, PyMongoError, ConnectionError, TimeoutError, OSError, RuntimeError, ValueError, TypeError) as e:
        log.error("worker_failed_task", task_id=task_id, error=str(e), exc_info=True)
        model = getattr(settings, "GEMINI_MODEL", None)
        result = TaskResult(error=str(e), stage="unknown", model=model).model_dump()
        with suppress(PyMongoError, ConnectionError, TimeoutError, OSError, RuntimeError):
            await db.update_task(task_id, status="failed", result=result)
    finally:
        with suppress(PyMongoError, ConnectionError, TimeoutError, OSError, RuntimeError):
            await db.close()


