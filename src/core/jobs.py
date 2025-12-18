from __future__ import annotations

import asyncio

import structlog

from src.core.logging import configure_logging
from src.core.settings import get_settings
from src.agents.content_agent import ContentAgent, ContentAgentConfig, ContentAgentError
from src.agents.peer_agent import PeerAgent
from src.db.mongo import Mongo
from src.models.task_models import TaskResult
from src.models.routing_models import TaskType

log = structlog.get_logger(__name__)


def process_task_job(task_id: str, task: str, mongo_url: str, log_level: str = "INFO") -> None:
    """RQ worker entrypoint (sync function)."""

    # Ensure worker logging is structured JSON (story/CLAUDE alignment).
    configure_logging(log_level)
    asyncio.run(_process_task(task_id=task_id, task=task, mongo_url=mongo_url))


async def _process_task(task_id: str, task: str, mongo_url: str) -> None:
    db = Mongo(mongo_url)
    try:
        settings = get_settings()

        log.info("worker_received_task", task_id=task_id)
        await db.update_task(task_id, status="processing")

        # Story 1.3: route-first execution via PeerAgent (LLM + keyword fallback).
        peer = PeerAgent(settings)
        routing = await peer.route(task)

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
            output = await agent.process(task)
            result = TaskResult(answer=output.answer, sources=output.sources, model=output.model, debug=debug).model_dump()
            await db.update_task(task_id, status="completed", result=result)
            log.info("worker_completed_task", task_id=task_id, route=routing.destination.value)
            return

        if routing.destination == TaskType.code:
            result = TaskResult(
                error="Agent not implemented yet: CodeAgent",
                stage="unknown",
                model=settings.GEMINI_MODEL,
                debug=debug,
            ).model_dump()
            await db.update_task(task_id, status="failed", result=result)
            log.info("worker_task_failed_unimplemented_agent", task_id=task_id, route=routing.destination.value)
            return

        if routing.destination == TaskType.business_discovery:
            result = TaskResult(
                error="Agent not implemented yet: BusinessDiscoveryAgent",
                stage="unknown",
                model=settings.GEMINI_MODEL,
                debug=debug,
            ).model_dump()
            await db.update_task(task_id, status="failed", result=result)
            log.info("worker_task_failed_unimplemented_agent", task_id=task_id, route=routing.destination.value)
            return

        # Any remaining case defaults to content (AC2).
        agent = ContentAgent(
            ContentAgentConfig(
                google_api_key=settings.GOOGLE_API_KEY or "",
                tavily_api_key=settings.TAVILY_API_KEY or "",
                model=settings.GEMINI_MODEL,
            )
        )
        output = await agent.process(task)
        result = TaskResult(answer=output.answer, sources=output.sources, model=output.model, debug=debug).model_dump()
        await db.update_task(task_id, status="completed", result=result)
        log.info("worker_completed_task", task_id=task_id, route="content_default")
    except ContentAgentError as e:
        log.error("worker_failed_task", task_id=task_id, error=str(e), stage=getattr(e, "stage", "unknown"), exc_info=True)
        result = TaskResult(error=str(e), stage=getattr(e, "stage", "unknown"), model=settings.GEMINI_MODEL).model_dump()
        await db.update_task(task_id, status="failed", result=result)
    except Exception as e:
        log.error("worker_failed_task", task_id=task_id, error=str(e), exc_info=True)
        result = TaskResult(error=str(e), stage="unknown").model_dump()
        await db.update_task(task_id, status="failed", result=result)
    finally:
        await db.close()


