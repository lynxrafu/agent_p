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
from src.agents.diagnosis_agent import DiagnosisAgent, DiagnosisAgentConfig
from src.agents.content_agent import ContentAgent, ContentAgentConfig, ContentAgentError
from src.agents.peer_agent import PeerAgent

# Import LangChain exceptions for proper error handling
try:
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
except ImportError:
    ChatGoogleGenerativeAIError = RuntimeError  # Fallback if import fails
from src.db.mongo import Mongo
from src.models.agent_logs import AgentLogEntry
from src.models.task_models import TaskResult
from src.models.routing_models import TaskType
from src.models.task_input import TaskInput

log = structlog.get_logger(__name__)


# =============================================================================
# LLM Provider Quota/Rate Limit Detection
# =============================================================================
# This handles errors from LLM providers (Gemini, OpenAI, etc.) when their
# API quota is exceeded. This is DIFFERENT from our own API rate limiting
# (handled in src/core/rate_limiter.py and src/api/main.py).
#
# LLM Quota Error: External provider limit (e.g., Gemini free tier: 20 req/day)
# API Rate Limit:  Our own rate limiting (e.g., 60 req/min per client)
# =============================================================================

LLM_QUOTA_ERROR_EN = (
    "LLM Quota Exceeded: The AI model service (Gemini) daily quota has been exhausted. "
    "Free tier limit: 20 requests/day. Please wait 24 hours for quota reset, "
    "use a different API key, or upgrade to a paid plan."
)
LLM_QUOTA_ERROR_TR = (
    "LLM Kotası Aşıldı: Yapay zeka model servisi (Gemini) günlük kotası doldu. "
    "Ücretsiz limit: 20 istek/gün. Kotanın sıfırlanması için 24 saat bekleyin, "
    "farklı bir API anahtarı kullanın veya ücretli plana geçin."
)

# Indicators of LLM provider quota/rate limit errors
_LLM_QUOTA_INDICATORS = frozenset([
    "resource_exhausted",
    "quota exceeded",
    "quota",
    "rate limit exceeded",
    "too many requests",
    "429",
])


def _is_llm_quota_error(error: Exception) -> bool:
    """Check if an exception is an LLM provider quota/rate limit error.
    
    Detects errors from:
    - Google Gemini API (RESOURCE_EXHAUSTED, quota exceeded)
    - OpenAI API (rate_limit_exceeded, 429)
    - Other LLM providers with similar patterns
    
    Note: This is separate from our API rate limiting in src/core/rate_limiter.py
    """
    error_str = str(error).lower()
    return any(indicator in error_str for indicator in _LLM_QUOTA_INDICATORS)


def _get_llm_quota_message(task: str) -> str:
    """Return user-friendly LLM quota error message.
    
    Language detection: Turkish characters → Turkish message.
    """
    turkish_chars = set("çğıöşüÇĞİÖŞÜ")
    if any(c in task for c in turkish_chars):
        return LLM_QUOTA_ERROR_TR
    return LLM_QUOTA_ERROR_EN


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
        peer = PeerAgent(settings, session_store=db)
        task_input = TaskInput(task=task, session_id=session_id)
        routing = await peer.route(task_input)

        # System-level session continuity: remember which agent is currently active.
        if session_id:
            with suppress(Exception):
                await db.set_active_agent(session_id, routing.destination.value)

        # Deep observability: log routing interaction (best-effort).
        if hasattr(db, "create_agent_log"):
            trace = getattr(peer, "last_trace", None) or {}
            with suppress(Exception):
                await db.create_agent_log(
                    AgentLogEntry(
                        task_id=task_id,
                        session_id=session_id,
                        agent="peer_agent",
                        stage=trace.get("stage") or "routing",
                        model=trace.get("model") or getattr(settings, "GEMINI_MODEL", None),
                        prompt=trace.get("prompt"),
                        raw_output=trace.get("raw_output"),
                        parsed_output=trace.get("parsed_output") or routing.model_dump(),
                        latency_ms=trace.get("latency_ms"),
                    )
                )

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
                ),
                mongo=db,
            )
            output = await agent.process(task_input)
            if hasattr(db, "create_agent_log"):
                trace = getattr(agent, "last_trace", None) or {}
                with suppress(Exception):
                    await db.create_agent_log(
                        AgentLogEntry(
                            task_id=task_id,
                            session_id=session_id,
                            agent="content_agent",
                            stage=trace.get("stage") or "main",
                            model=trace.get("model") or output.model,
                            prompt=trace.get("prompt"),
                            raw_output=trace.get("raw_output"),
                            parsed_output=trace.get("parsed_output") or output.model_dump(),
                            latency_ms=trace.get("latency_ms"),
                        )
                    )
            result = TaskResult(answer=output.answer, sources=output.sources, model=output.model, debug=debug).model_dump()
            await db.update_task(task_id, status="completed", result=result)
            log.info("worker_completed_task", task_id=task_id, route=routing.destination.value)
            return

        if routing.destination == TaskType.code:
            agent = CodeAgent(
                CodeAgentConfig(
                    google_api_key=settings.GOOGLE_API_KEY or "",
                    model=settings.GEMINI_MODEL,
                    tavily_api_key=settings.TAVILY_API_KEY,  # Research-first: web search for latest docs
                ),
                mongo=db,  # Mongo-backed session history for persistence across workers
            )
            answer = await agent.process(task_input)
            if hasattr(db, "create_agent_log"):
                trace = getattr(agent, "last_trace", None) or {}
                with suppress(Exception):
                    await db.create_agent_log(
                        AgentLogEntry(
                            task_id=task_id,
                            session_id=session_id,
                            agent="code_agent",
                            stage=trace.get("stage") or "main",
                            model=trace.get("model") or getattr(settings, "GEMINI_MODEL", None),
                            prompt=trace.get("prompt"),
                            raw_output=trace.get("raw_output"),
                            parsed_output=trace.get("parsed_output") or answer,
                            latency_ms=trace.get("latency_ms"),
                        )
                    )
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
            # If the discovery interview completed (final analysis), stop sticking the session.
            if session_id and "## Business Discovery Analysis" in question:
                with suppress(Exception):
                    await db.set_active_agent(session_id, None)
            if hasattr(db, "create_agent_log"):
                trace = getattr(agent, "last_trace", None) or {}
                with suppress(Exception):
                    await db.create_agent_log(
                        AgentLogEntry(
                            task_id=task_id,
                            session_id=session_id,
                            agent="business_discovery_agent",
                            stage=trace.get("stage") or "main",
                            model=trace.get("model") or getattr(settings, "GEMINI_MODEL", None),
                            prompt=trace.get("prompt"),
                            raw_output=trace.get("raw_output"),
                            parsed_output=trace.get("parsed_output") or question,
                            latency_ms=trace.get("latency_ms"),
                        )
                    )
            result = TaskResult(answer=question, model=settings.GEMINI_MODEL, debug=debug).model_dump()
            await db.update_task(task_id, status="completed", result=result)
            log.info("worker_completed_task", task_id=task_id, route=routing.destination.value)
            return

        if routing.destination == TaskType.diagnosis:
            agent = DiagnosisAgent(
                DiagnosisAgentConfig(
                    google_api_key=settings.GOOGLE_API_KEY or "",
                    model=settings.GEMINI_MODEL,
                    mongo_url=mongo_url,
                )
            )
            diag = await agent.process(task_input)
            # Call via instance to support dependency injection / monkeypatching in tests.
            answer = agent.render_markdown(diag)
            if hasattr(db, "create_agent_log"):
                trace = getattr(agent, "last_trace", None) or {}
                with suppress(Exception):
                    await db.create_agent_log(
                        AgentLogEntry(
                            task_id=task_id,
                            session_id=session_id,
                            agent="diagnosis_agent",
                            stage=trace.get("stage") or "main",
                            model=trace.get("model") or getattr(settings, "GEMINI_MODEL", None),
                            prompt=trace.get("prompt"),
                            raw_output=trace.get("raw_output"),
                            parsed_output=trace.get("parsed_output") or diag.model_dump(),
                            latency_ms=trace.get("latency_ms"),
                        )
                    )
            debug2 = {**debug, "diagnosis": diag.model_dump()}
            result = TaskResult(answer=answer, model=settings.GEMINI_MODEL, debug=debug2).model_dump()
            await db.update_task(task_id, status="completed", result=result)
            log.info("worker_completed_task", task_id=task_id, route=routing.destination.value)
            return

        # Any remaining case defaults to content (AC2).
        agent = ContentAgent(
            ContentAgentConfig(
                google_api_key=settings.GOOGLE_API_KEY or "",
                tavily_api_key=settings.TAVILY_API_KEY or "",
                model=settings.GEMINI_MODEL,
            ),
            mongo=db,
        )
        output = await agent.process(task_input)
        result = TaskResult(answer=output.answer, sources=output.sources, model=output.model, debug=debug).model_dump()
        await db.update_task(task_id, status="completed", result=result)
        log.info("worker_completed_task", task_id=task_id, route="content_default")
    except ContentAgentError as e:
        log.error("worker_failed_task", task_id=task_id, error=str(e), stage=getattr(e, "stage", "unknown"), exc_info=True)
        # Check if it's an LLM provider quota error (not our API rate limit)
        if _is_llm_quota_error(e):
            error_msg = _get_llm_quota_message(task)
            log.warning("worker_llm_quota_exceeded", task_id=task_id)
        else:
            error_msg = str(e)
        result = TaskResult(error=error_msg, stage=getattr(e, "stage", "unknown"), model=settings.GEMINI_MODEL).model_dump()
        await db.update_task(task_id, status="failed", result=result)
    except (ValidationError, PyMongoError, ConnectionError, TimeoutError, OSError, RuntimeError, ValueError, TypeError, ChatGoogleGenerativeAIError) as e:
        # Check if it's an LLM provider quota error (not our API rate limit)
        if _is_llm_quota_error(e):
            error_msg = _get_llm_quota_message(task)
            log.warning("worker_llm_quota_exceeded", task_id=task_id, original_error=str(e)[:200])
        else:
            error_msg = str(e)
            log.error("worker_failed_task", task_id=task_id, error=str(e), exc_info=True)
        model = getattr(settings, "GEMINI_MODEL", None)
        result = TaskResult(error=error_msg, stage="unknown", model=model).model_dump()
        with suppress(PyMongoError, ConnectionError, TimeoutError, OSError, RuntimeError):
            await db.update_task(task_id, status="failed", result=result)
    finally:
        with suppress(PyMongoError, ConnectionError, TimeoutError, OSError, RuntimeError):
            await db.close()


