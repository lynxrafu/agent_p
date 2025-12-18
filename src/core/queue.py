"""Redis/RQ queue wiring."""

from __future__ import annotations

import redis
from rq import Queue

from src.core.settings import get_settings


def get_redis_connection() -> redis.Redis:
    """Create a Redis connection using `REDIS_URL` from settings."""
    # Context7 redis-py docs show redis.from_url(...) usage.
    return redis.from_url(get_settings().REDIS_URL)


def get_task_queue() -> Queue:
    """Return the RQ Queue used to enqueue agent tasks."""
    return Queue("agent_tasks", connection=get_redis_connection())


