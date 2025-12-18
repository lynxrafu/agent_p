from __future__ import annotations

import redis
from rq import Queue

from src.core.settings import get_settings


def get_redis_connection():
    # Context7 redis-py docs show redis.from_url(...) usage.
    return redis.from_url(get_settings().REDIS_URL)


def get_task_queue() -> Queue:
    return Queue("agent_tasks", connection=get_redis_connection())


