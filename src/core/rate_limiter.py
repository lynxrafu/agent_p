"""Redis-backed token bucket rate limiter."""

from __future__ import annotations

from dataclasses import dataclass
import time

import redis.asyncio as redis


_TOKEN_BUCKET_LUA = r"""
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local data = redis.call("HMGET", key, "tokens", "ts")
local tokens = tonumber(data[1])
local ts = tonumber(data[2])

if tokens == nil then tokens = capacity end
if ts == nil then ts = now end

local delta = now - ts
if delta < 0 then delta = 0 end

tokens = math.min(capacity, tokens + (delta * rate))

local allowed = 0
local retry_after = 0
if tokens >= requested then
  allowed = 1
  tokens = tokens - requested
else
  retry_after = (requested - tokens) / rate
end

redis.call("HMSET", key, "tokens", tokens, "ts", now)
local ttl = math.ceil((capacity / rate) * 2)
redis.call("EXPIRE", key, ttl)

return {allowed, tokens, retry_after}
"""


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    remaining: float
    retry_after_s: float
    limit: int


@dataclass(frozen=True)
class RateLimiterConfig:
    redis_url: str
    requests_per_min: int = 60
    burst: int = 60
    key_prefix: str = "rl"


class RateLimiter:
    """Token bucket limiter using Redis for shared state."""

    def __init__(self, config: RateLimiterConfig, *, client: redis.Redis | None = None) -> None:
        if config.requests_per_min <= 0 or config.burst <= 0:
            raise ValueError("Rate limiter config must be positive")
        self._config = config
        self._client = client or redis.from_url(config.redis_url)

    async def close(self) -> None:
        await self._client.aclose()

    async def allow(self, identity: str) -> RateLimitResult:
        key = f"{self._config.key_prefix}:{identity}"
        capacity = float(self._config.burst)
        rate = float(self._config.requests_per_min) / 60.0  # tokens per second
        now = time.time()

        allowed, remaining, retry_after = await self._client.eval(
            _TOKEN_BUCKET_LUA,
            numkeys=1,
            keys=[key],
            args=[capacity, rate, now, 1],
        )

        return RateLimitResult(
            allowed=bool(int(allowed)),
            remaining=float(remaining),
            retry_after_s=float(retry_after),
            limit=int(self._config.requests_per_min),
        )


