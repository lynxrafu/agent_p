"""Application settings and lazy settings loader."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Note: docker-compose passes env vars into containers; we validate presence here.
    """

    # Load `.env` if present; always allow `env.example` for local defaults.
    model_config = SettingsConfigDict(
        env_file=(".env", "env.example"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    MONGODB_URL: str
    REDIS_URL: str
    LOG_LEVEL: str = "INFO"

    # Future stories (agents)
    GOOGLE_API_KEY: str | None = None
    TAVILY_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-3-pro"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (lazy-loaded)."""
    # Lazy-load to avoid import-time crashes in tooling/tests when env isn't set yet.
    return Settings()


