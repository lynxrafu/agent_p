"""MongoDB repository for persisting tasks and results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo import AsyncMongoClient

from src.models.agent_logs import AgentLogEntry


class Mongo:
    """Async MongoDB repository for agent task documents."""

    def __init__(self, mongo_url: str) -> None:
        self.client = AsyncMongoClient(mongo_url)
        self.db = self.client.agent_p
        self.tasks = self.db.agent_tasks
        self.agent_logs = self.db.agent_logs
        # Session-level state (e.g., active agent stickiness per session_id).
        self.sessions = self.db.agent_sessions

    async def ping(self) -> None:
        await self.client.admin.command("ping")

    async def close(self) -> None:
        # PyMongo async close is awaitable.
        await self.client.close()

    async def create_task(self, task_id: str, task: str) -> None:
        now = datetime.now(timezone.utc)
        await self.tasks.insert_one(
            {
                "task_id": task_id,
                "task": task,
                "status": "queued",
                "created_at": now,
                "updated_at": now,
            }
        )

    async def update_task(self, task_id: str, status: str, result: Any | None = None) -> None:
        now = datetime.now(timezone.utc)
        update: dict[str, Any] = {"status": status, "updated_at": now}
        if result is not None:
            update["result"] = result
        # Never silently lose updates if the task doc doesn't exist.
        await self.tasks.update_one(
            {"task_id": task_id},
            {"$set": update, "$setOnInsert": {"created_at": now, "task_id": task_id}},
            upsert=True,
        )

    async def set_task_session(self, task_id: str, session_id: str) -> None:
        """Attach a session_id to an existing task document (for stateful agents)."""
        now = datetime.now(timezone.utc)
        await self.tasks.update_one(
            {"task_id": task_id},
            {"$set": {"session_id": session_id, "updated_at": now}},
            upsert=False,
        )

    async def set_task_route(
        self,
        task_id: str,
        route: str,
        route_confidence: float | None,
        route_rationale: str | None,
    ) -> None:
        now = datetime.now(timezone.utc)
        await self.tasks.update_one(
            {"task_id": task_id},
            {
                "$set": {
                    "route": route,
                    "route_confidence": route_confidence,
                    "route_rationale": route_rationale,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now, "task_id": task_id},
            },
            upsert=True,
        )

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        doc = await self.tasks.find_one({"task_id": task_id}, projection={"_id": 0})
        return doc

    async def list_tasks_by_session(self, session_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        """List tasks for a session in chronological order (oldest first)."""
        cursor = (
            self.tasks.find({"session_id": session_id}, projection={"_id": 0})
            .sort("created_at", 1)
            .limit(limit)
        )
        return [doc async for doc in cursor]

    async def set_active_agent(self, session_id: str, active_agent: str | None) -> None:
        """Set (or clear) the active agent for a session.
        
        This enables "system-level session continuity" where the router can keep
        a multi-turn interaction on the same specialist agent.
        """
        now = datetime.now(timezone.utc)
        update: dict[str, Any] = {"updated_at": now}
        if active_agent is None:
            update["active_agent"] = None
        else:
            update["active_agent"] = active_agent
        await self.sessions.update_one(
            {"session_id": session_id},
            {"$set": update, "$setOnInsert": {"created_at": now, "session_id": session_id}},
            upsert=True,
        )

    async def get_active_agent(self, session_id: str) -> dict[str, Any] | None:
        """Get the session state doc (includes active_agent + timestamps)."""
        return await self.sessions.find_one({"session_id": session_id}, projection={"_id": 0})

    async def create_agent_log(self, entry: AgentLogEntry) -> None:
        """Persist a deep observability agent log entry."""
        now = datetime.now(timezone.utc)
        doc = entry.model_dump()
        doc["created_at"] = doc.get("created_at") or now
        await self.agent_logs.insert_one(doc)

    async def list_agent_logs_by_session(self, session_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        """Fetch recent agent logs for a session (most recent first)."""
        cursor = (
            self.agent_logs.find({"session_id": session_id}, projection={"_id": 0})
            .sort("created_at", -1)
            .limit(limit)
        )
        return [doc async for doc in cursor]


