from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo import AsyncMongoClient


class Mongo:
    def __init__(self, mongo_url: str) -> None:
        self.client = AsyncMongoClient(mongo_url)
        self.db = self.client.agent_p
        self.tasks = self.db.agent_tasks

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


