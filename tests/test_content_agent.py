from __future__ import annotations

import pytest

from src.agents.content_agent import ContentAgent, ContentAgentConfig, ContentSearchError
from src.models.task_input import TaskInput


@pytest.mark.asyncio
async def test_content_agent_formats_sources_urls_only():
    class DummyTavily:
        async def search(self, **kwargs):
            _ = kwargs
            return {
                "results": [
                    {"title": "A", "url": "https://a.example", "score": 0.9, "content": "aaa"},
                    {"title": "B", "url": "https://b.example", "score": 0.8, "content": "bbb"},
                ]
            }

    class DummyMsg:
        def __init__(self, content: str):
            self.content = content

    class DummyLLM:
        async def ainvoke(self, messages):
            _ = messages
            return DummyMsg("hello world")

    agent = ContentAgent(
        ContentAgentConfig(
            google_api_key="x",
            tavily_api_key="y",
            model="gemini-3-pro",
        ),
        tavily_client=DummyTavily(),
        llm=DummyLLM(),
    )

    out = await agent.process(TaskInput(task="what is x?"))
    assert "Answer:" in out.answer
    assert "Sources:" in out.answer
    assert "- https://a.example" in out.answer
    assert "- https://b.example" in out.answer


@pytest.mark.asyncio
async def test_content_agent_raises_on_zero_results():
    class DummyTavily:
        async def search(self, **kwargs):
            _ = kwargs
            return {"results": []}

    class DummyLLM:
        async def ainvoke(self, messages):
            _ = messages
            raise AssertionError("LLM should not be called when search has 0 results")

    agent = ContentAgent(
        ContentAgentConfig(
            google_api_key="x",
            tavily_api_key="y",
            model="gemini-3-pro",
        ),
        tavily_client=DummyTavily(),
        llm=DummyLLM(),
    )

    with pytest.raises(ContentSearchError):
        await agent.process(TaskInput(task="anything"))


