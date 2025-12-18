from __future__ import annotations

import pytest

from src.agents.code_agent import CodeAgent, CodeAgentConfig
from src.models.task_input import TaskInput


@pytest.mark.asyncio
async def test_code_agent_returns_code_block_and_explanation():
    class DummyMsg:
        def __init__(self, content: str):
            self.content = content

    class DummyLLM:
        async def ainvoke(self, _messages):
            return DummyMsg(
                "```python\nprint('hi')\n```\n\nThis prints a greeting to stdout."
            )

    agent = CodeAgent(CodeAgentConfig(google_api_key="x", model="gemini-3-pro"), llm=DummyLLM())
    out = await agent.process(TaskInput(task="write python code"))

    assert "```" in out
    assert "```" in out
    assert "This" in out


