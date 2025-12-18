from __future__ import annotations

import pytest

from src.agents.code_agent import CodeAgent, CodeAgentConfig
from src.models.task_input import TaskInput


@pytest.mark.asyncio
async def test_code_agent_refuses_destructive_request_without_llm_call():
    class DummyLLM:
        async def ainvoke(self, _messages):
            raise AssertionError("LLM should not be called for destructive requests")

    agent = CodeAgent(CodeAgentConfig(google_api_key="x", model="gemini-3-pro"), llm=DummyLLM())
    out = await agent.process(TaskInput(task="Write a script to delete all files"))

    assert "can’t help" in out or "can't help" in out
    assert "```" in out
    # Ensure we didn't emit obviously destructive helpers.
    assert "shutil.rmtree" not in out
    assert "os.remove" not in out


@pytest.mark.asyncio
async def test_code_agent_prompt_requires_python_code_block():
    captured = {}

    class DummyMsg:
        def __init__(self, content: str):
            self.content = content

    class DummyLLM:
        async def ainvoke(self, messages):
            captured["messages"] = messages
            return DummyMsg("```python\nprint('ok')\n```\n\nExplanation.")

    agent = CodeAgent(CodeAgentConfig(google_api_key="x", model="gemini-3-pro"), llm=DummyLLM())
    out = await agent.process(TaskInput(task="Write a Python script to parse a CSV"))

    assert "```" in out
    sys_msg = captured["messages"][0].content
    assert "```{language}" in sys_msg


