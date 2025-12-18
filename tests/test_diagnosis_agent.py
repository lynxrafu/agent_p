from __future__ import annotations

import pytest

from src.agents.diagnosis_agent import DiagnosisAgent, DiagnosisAgentConfig
from src.models.task_input import TaskInput


@pytest.mark.asyncio
async def test_diagnosis_agent_uses_session_transcript_when_available():
    class DummyMongo:
        async def list_tasks_by_session(self, session_id: str, *, limit: int = 50):
            assert session_id == "s1"
            return [
                {"task": "Sales are down", "result": {"answer": "Which region is impacted?"}},
                {"task": "EMEA is down 20%", "result": {"answer": "Which channel?"}},
            ]

    class DummyLLM:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _inputs):
            raise AssertionError("Chain should be used via prompt piping in this test")

    agent = DiagnosisAgent(
        DiagnosisAgentConfig(google_api_key="", model="gemini-3-pro", mongo_url="mongodb://x"),
        llm=None,
        mongo=DummyMongo(),  # type: ignore[arg-type]
    )
    # Force fallback to avoid network/LLM.
    agent._chain = None  # type: ignore[attr-defined]

    out = await agent.process(TaskInput(task="", session_id="s1"))
    md = DiagnosisAgent.render_markdown(out)
    assert "Problem tree" in md
    assert out.tree.label


