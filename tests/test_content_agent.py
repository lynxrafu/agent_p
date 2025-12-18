from __future__ import annotations

import pytest

from src.agents.content_agent import (
    ContentAgent,
    ContentAgentConfig,
)
from src.models.task_input import TaskInput

# This test module intentionally probes internal helpers for unit-level behavior.
# pylint: disable=protected-access


# ---------------------------------------------------------------------------
# Helper classes for mocking
# ---------------------------------------------------------------------------
class DummyMsg:
    """Mock LLM response."""

    def __init__(self, content: str):
        self.content = content


class DummyTavily:
    """Mock Tavily client returning standard results."""

    async def search(self, **kwargs):
        _ = kwargs
        return {
            "results": [
                {"title": "Source A", "url": "https://a.example", "score": 0.9, "content": "Content from source A"},
                {"title": "Source B", "url": "https://b.example", "score": 0.8, "content": "Content from source B"},
            ]
        }


class DummyLLM:
    """Mock LLM returning a simple response."""

    async def ainvoke(self, messages):
        _ = messages
        return DummyMsg("This is the synthesized answer [1]. Additional info [2].")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_content_agent_formats_with_numbered_references():
    """Test that ContentAgent formats output with numbered references."""
    agent = ContentAgent(
        ContentAgentConfig(
            google_api_key="x",
            tavily_api_key="y",
            model="gemini-3-flash-preview",
            enable_retry_search=False,
        ),
        tavily_client=DummyTavily(),
        llm=DummyLLM(),
    )

    out = await agent.process(TaskInput(task="what is x?"))

    # Check the new numbered reference format
    assert "## References" in out.answer
    assert "[1] Source A - https://a.example" in out.answer
    assert "[2] Source B - https://b.example" in out.answer
    assert "synthesized answer" in out.answer

    # Verify sources are returned
    assert len(out.sources) == 2
    assert out.sources[0].url == "https://a.example"
    assert out.sources[1].url == "https://b.example"


@pytest.mark.asyncio
async def test_content_agent_graceful_fallback_on_zero_results():
    """Test that ContentAgent returns graceful message when Tavily returns 0 results.
    
    Per spec §5: "If no information is found, the agent should clearly state: 
    'I could not find reliable sources regarding this specific topic.'"
    """
    class EmptyTavily:
        async def search(self, **kwargs):
            _ = kwargs
            return {"results": []}

    class UnusedLLM:
        async def ainvoke(self, messages):
            _ = messages
            raise AssertionError("LLM should not be called when search has 0 results")

    agent = ContentAgent(
        ContentAgentConfig(
            google_api_key="x",
            tavily_api_key="y",
            model="gemini-3-flash-preview",
            enable_retry_search=False,  # Disable retry to test immediate failure
        ),
        tavily_client=EmptyTavily(),
        llm=UnusedLLM(),
    )

    # Should NOT raise - instead returns graceful fallback message
    out = await agent.process(TaskInput(task="obscure topic xyz"))
    
    assert "could not find reliable sources" in out.answer.lower()
    assert len(out.sources) == 0
    assert agent.last_trace is not None
    assert agent.last_trace["stage"] == "no_results_fallback"


@pytest.mark.asyncio
async def test_content_agent_retry_finds_results_on_second_query():
    """Test that retry logic can find results with alternative query variations."""
    search_calls: list[str] = []
    
    class RetryTavily:
        async def search(self, query: str, **kwargs):
            _ = kwargs
            search_calls.append(query)
            # First call returns empty, subsequent calls return results
            if len(search_calls) == 1:
                return {"results": []}
            return {
                "results": [
                    {"title": "Found It", "url": "https://found.example", "score": 0.9, "content": "Finally found content"}
                ]
            }

    agent = ContentAgent(
        ContentAgentConfig(
            google_api_key="x",
            tavily_api_key="y",
            model="gemini-3-flash-preview",
            enable_retry_search=True,  # Enable retry
            min_sources_threshold=1,
        ),
        tavily_client=RetryTavily(),
        llm=DummyLLM(),
    )

    out = await agent.process(TaskInput(task="what is something"))
    
    # Should have retried and found results
    assert len(search_calls) >= 2  # Original + at least one retry
    assert len(out.sources) >= 1
    assert "## References" in out.answer


@pytest.mark.asyncio
async def test_content_agent_context_resolution():
    """Test that pronouns are resolved using conversation history."""
    agent = ContentAgent(
        ContentAgentConfig(
            google_api_key="x",
            tavily_api_key="y",
            model="gemini-3-flash-preview",
            enable_retry_search=False,
        ),
        tavily_client=DummyTavily(),
        llm=DummyLLM(),
    )

    # Simulate conversation history (as would be loaded from MongoDB)
    history = [
        {"role": "user", "content": "Tell me about Python programming language"},
        {"role": "assistant", "content": "Python is a high-level language..."},
    ]

    # Query with pronoun - should resolve using history
    resolved = agent._resolve_context_references("Who created it?", history)  # noqa: SLF001

    # Should include context from previous conversation
    assert "Context from conversation" in resolved
    assert "Python" in resolved

    # Query without pronoun - should not add context
    unchanged = agent._resolve_context_references("What is JavaScript?", history)  # noqa: SLF001
    assert "Context from conversation" not in unchanged
    assert unchanged == "What is JavaScript?"


@pytest.mark.asyncio
async def test_content_agent_query_variations():
    """Test query variation generation for retry logic."""
    agent = ContentAgent(
        ContentAgentConfig(
            google_api_key="x",
            tavily_api_key="y",
            model="gemini-3-flash-preview",
        ),
        tavily_client=DummyTavily(),
        llm=DummyLLM(),
    )

    # Test conceptual query variations
    variations = agent._generate_query_variations("What is quantum computing?")  # noqa: SLF001
    assert len(variations) == 2
    assert any("explanation" in v for v in variations)

    # Test non-conceptual query variations
    variations2 = agent._generate_query_variations("Python best practices")  # noqa: SLF001
    assert len(variations2) == 2
    assert any("overview" in v or "guide" in v for v in variations2)


@pytest.mark.asyncio
async def test_content_agent_trace_recording():
    """Test that ContentAgent records trace information."""
    agent = ContentAgent(
        ContentAgentConfig(
            google_api_key="x",
            tavily_api_key="y",
            model="gemini-3-flash-preview",
            enable_retry_search=False,
        ),
        tavily_client=DummyTavily(),
        llm=DummyLLM(),
    )

    await agent.process(TaskInput(task="test query"))

    # Verify trace is recorded
    assert agent.last_trace is not None
    assert agent.last_trace["agent"] == "content_agent"
    assert agent.last_trace["stage"] == "synthesis"
    assert agent.last_trace["model"] == "gemini-3-flash-preview"
    assert "latency_ms" in agent.last_trace
    assert agent.last_trace["sources_count"] == 2


@pytest.mark.asyncio
async def test_content_agent_uses_mongo_session_history_for_pronoun_resolution():
    """Ensure session continuity works across processes by loading history from Mongo."""

    class CaptureTavily(DummyTavily):
        def __init__(self):
            self.last_query = None

        async def search(self, **kwargs):
            self.last_query = kwargs.get("query")
            return await super().search(**kwargs)

    class DummyMongo:
        async def list_tasks_by_session(self, session_id: str, *, limit: int = 20):
            assert session_id == "mongo-sess-1"
            _ = limit
            return [
                {"task": "Tell me about Python programming language", "result": {"answer": "Python is a language."}},
            ]

    tav = CaptureTavily()
    agent = ContentAgent(
        ContentAgentConfig(
            google_api_key="x",
            tavily_api_key="y",
            model="gemini-3-flash-preview",
            enable_retry_search=False,
        ),
        tavily_client=tav,
        llm=DummyLLM(),
        mongo=DummyMongo(),  # type: ignore[arg-type]
    )

    await agent.process(TaskInput(task="Who created it?", session_id="mongo-sess-1"))
    assert tav.last_query is not None
    assert "Context from conversation" in tav.last_query
    assert "Tell me about Python" in tav.last_query

