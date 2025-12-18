"""Tests for the improved CodeAgent with web search and conversation memory.

Tests requirements from research_about_code_Agent.md:
- Web Search Tool integration (Tavily)
- Conversation history/memory
- Research-first approach
- Reference links
- Installation instructions
"""
from __future__ import annotations

import pytest

from src.agents.code_agent import (
    CodeAgent,
    CodeAgentConfig,
    ConversationHistory,
    _conversation_history,
)
from src.models.task_input import TaskInput


# =============================================================================
# Basic Code Generation Tests
# =============================================================================

@pytest.mark.asyncio
async def test_code_agent_returns_code_block_and_explanation():
    """Test that CodeAgent returns properly formatted code and explanation."""
    class DummyMsg:
        def __init__(self, content: str):
            self.content = content

    class DummyLLM:
        async def ainvoke(self, _messages):
            return DummyMsg(
                "```python\nprint('hi')\n```\n\nThis prints a greeting to stdout."
            )

    agent = CodeAgent(CodeAgentConfig(google_api_key="x", model="gemini-3-flash-preview"), llm=DummyLLM())
    out = await agent.process(TaskInput(task="write python code"))

    assert "```" in out
    assert "print" in out


@pytest.mark.asyncio
async def test_code_agent_detects_language_correctly():
    """Test language detection for various programming languages."""
    agent = CodeAgent(CodeAgentConfig(google_api_key="x", model="gemini-3-flash-preview"), llm=object())

    # Test Python (default)
    assert agent._detect_language("write a function") == "python"

    # Test TypeScript
    assert agent._detect_language("write typescript code") == "typescript"

    # Test JavaScript
    assert agent._detect_language("write javascript code") == "javascript"

    # Test Go
    assert agent._detect_language("write golang code") == "go"

    # Test Rust
    assert agent._detect_language("write rust code") == "rust"


# =============================================================================
# Web Search Integration Tests
# =============================================================================

@pytest.mark.asyncio
async def test_code_agent_with_web_search():
    """Test CodeAgent integrates web search results and appends references."""
    class DummyMsg:
        def __init__(self, content: str):
            self.content = content

    class DummyLLM:
        async def ainvoke(self, messages):
            # Verify system prompt includes search context
            sys_content = messages[0].content
            assert "Web Search Results" in sys_content or "No web search" in sys_content
            return DummyMsg(
                "```python\nimport requests\nresponse = requests.get('https://api.example.com')\n```\n\n"
                "This code makes an HTTP GET request using the requests library.\n\n"
                "**Installation:** `pip install requests`"
            )

    class DummyTavily:
        async def search(self, query, **kwargs):
            return {
                "results": [
                    {
                        "title": "Requests Documentation",
                        "url": "https://docs.python-requests.org/",
                        "content": "Requests is a simple HTTP library for Python.",
                    }
                ]
            }

    agent = CodeAgent(
        CodeAgentConfig(
            google_api_key="x",
            model="gemini-3-flash-preview",
            tavily_api_key="test-key",
        ),
        llm=DummyLLM(),
        tavily_client=DummyTavily(),
    )

    out = await agent.process(TaskInput(task="write code using requests library"))

    assert "```python" in out
    assert "requests" in out
    # References should be appended
    assert "Sources:" in out or "Requests Documentation" in out


@pytest.mark.asyncio
async def test_code_agent_handles_search_failure_gracefully():
    """Test CodeAgent works even when web search fails."""
    class DummyMsg:
        def __init__(self, content: str):
            self.content = content

    class DummyLLM:
        async def ainvoke(self, _messages):
            return DummyMsg("```python\nprint('ok')\n```\n\nExplanation.")

    class FailingTavily:
        async def search(self, query, **kwargs):
            raise RuntimeError("Search failed")

    agent = CodeAgent(
        CodeAgentConfig(
            google_api_key="x",
            model="gemini-3-flash-preview",
            tavily_api_key="test-key",
        ),
        llm=DummyLLM(),
        tavily_client=FailingTavily(),
    )

    # Should not raise, just proceed without search results
    out = await agent.process(TaskInput(task="write python code"))
    assert "```python" in out


# =============================================================================
# Conversation History Tests
# =============================================================================

def test_conversation_history_add_and_retrieve():
    """Test ConversationHistory stores and retrieves messages correctly."""
    history = ConversationHistory(max_turns=5)

    history.add_turn("session1", "user", "Write a function")
    history.add_turn("session1", "assistant", "Here's the code...")

    turns = history.get_history("session1")
    assert len(turns) == 2
    assert turns[0]["role"] == "user"
    assert turns[1]["role"] == "assistant"


def test_conversation_history_separate_sessions():
    """Test that different sessions have separate histories."""
    history = ConversationHistory(max_turns=5)

    history.add_turn("session1", "user", "Message 1")
    history.add_turn("session2", "user", "Message 2")

    assert len(history.get_history("session1")) == 1
    assert len(history.get_history("session2")) == 1
    assert history.get_history("session1")[0]["content"] == "Message 1"
    assert history.get_history("session2")[0]["content"] == "Message 2"


def test_conversation_history_max_turns_limit():
    """Test that history is truncated when max turns exceeded."""
    history = ConversationHistory(max_turns=2)

    for i in range(10):
        history.add_turn("session1", "user", f"Message {i}")

    # Should keep last max_turns * 2 = 4 messages
    turns = history.get_history("session1")
    assert len(turns) <= 4


def test_conversation_history_format_for_prompt():
    """Test formatting history for inclusion in prompts."""
    history = ConversationHistory(max_turns=5)

    history.add_turn("session1", "user", "Write a sorting function")
    history.add_turn("session1", "assistant", "Here's a quicksort implementation...")

    formatted = history.format_for_prompt("session1")

    assert "Previous Conversation" in formatted
    assert "User" in formatted
    assert "Assistant" in formatted
    assert "sorting function" in formatted


def test_conversation_history_empty_session():
    """Test formatting for session with no history."""
    history = ConversationHistory(max_turns=5)
    formatted = history.format_for_prompt("nonexistent")
    assert formatted == ""


@pytest.mark.asyncio
async def test_code_agent_uses_session_for_conversation_history():
    """Test that CodeAgent uses session_id for conversation tracking."""
    class DummyMsg:
        def __init__(self, content: str):
            self.content = content

    class DummyLLM:
        async def ainvoke(self, messages):
            return DummyMsg("```python\nprint('ok')\n```\n\nExplanation.")

    # Clear global history for test isolation
    _conversation_history._history.clear()

    agent = CodeAgent(CodeAgentConfig(google_api_key="x", model="gemini-3-flash-preview"), llm=DummyLLM())

    # First request in session
    await agent.process(TaskInput(task="write a hello world", session_id="test-session"))

    # Check that history was recorded
    history = _conversation_history.get_history("test-session")
    assert len(history) == 2  # user + assistant
    assert "hello world" in history[0]["content"]


# =============================================================================
# Search Query Building Tests
# =============================================================================

def test_build_search_query_with_library():
    """Test search query construction when library is mentioned."""
    agent = CodeAgent(CodeAgentConfig(google_api_key="x", model="gemini-3-flash-preview"), llm=object())

    query = agent._build_search_query("write code using langchain", "python")
    assert "langchain" in query.lower()
    assert "python" in query.lower()


def test_build_search_query_generic():
    """Test search query construction for generic requests."""
    agent = CodeAgent(CodeAgentConfig(google_api_key="x", model="gemini-3-flash-preview"), llm=object())

    query = agent._build_search_query("write a sorting algorithm", "python")
    assert "python" in query.lower()


# =============================================================================
# Trace/Observability Tests
# =============================================================================

@pytest.mark.asyncio
async def test_code_agent_records_trace_with_references():
    """Test that last_trace includes search references."""
    class DummyMsg:
        def __init__(self, content: str):
            self.content = content

    class DummyLLM:
        async def ainvoke(self, _messages):
            return DummyMsg("```python\ncode\n```\n\nExplanation.")

    class DummyTavily:
        async def search(self, query, **kwargs):
            return {
                "results": [
                    {"title": "Doc", "url": "https://example.com", "content": "Info"}
                ]
            }

    agent = CodeAgent(
        CodeAgentConfig(google_api_key="x", model="gemini-3-flash-preview", tavily_api_key="key"),
        llm=DummyLLM(),
        tavily_client=DummyTavily(),
    )

    await agent.process(TaskInput(task="write code"))

    assert agent.last_trace is not None
    assert agent.last_trace["agent"] == "code_agent"
    assert agent.last_trace["stage"] == "generation"
    assert "references" in agent.last_trace
    assert len(agent.last_trace["references"]) > 0


# =============================================================================
# Mongo-backed Session History Tests
# =============================================================================

@pytest.mark.asyncio
async def test_code_agent_uses_mongo_session_history_for_context():
    """Test that CodeAgent loads conversation history from Mongo for follow-ups."""
    class DummyMsg:
        def __init__(self, content: str):
            self.content = content

    # Capture the system prompt to verify history is included
    captured_prompts: list[str] = []

    class DummyLLM:
        async def ainvoke(self, messages):
            captured_prompts.append(messages[0].content)
            return DummyMsg("```python\nprint('fixed')\n```\n\nFixed the bug.")

    class DummyMongo:
        async def list_tasks_by_session(self, session_id: str, *, limit: int = 50):
            assert session_id == "code-session-123"
            return [
                {
                    "task": "Write a sorting function",
                    "result": {"answer": "```python\ndef sort(arr): return sorted(arr)\n```\n\nSorting function."},
                },
                {
                    "task": "There's a bug in the sort function",
                    "result": None,  # Current task, no response yet
                },
            ]

    agent = CodeAgent(
        CodeAgentConfig(google_api_key="x", model="gemini-3-flash-preview"),
        llm=DummyLLM(),
        mongo=DummyMongo(),
    )

    await agent.process(TaskInput(task="Fix the bug", session_id="code-session-123"))

    # Verify history was included in prompt
    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert "Previous Conversation" in prompt
    assert "sorting function" in prompt.lower() or "sort" in prompt.lower()


@pytest.mark.asyncio
async def test_code_agent_falls_back_to_inmemory_when_mongo_unavailable():
    """Test that CodeAgent falls back to in-memory history when Mongo fails."""
    class DummyMsg:
        def __init__(self, content: str):
            self.content = content

    class DummyLLM:
        async def ainvoke(self, _messages):
            return DummyMsg("```python\nprint('ok')\n```\n\nDone.")

    class FailingMongo:
        async def list_tasks_by_session(self, session_id: str, *, limit: int = 50):
            raise ConnectionError("Mongo unavailable")

    # Clear global history
    _conversation_history._history.clear()

    agent = CodeAgent(
        CodeAgentConfig(google_api_key="x", model="gemini-3-flash-preview"),
        llm=DummyLLM(),
        mongo=FailingMongo(),
    )

    # Should not raise, should use in-memory fallback
    out = await agent.process(TaskInput(task="write code", session_id="fallback-session"))
    assert "```python" in out