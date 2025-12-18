"""ContentAgent: web search + grounded synthesis with citations.

This agent acts as an expert Researcher and Technical Writer,
synthesizing information from web sources into well-structured, cited answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import AsyncTavilyClient
from tavily.errors import InvalidAPIKeyError as TavilyInvalidAPIKeyError

from src.agents.base import BaseAgent
from src.db.mongo import Mongo
from src.models.agent_content import ContentAgentOutput, ContentSource
from src.models.task_input import TaskInput


def _extract_llm_content(content: Any) -> str:
    """Extract text content from LLM response, handling Gemini 3's new format.
    
    Gemini 3 models return content as a list of dicts:
    [{'type': 'text', 'text': 'Hello', 'extras': {...}}]
    
    Older models return a simple string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Gemini 3 format: list of content blocks
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts)
    return ""


class ContentAgentError(RuntimeError):
    """Base error for ContentAgent failures."""

    stage: str = "unknown"


class ContentConfigError(ContentAgentError):
    stage = "config"


class ContentInputError(ContentAgentError):
    stage = "input"


class ContentSearchError(ContentAgentError):
    stage = "search"


class ContentSynthesisError(ContentAgentError):
    stage = "synthesis"


@dataclass(frozen=True)
class ContentAgentConfig:
    google_api_key: str
    tavily_api_key: str
    model: str
    max_results: int = 5
    search_depth: str = "advanced"
    enable_retry_search: bool = True  # Retry with refined query if initial search is poor
    min_sources_threshold: int = 2  # Minimum sources needed before retry


class ContentAgent(BaseAgent):
    """ContentAgent: Expert Researcher and Technical Writer.

    Synthesizes information from web sources into well-structured,
    properly cited answers following academic standards.

    Features:
    - MongoDB-backed conversation memory for cross-worker persistence
    - Query optimization with retry on insufficient results
    - Inline citation formatting [1], [2], etc.
    - Date-aware prompting for recency-sensitive queries
    - Conflict detection between sources
    """

    def __init__(
        self,
        config: ContentAgentConfig,
        *,
        tavily_client: AsyncTavilyClient | None = None,
        llm: Any | None = None,
        mongo: Mongo | None = None,
    ) -> None:
        if not config.google_api_key:
            raise ContentConfigError("GOOGLE_API_KEY is required for ContentAgent")
        if not config.tavily_api_key:
            raise ContentConfigError("TAVILY_API_KEY is required for ContentAgent")

        self._config = config
        self._tavily = tavily_client or AsyncTavilyClient(api_key=config.tavily_api_key)
        self._mongo = mongo
        # Vendor guidance: Gemini 3 models recommend keeping temperature at default 1.0.
        # For other models, we keep a lower temperature for grounded synthesis.
        temperature = 1.0 if config.model.startswith("gemini-3") else 0.2
        self._llm = llm or ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=config.google_api_key,
            temperature=temperature,
        )
        self.last_trace: dict[str, Any] | None = None

    # -------------------------------------------------------------------------
    # Session Management (MongoDB-backed for cross-worker persistence)
    # -------------------------------------------------------------------------
    async def _load_session_history(self, session_id: str, *, limit: int = 20) -> list[dict[str, str]]:
        """Load conversation history from MongoDB for session continuity.
        
        Returns list of {"role": "user"|"assistant", "content": str} dicts.
        Task results are persisted to MongoDB in jobs.py, so this reconstructs
        the conversation from the agent_tasks collection.
        """
        if self._mongo is None or not session_id:
            return []
        try:
            docs = await self._mongo.list_tasks_by_session(session_id, limit=limit)
        except (TimeoutError, OSError, ValueError, TypeError, RuntimeError):
            return []

        history: list[dict[str, str]] = []
        for d in docs:
            task = (d.get("task") or "").strip()
            if task:
                history.append({"role": "user", "content": task})
            result = d.get("result")
            if isinstance(result, dict):
                answer = (result.get("answer") or "").strip()
                if answer:
                    # Truncate to keep prompt/context bounded.
                    if len(answer) > 800:
                        answer = answer[:800] + "..."
                    history.append({"role": "assistant", "content": answer})
        return history

    def _resolve_context_references(
        self,
        query: str,
        history: list[dict[str, str]],
    ) -> str:
        """Resolve pronouns and references using conversation history.

        Example: "Who created it?" after discussing Python -> adds context
        """
        if not history:
            return query

        # Check for pronouns that might need resolution
        # Use word boundary matching to catch pronouns at end of sentences
        pronouns = ["it", "this", "that", "they", "them", "he", "she", "its"]
        query_lower = query.lower()

        # Normalize query for pronoun detection (remove punctuation for matching)
        import re
        normalized = re.sub(r"[^\w\s]", " ", query_lower)
        words = set(normalized.split())
        has_pronoun = bool(words.intersection(pronouns))

        if not has_pronoun:
            return query

        # Build context summary from recent history
        recent_context = []
        for msg in history[-4:]:  # Last 2 exchanges
            if msg["role"] == "user":
                recent_context.append(f"Previous question: {msg['content']}")

        if recent_context:
            return f"{query}\n\n[Context from conversation: {'; '.join(recent_context)}]"
        return query

    # -------------------------------------------------------------------------
    # System Prompt
    # -------------------------------------------------------------------------
    def _get_system_prompt(self, sources: list[ContentSource]) -> str:
        """Build the expert researcher system prompt with date awareness."""
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Build source reference list for inline citations
        source_refs = "\n".join(
            f"[{i+1}] {s.title or 'Untitled'} - {s.url}"
            for i, s in enumerate(sources)
        )

        return f"""You are an expert Researcher and Technical Writer. You excel at synthesizing information from various sources into clear, easy-to-understand summaries.

## Current Date
Today is {current_date}. Use this for evaluating information recency.

## Available Sources for Citation
{source_refs}

## Citation Rules (MANDATORY)
1. **Every factual claim must be cited** using inline brackets matching the source numbers above: [1], [2], etc.
2. You may cite multiple sources for a single claim: [1][2]
3. Example: "Python was created by Guido van Rossum [1] and first released in 1991 [2]."

## Writing Guidelines
1. **Structure**: Use Markdown headers (##), bullet points, and **bold** for emphasis
2. **Objectivity**: Maintain neutral, informative tone - avoid personal opinions
3. **Clarity**: Break complex topics into digestible sections
4. **Verification**: If sources conflict, note explicitly: "Source [1] states X, while [2] suggests Y"
5. **Recency**: For "recent" or "latest" queries, prioritize newer information

## Response Format
Structure your response as:
1. Brief introduction/overview
2. Main content with headers and bullet points as needed
3. Use inline citations [1], [2] throughout
4. Do NOT include a references section (it will be added automatically)

## Constraints
- Use ONLY information from the provided web search context below
- Do NOT invent facts, dates, or statistics
- If context is insufficient, clearly state what is missing
- Never fabricate or guess at information

## Example
**Good**: "The framework was released in 2023 [1] and has gained significant adoption [2]."
**Bad**: "The framework was released recently and is very popular." (no citations)"""

    async def process(self, input_data: TaskInput) -> ContentAgentOutput:
        """Process a content research request.

        Args:
            input_data: TaskInput containing the query and optional session_id

        Returns:
            ContentAgentOutput with formatted answer, sources, and model info
        """
        task = input_data.task.strip()
        if not task:
            raise ContentInputError("Task content cannot be empty")

        session_id = getattr(input_data, "session_id", None)

        # Load conversation history from MongoDB for context continuity
        history: list[dict[str, str]] = []
        if session_id:
            history = await self._load_session_history(session_id)

        # Resolve context references (e.g., "it", "this") using session history
        resolved_query = self._resolve_context_references(task, history)

        # Search with retry optimization
        sources, context = await self._search_with_retry(resolved_query)

        # Handle case where no sources found (graceful fallback per spec §5)
        if not sources:
            answer_text = self._generate_no_results_response(task)
            answer = answer_text  # No references to append
            self.last_trace = {
                "agent": "content_agent",
                "stage": "no_results_fallback",
                "model": self._config.model,
                "prompt": None,
                "raw_output": None,
                "parsed_output": answer_text,
                "latency_ms": 0.0,
                "sources_count": 0,
            }
        else:
            # Synthesize answer with the new prompt system
            answer_text = await self._synthesize(task, context, sources)
            # Format final answer with numbered references
            answer = self._format_answer(answer_text, sources)

        # Note: Task results are persisted to MongoDB in jobs.py via db.update_task()
        # This automatically provides session history for future requests.

        return ContentAgentOutput(answer=answer, sources=sources, model=self._config.model)

    async def _search_with_retry(
        self, query: str
    ) -> tuple[list[ContentSource], str]:
        """Search with automatic query refinement if initial results are poor.
        
        Returns empty sources/context if all search attempts fail (graceful fallback).
        """
        sources: list[ContentSource] = []
        context: str = ""
        
        # First attempt
        try:
            sources, context = await self._search(query)
        except ContentSearchError:
            # Initial search failed, will try alternatives if retry enabled
            pass

        # Retry with refined query if below threshold and retry is enabled
        if (
            self._config.enable_retry_search
            and len(sources) < self._config.min_sources_threshold
        ):
            # Generate alternative query variations
            alt_queries = self._generate_query_variations(query)
            for alt_query in alt_queries[:2]:  # Try up to 2 alternatives
                try:
                    alt_sources, alt_context = await self._search(alt_query)
                    # Merge results, avoiding duplicates
                    existing_urls = {s.url for s in sources}
                    for s in alt_sources:
                        if s.url not in existing_urls:
                            sources.append(s)
                            existing_urls.add(s.url)
                    if alt_context:
                        context = f"{context}\n\n---\n\n{alt_context}" if context else alt_context
                    if len(sources) >= self._config.min_sources_threshold:
                        break
                except ContentSearchError:
                    continue

        return sources, context

    def _generate_query_variations(self, query: str) -> list[str]:
        """Generate alternative search queries for better coverage."""
        variations = []

        # Add "explained" or "definition" for conceptual queries
        query_lower = query.lower()
        if any(w in query_lower for w in ["what is", "explain", "define"]):
            variations.append(f"{query} simple explanation")
            variations.append(f"{query} definition")
        else:
            # For other queries, try adding context
            variations.append(f"{query} overview")
            variations.append(f"{query} guide")

        return variations

    def _generate_no_results_response(self, query: str) -> str:
        """Generate a graceful response when no search results are found.
        
        Per spec §5 Acceptance Criteria: "If no information is found, the agent 
        should clearly state: 'I could not find reliable sources regarding this 
        specific topic.'"
        """
        return f"""I could not find reliable sources regarding this specific topic.

## What I Searched For
- "{query}"
- Alternative variations of your question

## Suggestions
- Try rephrasing your question with more specific keywords
- Check if the topic name is spelled correctly
- For very recent events, information may not yet be widely available online

If you have a more specific aspect of this topic you'd like me to research, please let me know and I'll search again."""

    async def _search(self, query: str) -> tuple[list[ContentSource], str]:
        """Execute a single Tavily search and parse results."""
        try:
            response = await self._tavily.search(
                query=query,
                max_results=self._config.max_results,
                search_depth=self._config.search_depth,
                include_answer=False,
                include_images=False,
            )
        except (httpx.HTTPError, TimeoutError, OSError, ValueError, RuntimeError, TavilyInvalidAPIKeyError) as e:
            raise ContentSearchError(f"Tavily search failed: {e}") from e

        results = (response or {}).get("results") or []
        if not results:
            raise ContentSearchError("Tavily returned 0 results")

        sources: list[ContentSource] = []
        context_parts: list[str] = []
        for idx, r in enumerate(results):
            url = r.get("url")
            if not url:
                continue
            title = r.get("title")
            score = r.get("score")
            snippet = r.get("content") or r.get("raw_content") or ""
            if len(snippet) > 1200:
                snippet = snippet[:1200] + "..."

            sources.append(ContentSource(title=title, url=url, score=score))
            # Format context with source number for easy reference
            context_parts.append(
                "\n".join(
                    [
                        f"[Source {idx + 1}]",
                        f"Title: {title or 'N/A'}",
                        f"URL: {url}",
                        f"Content: {snippet}",
                    ]
                )
            )

        if not sources:
            raise ContentSearchError("Tavily results did not contain any URLs")

        return sources, "\n\n---\n\n".join(context_parts)

    async def _synthesize(
        self, question: str, context: str, sources: list[ContentSource]
    ) -> str:
        """Synthesize a well-structured answer using the enhanced system prompt."""
        system_prompt = self._get_system_prompt(sources)

        user_prompt = f"""## Question
{question}

## Web Search Context
{context}

---
Write a comprehensive, well-structured answer based on the context above.
Remember to use inline citations [1], [2], etc. for all factual claims."""

        try:
            start = time.perf_counter()
            resp = await self._llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
        except (httpx.HTTPError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            raise ContentSynthesisError(f"LLM synthesis failed: {e}") from e

        raw_content = getattr(resp, "content", None)
        content = _extract_llm_content(raw_content)
        if not content.strip():
            raise ContentSynthesisError("LLM returned empty content")

        latency_ms = (time.perf_counter() - start) * 1000.0
        self.last_trace = {
            "agent": "content_agent",
            "stage": "synthesis",
            "model": self._config.model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "raw_output": content,
            "parsed_output": content.strip(),
            "latency_ms": latency_ms,
            "sources_count": len(sources),
        }
        return content.strip()

    def _format_answer(
        self, answer_text: str, sources: list[ContentSource]
    ) -> str:
        """Format the final answer with numbered references section."""
        # Build numbered references section
        references = []
        for idx, source in enumerate(sources):
            title = source.title or "Untitled"
            url = source.url
            references.append(f"[{idx + 1}] {title} - {url}")

        references_block = "\n".join(references)

        return f"""{answer_text}

---

## References
{references_block}"""


