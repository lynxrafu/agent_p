from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import AsyncTavilyClient

from src.agents.base import BaseAgent
from src.models.agent_content import ContentAgentOutput, ContentSource


class ContentAgentError(RuntimeError):
    """Base error for ContentAgent failures."""

    stage: str = "unknown"


class ContentConfigError(ContentAgentError):
    stage = "unknown"

class ContentInputError(ContentAgentError):
    stage = "unknown"


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


class ContentAgent(BaseAgent):
    """ContentAgent: web search + grounded synthesis with citations."""

    def __init__(
        self,
        config: ContentAgentConfig,
        *,
        tavily_client: AsyncTavilyClient | None = None,
        llm: Any | None = None,
    ) -> None:
        if not config.google_api_key:
            raise ContentConfigError("GOOGLE_API_KEY is required for ContentAgent")
        if not config.tavily_api_key:
            raise ContentConfigError("TAVILY_API_KEY is required for ContentAgent")

        self._config = config
        self._tavily = tavily_client or AsyncTavilyClient(api_key=config.tavily_api_key)
        self._llm = llm or ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=config.google_api_key,
            temperature=0.2,
        )

    async def process(self, task: str) -> ContentAgentOutput:
        task = task.strip()
        if not task:
            raise ContentInputError("Task content cannot be empty")

        sources, context = await self._search(task)
        answer_text = await self._synthesize(task, context)

        answer = self._format_answer(answer_text, sources)
        return ContentAgentOutput(answer=answer, sources=sources, model=self._config.model)

    async def _search(self, query: str) -> tuple[list[ContentSource], str]:
        try:
            response = await self._tavily.search(
                query=query,
                max_results=self._config.max_results,
                search_depth=self._config.search_depth,
                include_answer=False,
                include_images=False,
            )
        except Exception as e:
            raise ContentSearchError(f"Tavily search failed: {e}") from e

        results = (response or {}).get("results") or []
        if not results:
            raise ContentSearchError("Tavily returned 0 results")

        sources: list[ContentSource] = []
        context_parts: list[str] = []
        for r in results:
            url = r.get("url")
            if not url:
                continue
            title = r.get("title")
            score = r.get("score")
            snippet = r.get("content") or r.get("raw_content") or ""
            if len(snippet) > 1200:
                snippet = snippet[:1200] + "..."

            sources.append(ContentSource(title=title, url=url, score=score))
            context_parts.append(
                "\n".join(
                    [
                        f"Title: {title or 'N/A'}",
                        f"URL: {url}",
                        f"Snippet: {snippet}",
                    ]
                )
            )

        if not sources:
            raise ContentSearchError("Tavily results did not contain any URLs")

        return sources, "\n\n---\n\n".join(context_parts)

    async def _synthesize(self, question: str, context: str) -> str:
        system_prompt = (
            "You are ContentAgent.\n"
            "Answer the user's question using ONLY the provided web search context.\n"
            "Do not use outside knowledge. Do not hallucinate.\n"
            "If the context is insufficient to answer, say so plainly."
        )
        user_prompt = f"Question:\n{question}\n\nWeb Search Context:\n{context}\n\nWrite the best possible answer."
        try:
            resp = await self._llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        except Exception as e:
            raise ContentSynthesisError(f"Gemini synthesis failed: {e}") from e

        content = getattr(resp, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ContentSynthesisError("Gemini returned empty content")
        return content.strip()

    def _format_answer(self, answer_text: str, sources: list[ContentSource]) -> str:
        urls = [s.url for s in sources if s.url]
        sources_block = "\n".join([f"- {u}" for u in urls])
        return f"Answer:\n{answer_text}\n\nSources:\n{sources_block}"


