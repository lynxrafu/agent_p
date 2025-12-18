## Technical Test Requirements (Recruiter Spec)

This file mirrors the requirements from `CLAUDE.md` (the recruiter evaluation spec) so they are visible in the repository even if local agent tooling files are ignored.

Key expectations:

### 1) Peer Agent (Router)
- Use an agentic framework (LangChain/LangGraph) to route user tasks to specialist agents.
- Support at least two intent types (e.g., ContentAgent, CodeAgent).
- Routing can be keyword-based, but should be extensible and robust.
- Routing must be modular and easy to extend with new agents.

### 2) ContentAgent
- Must be able to access the web and return answers with source citations (URLs).
- Must use an LLM call to produce the final answer.

### 3) CodeAgent (future story in this repo)
- Must generate code snippets and a short explanation using an LLM call.

### 4) Business agents (future stories in this repo)
- Business Sense Discovery Agent: asks questions, maintains state, does not propose solutions during discovery.
- Diagnosis/Structuring Agent: builds a structured problem tree output.

### 5) Logging / Observability
- Persist tasks and logs (preferably MongoDB) using Pydantic schemas.
- README must justify logging choices and explain how logs can be queried/used.

### 6) API + DevOps + Tests
- FastAPI endpoint `POST /v1/agent/execute` (async task submission) + polling.
- Queue-backed worker processing.
- Dockerfile + docker-compose.
- GitHub Actions workflow, appspec + deploy scripts.
- At least 1–2 meaningful tests; README should explain how to expand test coverage.

### 7) README requirements
- Setup/run instructions.
- Architectural diagram (mandatory) showing agent routing + queue + logging + services.
- Explain LLM selection and prompt engineering practices.
- Provide production hardening recommendations (rate limiting, etc.).


