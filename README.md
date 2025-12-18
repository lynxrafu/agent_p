## agent_p

### Quick start (Docker Compose)

- (Optional) Copy `env.example` to `.env` and fill values (Compose already loads `env.example`)
- Start:
  - `docker-compose up --build`

### Local dev (Python venv)

Recommended to use **Python 3.12** (matches CI).

- Create + activate venv:
  - **Windows (PowerShell)**:
    - `py -3.12 -m venv .venv`
    - `.\.venv\Scripts\Activate.ps1`
  - **macOS/Linux**:
    - `python3.12 -m venv .venv`
    - `source .venv/bin/activate`
- Install deps:
  - `python -m pip install --upgrade pip`
  - `python -m pip install -r requirements.txt`
- Run tests:
  - `python -m pytest -q`

### API endpoints

- **POST** `/v1/agent/execute`
  - Body: `{"task": "..."}`
  - Returns: HTTP 202 with `task_id` + `status`
- **GET** `/v1/agent/tasks/{task_id}`
  - Returns: task status/result (and optional routing metadata when present)
- **GET** `/health`
  - Returns: `{"status":"healthy"}`

### Notes

- MongoDB access uses **PyMongo Async** (`AsyncMongoClient`). Motor is considered deprecated per the provided deprecation notice.
- Worker is an RQ worker consuming the `agent_tasks` queue (see `docker-compose.yml` `worker.command`).

### LLM selection, prompting, and observability

This project uses **Gemini** via **LangChain** (`langchain-google-genai`) for:

- **Routing (PeerAgent)**: low-variance classification (uses `temperature=0`), with **schema-validated output** via Pydantic.
- **Answer synthesis (ContentAgent)**: grounded generation from web-search context (uses a low temperature and strict “context-only” rules).

#### Prompt engineering practices used here

- **Clear role + objective**: each agent prompt states what the agent is and what it must produce.
- **Hard constraints**: routing must only output schema fields; ContentAgent must not use outside knowledge.
- **Few-shot examples**: short examples reduce ambiguity and improve consistency.
- **Format control**: we keep the model output “machine-parseable” where needed (routing) and keep final user answers structured (“Answer” + “Sources”).
- **Token control**: search snippets are capped to reduce prompt bloat and improve reliability.

#### What we persist to MongoDB

Each task is stored in MongoDB with:

- `status`: `queued | processing | completed | failed`
- `result`: Pydantic `TaskResult` (answer, sources, model, error/stage, debug)
- `route`, `route_confidence`, `route_rationale`: routing metadata for observability

#### Future agents (Epic 2 / Epic 3)

`CodeAgent` and `BusinessDiscoveryAgent` will be added as separate modules under `src/agents/`, each with:

- A dedicated prompt appropriate to its task type
- Pydantic input/output models
- Outputs persisted on the same task record for consistency and observability


