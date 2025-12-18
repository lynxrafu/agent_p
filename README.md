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


