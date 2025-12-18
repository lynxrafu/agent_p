## agent_p (Story 1.1 skeleton)

### Running (Docker Compose)

- (Optional) Copy `env.example` to `.env` and fill values (Compose already loads `env.example`)
- Start:
  - `docker-compose up --build`

### Endpoints

- **POST** `/v1/agent/execute`
  - Body: `{"task": "..."}` (see `CLAUDE.md`)
  - Returns: HTTP 202 with `task_id` + `status`
- **GET** `/health`
  - Returns: `{"status":"healthy"}`

### Notes

- MongoDB access uses **PyMongo Async** (`AsyncMongoClient`). Motor is considered deprecated per the provided deprecation notice.
- Worker is an RQ worker consuming the `agent_tasks` queue (see `docker-compose.yml` `worker.command`).


