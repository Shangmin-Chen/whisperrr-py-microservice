# Whisperrr Python microservice

FastAPI service using Faster Whisper (CTranslate2) for transcription.

Sibling packages in this repo: `whisperrr-frontend/`, `whisperrr-backend/`.

## Prerequisites

- Python 3.12
- FFmpeg

## Configuration

CORS is mainly relevant if browsers call FastAPI directly. Optional `./setup-env.sh` writes **`CORS_ORIGINS`**. **`CORS_ALLOW_LOOPBACK_REGEX`** (maps to `cors_allow_loopback_regex`): default allows `localhost` / `127.0.0.1` / `[::1]` with any port; set to `false` if the API is publicly exposed without a gateway.

## Run locally

Use a **virtualenv** so dependencies do not touch your system Python.

1. **Environment file (CORS)** — first time, or when frontend/backend ports change:

   ```bash
   ./setup-env.sh
   ```

2. **Create and activate venv** (from this directory):

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate
   pip install -e '.[dev]'
   ```

3. **Start the server**:

   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 5001
   ```

   Add `--reload` during development if you want auto-reload on code changes.

Health check: http://localhost:5001/health

### Using uv instead

If you use [uv](https://github.com/astral-sh/uv), a manual `venv` + `activate` is optional; `uv` manages `.venv`:

```bash
uv sync --extra dev
uv run uvicorn app.main:app --host 0.0.0.0 --port 5001
```

### Uvicorn workers and async jobs

The service keeps **asynchronous transcription jobs in process memory** (`JobManager`). Use **a single Uvicorn worker** (the Docker image defaults `UVICORN_WORKERS=1`). If you set `--workers`/`UVICORN_WORKERS` greater than 1, `/jobs/*` may break: each worker has its own job map, so polling can hit a different process and return 404. Before scaling out, add a **shared job store** (and usually session affinity or a single API tier) and route job creation and status checks through that store.

## Tests

```bash
python -m pytest
```

## Formatting

```bash
black app/
```

## License

MIT — see [LICENSE](LICENSE).
