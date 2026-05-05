# Whisperrr Python microservice

FastAPI service using Faster Whisper (CTranslate2) for transcription.

Sibling packages in this repo: `whisperrr-frontend/`, `whisperrr-backend/`.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.12
- FFmpeg

## Configuration

CORS is mainly relevant if browsers call FastAPI directly. Optional `./setup-env.sh` writes **`CORS_ORIGINS`**. **`CORS_ALLOW_LOOPBACK_REGEX`** (maps to `cors_allow_loopback_regex`): default allows `localhost` / `127.0.0.1` / `[::1]` with any port; set to `false` if the API is publicly exposed without a gateway.

## Run locally

From this directory:

1. **Environment file (CORS)** — first time, or when frontend/backend ports change:

   ```bash
   ./setup-env.sh
   ```

2. **Sync dependencies** from `uv.lock`:

   ```bash
   uv sync --extra dev
   ```

3. **Start the server**:

   ```bash
   uv run uvicorn app.main:app --host 0.0.0.0 --port 5001
   ```

   Add `--reload` during development if you want auto-reload on code changes.

Health check: http://localhost:5001/health

### Uvicorn workers and async jobs

The service keeps **asynchronous transcription jobs in process memory** (`JobManager`). Use **a single Uvicorn worker** unless you introduce a shared job store. If you set `--workers`/`UVICORN_WORKERS` greater than 1, `/jobs/*` may break: each worker has its own job map, so polling can hit a different process and return 404. Before scaling out, add a **shared job store** (and usually session affinity or a single API tier) and route job creation and status checks through that store.

## Tests

```bash
uv run pytest
```

## Formatting

```bash
uv run black app/
```

## License

MIT — see [LICENSE](LICENSE).
