# Whisperrr Python microservice

FastAPI service using Faster Whisper (CTranslate2) for audio transcription.

## Related repositories

- [whisperrr-frontend](https://github.com/) — React UI (replace with your remote)
- [whisperrr-backend](https://github.com/) — Spring Boot API (replace with your remote)

## Prerequisites

- Python 3.12
- FFmpeg

## Configuration

CORS origins must include your frontend and backend URLs. Run:

```bash
./setup-env.sh
```

This writes `.env` with `CORS_ORIGINS` (comma-separated).

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 5001
```

Health check: http://localhost:5001/health

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
