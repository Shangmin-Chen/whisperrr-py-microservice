"""HTTP-level tests via TestClient (uploads, validation, job lifecycle)."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import require_whisper_model_loaded
from app.job_manager import job_manager
from app.models import TranscriptionResponse


@pytest.fixture(autouse=True)
def reset_job_registry():
    yield
    with job_manager._lock:
        job_manager._jobs.clear()


@pytest.fixture()
def api_client():
    """Stub startup (no GPU model load); skip Depends model gate for deterministic HTTP tests."""

    async def _skip_model_dependency():
        return None

    with patch(
        "app.whisper_service.whisper_service.load_model",
        AsyncMock(return_value={"success": True}),
    ):
        with patch(
            "app.whisper_service.whisper_service.cleanup", new_callable=AsyncMock
        ):
            from app.main import app as fastapi_application

            fastapi_application.dependency_overrides[
                require_whisper_model_loaded
            ] = _skip_model_dependency
            try:
                with TestClient(fastapi_application) as client:
                    yield client
            finally:
                fastapi_application.dependency_overrides.clear()


@pytest.fixture()
def mock_transcription_success():
    resp = TranscriptionResponse(
        text="stub",
        language="en",
        duration=0.5,
        segments=[],
        confidence_score=None,
        model_used="base",
        processing_time=0.01,
    )
    mock = AsyncMock(return_value=resp)
    with patch("app.whisper_service.whisper_service.transcribe_audio", mock):
        yield mock


class TestTranscribeHTTP:
    def test_transcribe_rejects_unknown_extension(self, api_client):
        async def validating_only(file_path, **kwargs):
            from app.utils import validate_audio_file

            validate_audio_file(file_path)

        with patch(
            "app.whisper_service.whisper_service.transcribe_audio",
            side_effect=validating_only,
        ):
            files = {"file": ("readme.txt", b"hello", "text/plain")}
            r = api_client.post("/transcribe", files=files)
            assert r.status_code == 400
            body = r.json()
            assert body.get("error_type") == "INVALID_AUDIO_FORMAT"

    def test_transcribe_ok_with_patched_inference(
        self, api_client, mock_transcription_success
    ):
        files = {"file": ("clip.mp3", b"ID3fakecontent", "audio/mpeg")}
        r = api_client.post("/transcribe", files=files)
        assert r.status_code == 200
        assert r.json()["text"] == "stub"
        mock_transcription_success.assert_called_once()


class TestJobLifecycleHTTP:
    def test_submit_then_progress_stays_pending_when_worker_mocked(self, api_client):
        with patch("app.routers.jobs.process_transcription_job", new=AsyncMock()):
            files = {"file": ("tiny.mp3", b"ID3x", "audio/mpeg")}
            r = api_client.post("/jobs/submit", files=files)
            assert r.status_code == 200
            payload = r.json()
            job_id = payload["job_id"]
            assert payload["status"] == "PENDING"

            prog = api_client.get(f"/jobs/{job_id}/progress")
            assert prog.status_code == 200
            body = prog.json()
            assert body["job_id"] == job_id
            assert body["status"] == "PENDING"

    def test_job_progress_404_unknown_id(self, api_client):
        unknown = str(uuid.uuid4())
        r = api_client.get(f"/jobs/{unknown}/progress")
        assert r.status_code == 404
