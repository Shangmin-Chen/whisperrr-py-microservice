"""WhisperService: async orchestration over Faster Whisper (delegates to ``app.whisper``)."""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from .config import settings
from .exceptions import (
    ModelLoadFailed,
    ModelNotLoaded,
    TranscriptionFailed,
    WhisperrrException,
)
from .models import ModelInfoResponse, TranscriptionResponse
from .utils import cleanup_temp_file, get_memory_usage
from .whisper.compute_selection import detect_device, resolve_compute_type
from .whisper.model_lifecycle import load_whisper_model_sync
from .whisper.progress_mapping import wrap_transcription_progress_callback
from .whisper.response_builder import build_transcription_response
from .whisper.transcription_pipeline import (
    preprocess_for_transcription,
    transcribe_audio_sync,
    validate_input_audio,
)


class WhisperService:
    """Singleton service coordinating model lifecycle and transcription."""

    _instance = None
    _lock = threading.Lock()
    _logger = logging.getLogger(__name__)

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._model = None
        self._model_size = None
        self._model_load_time = None
        self._is_loading = False
        self._active_transcriptions = 0
        self._start_time = time.time()
        self._executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_transcriptions)

        self._device = detect_device()
        self._compute_type = resolve_compute_type(self._device, settings)

        self._logger.info(
            "WhisperService initialized: device=%s, compute_type=%s, max_concurrent_transcriptions=%s",
            self._device,
            self._compute_type,
            settings.max_concurrent_transcriptions,
        )

        self._model_descriptions = settings.model_descriptions
        self._supported_languages = settings.supported_languages

    async def load_model(self, model_size: str = None) -> dict:
        if model_size is None:
            model_size = settings.model_size

        if self._model is not None and self._model_size == model_size and not self._is_loading:
            return {
                "success": True,
                "model_size": model_size,
                "load_time_seconds": 0.0,
                "memory_usage_mb": get_memory_usage(),
                "message": f"Model {model_size} already loaded",
            }

        if self._is_loading:
            raise ModelLoadFailed(
                message="Model is already being loaded",
                model_size=model_size,
            )

        self._is_loading = True
        start_time = time.time()

        try:
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(
                self._executor,
                load_whisper_model_sync,
                model_size,
                self._device,
                self._compute_type,
            )

            self._model_size = model_size
            self._model_load_time = time.time()
            load_time = time.time() - start_time

            return {
                "success": True,
                "model_size": model_size,
                "load_time_seconds": round(load_time, 3),
                "memory_usage_mb": get_memory_usage(),
                "message": f"Model {model_size} loaded successfully",
            }

        except Exception as e:
            raise ModelLoadFailed(
                message=f"Failed to load model {model_size}",
                model_size=model_size,
                original_error=str(e),
            )

        finally:
            self._is_loading = False

    async def transcribe_audio(
        self,
        file_path: str,
        model_size: Optional[str] = None,
        language: Optional[str] = None,
        temperature: float = 0.0,
        task: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> TranscriptionResponse:
        if self._model is None:
            raise ModelNotLoaded("No model is currently loaded")

        effective_task = task if task is not None else settings.default_task

        if model_size and model_size != self._model_size:
            await self.load_model(model_size)

        start_time = time.time()
        self._active_transcriptions += 1

        try:
            file_info = validate_input_audio(file_path, progress_callback)

            processed_file = None
            try:
                processed_file = preprocess_for_transcription(file_path, progress_callback)

                inner_progress = wrap_transcription_progress_callback(progress_callback, settings)

                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    self._transcribe_sync,
                    processed_file,
                    language,
                    temperature,
                    effective_task,
                    inner_progress,
                )

                processing_time = time.time() - start_time
                return build_transcription_response(
                    result, file_info, processing_time, self._model_size
                )

            finally:
                if processed_file and settings.cleanup_temp_files:
                    cleanup_temp_file(processed_file)

        except WhisperrrException:
            raise
        except Exception as e:
            raise TranscriptionFailed(
                message="Transcription failed",
                original_error=str(e),
                file_path=file_path,
            )

        finally:
            self._active_transcriptions -= 1

    def _transcribe_sync(
        self,
        file_path: str,
        language: Optional[str],
        temperature: float,
        task: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        return transcribe_audio_sync(
            self._model,
            file_path,
            language,
            temperature,
            task,
            progress_callback,
            settings,
        )

    def get_model_info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            model_size=self._model_size or "none",
            memory_usage_mb=get_memory_usage(),
            load_time_seconds=0.0 if not self._model_load_time else time.time() - self._model_load_time,
            supported_languages=self._supported_languages,
            is_loaded=self._model is not None,
            last_loaded=datetime.fromtimestamp(self._model_load_time, tz=timezone.utc)
            if self._model_load_time
            else None,
        )

    def get_uptime(self) -> float:
        return time.time() - self._start_time

    def get_active_transcriptions(self) -> int:
        return self._active_transcriptions

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def get_current_model_size(self) -> Optional[str]:
        return self._model_size

    async def cleanup(self):
        try:
            while self._active_transcriptions > 0:
                await asyncio.sleep(1)

            self._executor.shutdown(wait=True)

            if self._model is not None:
                del self._model
                self._model = None
                self._model_size = None

            import gc

            gc.collect()

        except Exception:
            pass


whisper_service = WhisperService()
