"""WhisperService for managing Faster Whisper models and transcription."""

import asyncio
import time
import threading
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel

from .config import settings
from .models import (
    TranscriptionResponse,
    TranscriptionSegment,
    ModelInfoResponse
)
from .exceptions import (
    WhisperrrException,
    ModelNotLoaded,
    ModelLoadFailed,
    TranscriptionFailed,
    AudioProcessingError
)
from .utils import (
    preprocess_audio,
    validate_audio_file,
    get_memory_usage,
    cleanup_temp_file
)


class WhisperService:
    """Singleton service for managing Faster Whisper models and transcription."""
    
    _instance = None
    _lock = threading.Lock()
    _logger = logging.getLogger(__name__)
    
    def __new__(cls):
        """Ensure singleton pattern with thread-safe double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the WhisperService singleton instance."""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._model = None
        self._model_size = None
        self._model_load_time = None
        self._is_loading = False
        self._active_transcriptions = 0
        self._start_time = time.time()
        self._executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_transcriptions)
        
        # Detect device and compute type
        self._device = self._detect_device()
        self._compute_type = self._get_compute_type()
        
        # Log device and compute type for debugging
        self._logger.info(
            f"WhisperService initialized: device={self._device}, compute_type={self._compute_type}, "
            f"max_concurrent_transcriptions={settings.max_concurrent_transcriptions}"
        )
        
        # Model descriptions from config
        self._model_descriptions = settings.model_descriptions
        
        # Supported languages from config
        self._supported_languages = settings.supported_languages
        
    
    def _detect_device(self) -> str:
        """Detect available device (cuda or cpu)."""
        # Try to detect CUDA via ctranslate2 (used by faster-whisper)
        try:
            import ctranslate2
            # Check if CUDA is available via ctranslate2
            # ctranslate2.get_supported_compute_types("cuda") will return empty if CUDA not available
            if ctranslate2.get_supported_compute_types("cuda"):
                return "cuda"
        except (ImportError, Exception):
            pass
        
        # Default to CPU
        return "cpu"
    
    def _get_compute_type(self) -> str:
        """Get optimal compute type based on device and configuration."""
        # Check if compute type is explicitly set via config/environment
        configured_type = settings.compute_type.lower()
        
        # Validate compute type is appropriate for the device
        if self._device == "cuda":
            # GPU supports float16 and float32
            if configured_type in ["float16", "float32"]:
                return configured_type
            # Default to float16 for GPU (best performance/accuracy balance)
            return "float16"
        else:
            # CPU supports int8 and float32
            # int8 is faster but less accurate
            # float32 is slower but more accurate (better for production with sufficient resources)
            if configured_type in ["int8", "float32"]:
                return configured_type
            # Default to int8 for CPU (faster, but can be overridden via COMPUTE_TYPE env var)
            return "int8"
    
    async def load_model(self, model_size: str = None) -> dict:
        """Load a Faster Whisper model."""
        if model_size is None:
            model_size = settings.model_size
        
        # Check if model is already loaded
        if self._model is not None and self._model_size == model_size and not self._is_loading:
            return {
                "success": True,
                "model_size": model_size,
                "load_time_seconds": 0.0,
                "memory_usage_mb": get_memory_usage(),
                "message": f"Model {model_size} already loaded"
            }
        
        # Prevent concurrent loading
        if self._is_loading:
            raise ModelLoadFailed(
                message="Model is already being loaded",
                model_size=model_size
            )
        
        self._is_loading = True
        start_time = time.time()
        
        try:
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                self._executor,
                self._load_model_sync,
                model_size
            )
            
            self._model_size = model_size
            self._model_load_time = time.time()
            load_time = time.time() - start_time
            
            return {
                "success": True,
                "model_size": model_size,
                "load_time_seconds": round(load_time, 3),
                "memory_usage_mb": get_memory_usage(),
                "message": f"Model {model_size} loaded successfully"
            }
        
        except Exception as e:
            raise ModelLoadFailed(
                message=f"Failed to load model {model_size}",
                model_size=model_size,
                original_error=str(e)
            )
        
        finally:
            self._is_loading = False
    
    def _load_model_sync(self, model_size: str):
        """Synchronous model loading (runs in thread pool)."""
        # Load the Faster Whisper model
        model = WhisperModel(
            model_size,
            device=self._device,
            compute_type=self._compute_type
        )
        
        return model
    
    async def transcribe_audio(
        self,
        file_path: str,
        model_size: Optional[str] = None,
        language: Optional[str] = None,
        temperature: float = 0.0,
        task: str = "transcribe",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> TranscriptionResponse:
        """Transcribe audio file using Faster Whisper."""
        if self._model is None:
            raise ModelNotLoaded("No model is currently loaded")
        
        if model_size and model_size != self._model_size:
            # Load different model if requested
            await self.load_model(model_size)
        
        start_time = time.time()
        self._active_transcriptions += 1
        
        try:
            # Validate audio file
            if progress_callback:
                progress_callback(0.0, "Validating file format...")
            file_info = validate_audio_file(file_path)
            if file_info is None:
                raise TranscriptionFailed(
                    message="File validation returned None",
                    original_error="validate_audio_file returned None",
                    file_path=file_path
                )
            
            # Preprocess audio
            processed_file = None
            try:
                processed_file = preprocess_audio(file_path, progress_callback=progress_callback)
                
                # Run transcription in thread pool
                def transcription_callback(p: float, m: str):
                    if progress_callback:
                        # Map transcription progress (0-100%) to overall progress (40-100%)
                        progress_range = settings.transcription_progress_max - settings.transcription_progress_min
                        mapped_progress = settings.transcription_progress_min + (p * progress_range / 100.0)
                        progress_callback(mapped_progress, m)
                
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    self._transcribe_sync,
                    processed_file,
                    language,
                    temperature,
                    task,
                    transcription_callback
                )
                
                processing_time = time.time() - start_time
                
                # Convert result to response model
                response = self._create_transcription_response(
                    result, file_info, processing_time
                )
                
                return response
            
            finally:
                # Cleanup processed file
                if processed_file and settings.cleanup_temp_files:
                    cleanup_temp_file(processed_file)
        
        except WhisperrrException:
            # Re-raise Whisperrr exceptions as-is so they're handled properly
            raise
        except Exception as e:
            raise TranscriptionFailed(
                message="Transcription failed",
                original_error=str(e),
                file_path=file_path
            )
        
        finally:
            self._active_transcriptions -= 1
    
    def _transcribe_sync(
        self,
        file_path: str,
        language: Optional[str],
        temperature: float,
        task: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        """Synchronous transcription (runs in thread pool)."""
        # Prepare transcription options
        options = {
            "beam_size": settings.beam_size,
            "temperature": temperature,
            "task": settings.default_task
        }
        
        if progress_callback:
            progress_callback(0.0, "Initializing Whisper model...")
        if progress_callback:
            progress_callback(5.0, "Starting audio transcription...")
        
        segments_generator, info = self._model.transcribe(file_path, **options)
        segments = list(segments_generator)
        
        if info is None:
            raise TranscriptionFailed(
                message="Transcription returned None info object",
                original_error="info is None",
                file_path=file_path
            )
        
        if progress_callback:
            progress_callback(10.0, "Processing audio segments...")
        
        segments_with_text = []
        for idx, seg in enumerate(segments):
            seg_text = getattr(seg, "text", "").strip()
            segments_with_text.append((seg, seg_text))
            
            if progress_callback and idx % 10 == 0:
                segment_progress = min(
                    settings.segment_progress_max,
                    settings.segment_progress_base + (idx * settings.segment_progress_multiplier)
                )
                progress_callback(segment_progress, f"Transcribed {idx + 1} segment(s)...")
        
        if progress_callback:
            progress_callback(95.0, "Formatting transcription results...")
        
        # Build result dictionary compatible with existing response format
        if info is None:
            raise TranscriptionFailed(
                message="Transcription info object is None",
                original_error="info is None",
                file_path=file_path
            )
        
        # Safely extract language and language_probability
        try:
            language_value = info.language if hasattr(info, 'language') else None
            language_prob_value = getattr(info, 'language_probability', None)
        except Exception as e:
            language_value = None
            language_prob_value = None
        
        # Build segments list
        segments_dict_list = []
        text_parts = []
        
        for seg, seg_text in segments_with_text:
            if not seg_text:
                seg_text = getattr(seg, "text", "").strip()
            
            if seg_text:
                text_parts.append(seg_text)
            
            segments_dict_list.append({
                "start": getattr(seg, "start", 0.0),
                "end": getattr(seg, "end", 0.0),
                "text": seg_text,
                "avg_logprob": getattr(seg, "avg_logprob", None),
                "no_speech_prob": getattr(seg, "no_speech_prob", None)
            })
        
        full_text = " ".join(text_parts) if text_parts else ""
        
        result = {
            "text": full_text,
            "language": language_value,
            "language_probability": language_prob_value,
            "segments": segments_dict_list
        }
        
        if progress_callback:
            progress_callback(100.0, "Transcription completed successfully")
        
        return result
    
    def _create_transcription_response(
        self,
        whisper_result: Dict[str, Any],
        file_info: Optional[Dict[str, Any]],
        processing_time: float
    ) -> TranscriptionResponse:
        """Create TranscriptionResponse from Faster Whisper result."""
        if whisper_result is None:
            raise TranscriptionFailed(
                message="Transcription result is None",
                original_error="whisper_result is None",
                file_path="unknown"
            )
        
        # Extract text and segments
        text_from_result = whisper_result.get("text", "")
        segments_data = whisper_result.get("segments", [])
        
        # Extract segments
        segments = []
        if segments_data is None:
            segments_data = []
        
        for idx, segment in enumerate(segments_data):
            seg_text = segment.get("text", "").strip() if isinstance(segment, dict) else ""
            segments.append(TranscriptionSegment(
                start_time=segment.get("start", 0.0) if isinstance(segment, dict) else 0.0,
                end_time=segment.get("end", 0.0) if isinstance(segment, dict) else 0.0,
                text=seg_text,
                confidence=None  # Faster Whisper doesn't provide direct confidence scores
            ))
        
        # Calculate overall confidence (if available)
        confidence_score = None
        if "segments" in whisper_result and whisper_result["segments"]:
            # Use average of segment-level log probabilities if available
            confidences = [
                seg.get("avg_logprob", 0) for seg in whisper_result["segments"]
                if seg.get("avg_logprob") is not None
            ]
            if confidences:
                # Convert log probability to confidence score (approximate)
                # avg_logprob is typically negative, so we normalize it
                avg_logprob = sum(confidences) / len(confidences)
                # Convert to a 0-1 scale (rough approximation)
                confidence_score = max(0, min(1, (avg_logprob + 1) / 2))
        
        # Safely extract duration from file_info (handle None case)
        duration = 0.0
        if file_info is not None:
            duration = file_info.get("duration", 0.0)
            if duration is None:
                duration = 0.0
        
        # Use text from result, or build from segments if empty
        final_text = text_from_result.strip() if text_from_result else ""
        
        if not final_text and segments:
            segment_texts = [seg.text.strip() for seg in segments if hasattr(seg, 'text') and seg.text]
            final_text = " ".join(segment_texts)
        
        response = TranscriptionResponse(
            text=final_text,
            language=whisper_result.get("language"),
            duration=duration,
            segments=segments,
            confidence_score=confidence_score,
            model_used=self._model_size,
            processing_time=round(processing_time, 3)
        )
        
        return response
    
    def get_model_info(self) -> ModelInfoResponse:
        """Get information about the currently loaded model."""
        return ModelInfoResponse(
            model_size=self._model_size or "none",
            memory_usage_mb=get_memory_usage(),
            load_time_seconds=0.0 if not self._model_load_time else time.time() - self._model_load_time,
            supported_languages=self._supported_languages,
            is_loaded=self._model is not None,
            last_loaded=datetime.fromtimestamp(self._model_load_time) if self._model_load_time else None
        )
    
    
    def get_uptime(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self._start_time
    
    def get_active_transcriptions(self) -> int:
        """Get number of active transcriptions."""
        return self._active_transcriptions
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model is not None
    
    def get_current_model_size(self) -> Optional[str]:
        """Get currently loaded model size."""
        return self._model_size
    
    async def cleanup(self):
        """Cleanup resources and shutdown executor."""
        try:
            # Wait for active transcriptions to complete
            while self._active_transcriptions > 0:
                await asyncio.sleep(1)
            
            # Shutdown executor
            self._executor.shutdown(wait=True)
            
            # Clear model from memory
            if self._model is not None:
                del self._model
                self._model = None
                self._model_size = None
            
            # Force garbage collection
            import gc
            gc.collect()
        
        except Exception as e:
            pass


# Global service instance
whisper_service = WhisperService()
