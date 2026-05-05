"""Synchronous Faster Whisper model construction (runs in worker threads)."""

from faster_whisper import WhisperModel


def load_whisper_model_sync(model_size: str, device: str, compute_type: str) -> WhisperModel:
    """Instantiate and return a ``WhisperModel`` for the given size and backend settings."""
    return WhisperModel(model_size, device=device, compute_type=compute_type)
