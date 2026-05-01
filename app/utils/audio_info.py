"""Read audio metadata via librosa."""

import os
from typing import Any, Dict

import librosa

from ..exceptions import AudioProcessingError


def get_audio_info(file_path: str) -> Dict[str, Any]:
    """Get audio file information."""
    try:
        duration = librosa.get_duration(path=file_path)
        sr = librosa.get_samplerate(file_path)
        file_size = os.path.getsize(file_path)

        return {
            "duration": duration,
            "sample_rate": sr,
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
        }
    except Exception as e:
        raise AudioProcessingError(
            message="Failed to get audio information",
            original_error=str(e),
            processing_step="get_audio_info",
        )
