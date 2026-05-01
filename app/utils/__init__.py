"""Public utils API — re-exported for backward compatibility."""

import warnings

# Suppress librosa's PySoundFile fallback warning (it's expected behavior)
warnings.filterwarnings(
    "ignore", message="PySoundFile failed. Trying audioread instead."
)

# Re-export for tests patching `app.utils.settings`
from ..config import settings  # noqa: F401

from .audio_info import get_audio_info
from .ffmpeg_ops import (
    convert_audio_file,
    convert_video_to_audio,
    validate_audio_file_integrity,
)
from .files import (
    get_correlation_id,
    get_file_extension,
    safe_filename,
    validate_file_format,
    validate_file_size,
)
from .format_detection import detect_audio_format
from .memory import get_memory_usage
from .preprocess import preprocess_audio
from .temp_files import cleanup_temp_file, create_temp_file
from .validation import validate_audio_file

__all__ = [
    "settings",
    "cleanup_temp_file",
    "convert_audio_file",
    "convert_video_to_audio",
    "create_temp_file",
    "detect_audio_format",
    "get_audio_info",
    "get_correlation_id",
    "get_file_extension",
    "get_memory_usage",
    "preprocess_audio",
    "safe_filename",
    "validate_audio_file",
    "validate_audio_file_integrity",
    "validate_file_format",
    "validate_file_size",
]
