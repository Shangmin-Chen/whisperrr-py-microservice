"""Synchronous transcription steps: validation, preprocessing, and Whisper inference."""

from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import Settings
from ..exceptions import TranscriptionFailed
from ..utils import preprocess_audio, validate_audio_file
from .progress_mapping import segment_loop_progress


def validate_input_audio(
    file_path: str,
    progress_callback: Optional[Callable[[float, str], None]],
) -> Dict[str, Any]:
    """Validate audio at ``file_path``; return metadata dict or raise."""
    if progress_callback:
        progress_callback(0.0, "Validating file format...")
    file_info = validate_audio_file(file_path)
    if file_info is None:
        raise TranscriptionFailed(
            message="File validation returned None",
            original_error="validate_audio_file returned None",
            file_path=file_path,
        )
    return file_info


def preprocess_for_transcription(
    file_path: str,
    progress_callback: Optional[Callable[[float, str], None]],
) -> str:
    """Run FFmpeg preprocessing; return path to audio suitable for Whisper."""
    return preprocess_audio(file_path, progress_callback=progress_callback)


def transcribe_audio_sync(
    model: Any,
    file_path: str,
    language: Optional[str],
    temperature: float,
    task: str,
    progress_callback: Optional[Callable[[float, str], None]],
    settings: Settings,
) -> Dict[str, Any]:
    """Run Faster Whisper on ``file_path``; return a plain dict for response building."""
    options = {
        "beam_size": settings.beam_size,
        "temperature": temperature,
        "task": task,
    }

    if progress_callback:
        progress_callback(0.0, "Initializing Whisper model...")
        progress_callback(5.0, "Starting audio transcription...")

    segments_generator, info = model.transcribe(file_path, **options)
    segments = list(segments_generator)

    if info is None:
        raise TranscriptionFailed(
            message="Transcription returned None info object",
            original_error="info is None",
            file_path=file_path,
        )

    if progress_callback:
        progress_callback(10.0, "Processing audio segments...")

    segments_with_text: List[Tuple[Any, str]] = []
    for idx, seg in enumerate(segments):
        seg_text = getattr(seg, "text", "").strip()
        segments_with_text.append((seg, seg_text))

        if progress_callback and idx % 10 == 0:
            progress_callback(
                segment_loop_progress(idx, settings),
                f"Transcribed {idx + 1} segment(s)...",
            )

    if progress_callback:
        progress_callback(95.0, "Formatting transcription results...")

    try:
        language_value = info.language if hasattr(info, "language") else None
        language_prob_value = getattr(info, "language_probability", None)
    except Exception:
        language_value = None
        language_prob_value = None

    segments_dict_list: List[Dict[str, Any]] = []
    text_parts: List[str] = []

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
            "no_speech_prob": getattr(seg, "no_speech_prob", None),
        })

    full_text = " ".join(text_parts) if text_parts else ""

    result = {
        "text": full_text,
        "language": language_value,
        "language_probability": language_prob_value,
        "segments": segments_dict_list,
    }

    if progress_callback:
        progress_callback(100.0, "Transcription completed successfully")

    return result
