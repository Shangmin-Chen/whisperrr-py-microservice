"""Build API ``TranscriptionResponse`` from raw Faster Whisper pipeline output."""

from typing import Any, Dict, List, Optional

from ..exceptions import TranscriptionFailed
from ..models import TranscriptionResponse, TranscriptionSegment


def build_transcription_response(
    whisper_result: Dict[str, Any],
    file_info: Optional[Dict[str, Any]],
    processing_time: float,
    model_size: Optional[str],
) -> TranscriptionResponse:
    """Assemble ``TranscriptionResponse`` from pipeline dict and file metadata."""
    if whisper_result is None:
        raise TranscriptionFailed(
            message="Transcription result is None",
            original_error="whisper_result is None",
            file_path="unknown",
        )

    text_from_result = whisper_result.get("text", "")
    segments_data = whisper_result.get("segments", [])
    if segments_data is None:
        segments_data = []

    segments: List[TranscriptionSegment] = []
    for segment in segments_data:
        seg_text = segment.get("text", "").strip() if isinstance(segment, dict) else ""
        segments.append(TranscriptionSegment(
            start_time=segment.get("start", 0.0) if isinstance(segment, dict) else 0.0,
            end_time=segment.get("end", 0.0) if isinstance(segment, dict) else 0.0,
            text=seg_text,
            confidence=None,
        ))

    confidence_score = None
    raw_segments = whisper_result.get("segments") or []
    if raw_segments:
        confidences = [
            seg.get("avg_logprob", 0)
            for seg in raw_segments
            if isinstance(seg, dict) and seg.get("avg_logprob") is not None
        ]
        if confidences:
            avg_logprob = sum(confidences) / len(confidences)
            confidence_score = max(0, min(1, (avg_logprob + 1) / 2))

    duration = 0.0
    if file_info is not None:
        duration = file_info.get("duration", 0.0)
        if duration is None:
            duration = 0.0

    final_text = text_from_result.strip() if text_from_result else ""

    if not final_text and segments:
        segment_texts = [
            seg.text.strip()
            for seg in segments
            if hasattr(seg, "text") and seg.text
        ]
        final_text = " ".join(segment_texts)

    return TranscriptionResponse(
        text=final_text,
        language=whisper_result.get("language"),
        duration=duration,
        segments=segments,
        confidence_score=confidence_score,
        model_used=model_size,
        processing_time=round(processing_time, 3),
    )
