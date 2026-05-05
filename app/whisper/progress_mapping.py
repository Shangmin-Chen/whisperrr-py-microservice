"""Map inner transcription progress (0–100) to overall job progress bands."""

from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from ..config import Settings


def wrap_transcription_progress_callback(
    outer: Optional[Callable[[float, str], None]],
    settings: "Settings",
) -> Optional[Callable[[float, str], None]]:
    """If ``outer`` is set, return a callback that rescales 0–100% into the transcription band."""
    if outer is None:
        return None

    progress_range = settings.transcription_progress_max - settings.transcription_progress_min

    def inner(percent: float, message: str) -> None:
        mapped = settings.transcription_progress_min + (percent * progress_range / 100.0)
        outer(mapped, message)

    return inner


def segment_loop_progress(idx: int, settings: "Settings") -> float:
    """Progress value while iterating segments (whisper-internal scale)."""
    return min(
        settings.segment_progress_max,
        settings.segment_progress_base + (idx * settings.segment_progress_multiplier),
    )
