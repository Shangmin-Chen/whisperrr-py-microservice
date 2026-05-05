"""Device and compute-type selection for Faster Whisper / CTranslate2."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Settings


def detect_device() -> str:
    """Return ``cuda`` when CTranslate2 reports CUDA compute types, else ``cpu``."""
    try:
        import ctranslate2

        if ctranslate2.get_supported_compute_types("cuda"):
            return "cuda"
    except (ImportError, Exception):
        pass
    return "cpu"


def resolve_compute_type(device: str, settings: "Settings") -> str:
    """Pick a valid compute type for ``device`` using configured preference when allowed."""
    configured = settings.compute_type.lower()
    if device == "cuda":
        if configured in ("float16", "float32"):
            return configured
        return "float16"
    if configured in ("int8", "float32"):
        return configured
    return "int8"
