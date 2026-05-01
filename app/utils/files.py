"""Filename, extension, and trivial request helpers."""

import os
import re
import uuid
from pathlib import Path

from ..config import settings


def get_file_extension(filename: str) -> str:
    """Extract file extension from filename."""
    return Path(filename).suffix.lower().lstrip(".")


def validate_file_format(filename: str) -> bool:
    """Validate if file format is supported."""
    extension = get_file_extension(filename)
    return extension in settings.supported_formats_set


def validate_file_size(file_size: int) -> bool:
    """Validate if file size is within limits."""
    return file_size <= settings.max_file_size_bytes


def safe_filename(filename: str) -> str:
    """Create a safe filename by removing dangerous characters."""
    if not filename:
        return "file"

    safe = os.path.basename(filename)
    safe = re.sub(r"\.\./", "", safe)
    safe = re.sub(r"\.\.\\", "", safe)
    safe = re.sub(r"\.\.", "", safe)
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", safe)
    safe = re.sub(r"_+", "_", safe)
    safe = safe.strip(". _")

    if len(safe) > 255:
        safe = safe[:255]
    if not safe:
        safe = "file"
    return safe


def get_correlation_id() -> str:
    """Generate a unique correlation ID for request tracking."""
    return str(uuid.uuid4())
