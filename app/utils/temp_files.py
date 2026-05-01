"""Temporary paths under configured upload directory."""

import os
import uuid
from typing import Optional, Tuple

from ..config import settings
from ..exceptions import FileSystemError


def create_temp_file(suffix: Optional[str] = None) -> Tuple[str, str]:
    """Create a temporary file and return path and filename."""
    try:
        unique_id = str(uuid.uuid4())
        if suffix:
            filename = f"whisperrr_{unique_id}.{suffix}"
        else:
            filename = f"whisperrr_{unique_id}"

        file_path = os.path.join(settings.upload_dir, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return file_path, filename
    except Exception as e:
        raise FileSystemError(
            message="Failed to create temporary file",
            operation="create_temp_file",
            original_error=str(e),
        )


def cleanup_temp_file(file_path: str) -> None:
    """Safely cleanup temporary file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass
