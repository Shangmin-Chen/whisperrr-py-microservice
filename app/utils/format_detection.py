"""Format sniffing via ffprobe and static file signatures."""

import json
import os
import subprocess
from typing import Optional

from .files import get_file_extension


def detect_audio_format(file_path: str) -> Optional[str]:
    """Detect audio/video format using file signature and ffprobe."""
    try:
        if not os.path.exists(file_path):
            return None
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return None
    except Exception:
        return None

    try:
        try:
            probe_cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                file_path,
            ]
            result = subprocess.run(
                probe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )

            if result.returncode == 0:
                probe_data = json.loads(result.stdout.decode("utf-8"))
                format_name = probe_data.get("format", {}).get("format_name", "").lower()

                if "m4a" in format_name or "mp4" in format_name:
                    ext = get_file_extension(file_path)
                    if ext == "m4a":
                        return "m4a"
                    elif ext in ["mp4", "mov", "m4v"]:
                        return ext
                    if "audio" in format_name.lower() or "aac" in format_name.lower():
                        return "m4a"
                    return "mp4"
                elif "matroska" in format_name:
                    ext = get_file_extension(file_path)
                    return ext if ext in ["mkv", "webm"] else "mkv"
                elif "avi" in format_name:
                    return "avi"
                elif "flv" in format_name:
                    return "flv"
                elif "wmv" in format_name:
                    return "wmv"
                elif "3gp" in format_name:
                    return "3gp"
        except (
            FileNotFoundError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            Exception,
        ):
            pass

        with open(file_path, "rb") as f:
            header32 = f.read(32)
        header = header32[:16]

        if header.startswith(b"ID3") or (len(header) >= 4 and header[1:4] == b"ID3"):
            return "mp3"
        elif header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WAVE":
            return "wav"
        elif header.startswith(b"OggS"):
            return "ogg"
        elif header.startswith(b"fLaC"):
            return "flac"
        elif len(header32) >= 12 and header32[4:8] == b"ftyp":
            if len(header32) >= 16 and (
                b"M4A" in header32[8:16] or b"qt" in header32[8:16]
            ):
                ext = get_file_extension(file_path)
                return "m4a" if ext == "m4a" else ("mov" if ext == "mov" else "mp4")
            ext = get_file_extension(file_path)
            if ext == "m4a":
                return "m4a"
            elif ext in ["mp4", "mov", "m4v", "3gp"]:
                return ext
        elif header.startswith(b"\x30\x26\xB2\x75"):
            return "wma"
        elif header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"AVI ":
            return "avi"
        elif header.startswith(b"\x1a\x45\xdf\xa3"):
            ext = get_file_extension(file_path)
            if ext in ["mkv", "webm"]:
                return ext
        elif header.startswith(b"FLV"):
            return "flv"
        elif header.startswith(b"\x30\x26\xB2\x75\x8E\x66\xCF\x11"):
            return "wmv"
        elif header.startswith(b"\xFF\xF1") or header.startswith(b"\xFF\xF9"):
            return "aac"

        return get_file_extension(file_path)

    except Exception:
        return get_file_extension(file_path)
