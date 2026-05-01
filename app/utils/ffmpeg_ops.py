"""FFprobe / FFmpeg subprocess helpers."""

import os
import subprocess
from typing import Callable, Optional, Tuple

import librosa

from ..config import settings
from ..exceptions import AudioProcessingError

from .temp_files import create_temp_file


def validate_audio_file_integrity(file_path: str) -> Tuple[bool, Optional[str]]:
    """Validate if an audio file is valid and can be processed."""
    try:
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            file_path,
        ]

        result = subprocess.run(
            probe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=settings.ffprobe_timeout_seconds,
        )

        if result.returncode != 0:
            probe_cmd_video = [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-of",
                "json",
                file_path,
            ]
            result_video = subprocess.run(
                probe_cmd_video,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=settings.ffprobe_timeout_seconds,
            )

            if result_video.returncode != 0:
                error_msg = result_video.stderr.decode("utf-8", errors="ignore")
                return False, f"File validation failed: {error_msg[:200]}"

        try:
            librosa.get_duration(path=file_path)
        except Exception as e:
            return False, f"File cannot be read as audio: {str(e)[:200]}"

        return True, None

    except subprocess.TimeoutExpired:
        return False, "File validation timed out"
    except FileNotFoundError:
        return True, None
    except Exception:
        return True, None


def convert_audio_file(
    input_path: str,
    output_path: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """Convert audio file to clean WAV format using ffmpeg."""
    try:
        if progress_callback:
            progress_callback(0.0, "Starting audio file conversion...")
        if output_path is None:
            output_path, _ = create_temp_file("wav")
        if progress_callback:
            progress_callback(10.0, "Extracting audio stream...")

        ffmpeg_cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(settings.target_sample_rate),
            "-ac",
            str(settings.audio_channels),
            "-y",
            output_path,
        ]

        if progress_callback:
            progress_callback(30.0, "Processing audio conversion...")

        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=settings.ffmpeg_conversion_timeout_seconds,
        )

        if progress_callback:
            progress_callback(80.0, "Finalizing converted audio...")

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown ffmpeg error"
            raise AudioProcessingError(
                message="Failed to convert audio file",
                original_error=error_msg,
                processing_step="convert_audio_file",
            )

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise AudioProcessingError(
                message="Conversion produced empty or missing output file",
                original_error="Output file verification failed",
                processing_step="convert_audio_file",
            )

        if progress_callback:
            progress_callback(100.0, "Audio conversion completed")
        return output_path

    except subprocess.TimeoutExpired:
        raise AudioProcessingError(
            message=f"Audio conversion timed out (exceeded {settings.ffmpeg_conversion_timeout_seconds} seconds)",
            original_error="FFmpeg conversion timeout",
            processing_step="convert_audio_file",
        )
    except FileNotFoundError:
        raise AudioProcessingError(
            message="FFmpeg not found. Please ensure ffmpeg is installed.",
            original_error="FFmpeg executable not found",
            processing_step="convert_audio_file",
        )
    except Exception as e:
        raise AudioProcessingError(
            message="Failed to convert audio file",
            original_error=str(e),
            processing_step="convert_audio_file",
        )


def convert_video_to_audio(
    input_path: str,
    output_path: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """Convert video or music file to audio format using ffmpeg."""
    try:
        if progress_callback:
            progress_callback(0.0, "Starting video to audio conversion...")
        if output_path is None:
            output_path, _ = create_temp_file("wav")
        if progress_callback:
            progress_callback(10.0, "Extracting audio from video...")

        ffmpeg_cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(settings.target_sample_rate),
            "-ac",
            str(settings.audio_channels),
            "-y",
            output_path,
        ]

        if progress_callback:
            progress_callback(30.0, "Processing video conversion...")

        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=settings.ffmpeg_conversion_timeout_seconds,
        )

        if progress_callback:
            progress_callback(80.0, "Finalizing audio extraction...")

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown ffmpeg error"
            raise AudioProcessingError(
                message="Failed to convert video/music file to audio",
                original_error=error_msg,
                processing_step="convert_video_to_audio",
            )

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise AudioProcessingError(
                message="Conversion produced empty or missing output file",
                original_error="Output file verification failed",
                processing_step="convert_video_to_audio",
            )

        if progress_callback:
            progress_callback(100.0, "Video to audio conversion completed")
        return output_path

    except subprocess.TimeoutExpired:
        raise AudioProcessingError(
            message=f"Video conversion timed out (exceeded {settings.ffmpeg_conversion_timeout_seconds} seconds)",
            original_error="FFmpeg conversion timeout",
            processing_step="convert_video_to_audio",
        )
    except FileNotFoundError:
        raise AudioProcessingError(
            message="FFmpeg not found. Please ensure ffmpeg is installed.",
            original_error="FFmpeg executable not found",
            processing_step="convert_video_to_audio",
        )
    except Exception as e:
        raise AudioProcessingError(
            message="Failed to convert video/music file to audio",
            original_error=str(e),
            processing_step="convert_video_to_audio",
        )
