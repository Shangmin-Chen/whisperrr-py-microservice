"""Librosa-heavy path: resample, normalize, write WAV for Whisper."""


import os
from typing import Callable, Optional

import librosa
import soundfile as sf

from ..config import settings
from ..exceptions import AudioProcessingError, WhisperrrException

from .audio_info import get_audio_info
from .ffmpeg_ops import (
    convert_audio_file,
    convert_video_to_audio,
    validate_audio_file_integrity,
)
from .files import get_file_extension
from .temp_files import cleanup_temp_file, create_temp_file


def preprocess_audio(
    file_path: str,
    target_sr: Optional[int] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """Preprocess audio file for Whisper with validation and conversion."""
    if target_sr is None:
        target_sr = settings.target_sample_rate

    converted_file = None
    original_file_path = file_path

    def update_progress(progress: float, message: str) -> None:
        if progress_callback:
            mapped_progress = progress * (settings.preprocessing_progress_max / 100.0)
            progress_callback(mapped_progress, message)

    try:
        update_progress(0.0, "Validating file...")

        file_extension = get_file_extension(file_path)
        is_video_format = file_extension in settings.formats_requiring_conversion_set
        is_audio_format = file_extension in [
            "mp3",
            "wav",
            "m4a",
            "flac",
            "ogg",
            "wma",
            "aac",
        ]

        update_progress(2.0, f"Detected file format: {file_extension}")

        if is_video_format:
            update_progress(5.0, f"Converting {file_extension.upper()} video to audio...")
            converted_file = convert_video_to_audio(
                file_path,
                progress_callback=lambda p, m: update_progress(5.0 + p * 0.25, m),
            )
            file_path = converted_file
            update_progress(30.0, "Video conversion completed")

        elif is_audio_format:
            update_progress(5.0, "Validating audio file integrity...")
            is_valid, error_msg = validate_audio_file_integrity(file_path)

            if not is_valid or error_msg:
                update_progress(10.0, "Audio file has issues, converting to clean format...")
                converted_file = convert_audio_file(
                    file_path,
                    progress_callback=lambda p, m: update_progress(10.0 + p * 0.20, m),
                )
                file_path = converted_file
                update_progress(30.0, "Audio conversion completed")
            else:
                update_progress(10.0, "Audio file is valid")

        update_progress(30.0, "Analyzing audio properties...")
        get_audio_info(file_path)

        update_progress(32.0, "Loading audio data...")
        attempted_load_recovery = False
        while True:
            try:
                audio, sr = librosa.load(file_path, sr=None)
                break
            except Exception as load_err:
                if attempted_load_recovery:
                    raise load_err
                attempted_load_recovery = True
                update_progress(35.0, "Audio loading failed, converting file...")
                previous_temp = converted_file
                new_wav = convert_audio_file(
                    file_path,
                    progress_callback=lambda p, m: update_progress(
                        35.0 + p * 0.05, m
                    ),
                )
                if (
                    previous_temp
                    and previous_temp != original_file_path
                    and previous_temp != new_wav
                    and os.path.exists(previous_temp)
                ):
                    try:
                        cleanup_temp_file(previous_temp)
                    except Exception:
                        pass
                converted_file = new_wav
                file_path = new_wav

        if converted_file and os.path.exists(converted_file) and converted_file != file_path:
            try:
                cleanup_temp_file(converted_file)
                converted_file = None
            except Exception:
                pass

        update_progress(36.0, "Processing audio...")
        if sr != target_sr:
            update_progress(37.0, f"Resampling audio from {sr}Hz to {target_sr}Hz...")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        update_progress(38.0, "Normalizing audio levels...")
        audio = librosa.util.normalize(audio)

        update_progress(39.0, "Saving preprocessed audio...")
        output_path, _ = create_temp_file("wav")
        sf.write(output_path, audio, sr, format="WAV", subtype="PCM_16")

        update_progress(
            40.0,
            "Audio preprocessing completed, ready for transcription",
        )
        return output_path

    except WhisperrrException:
        if converted_file and os.path.exists(converted_file):
            try:
                cleanup_temp_file(converted_file)
            except Exception:
                pass
        raise
    except Exception as e:
        if converted_file and os.path.exists(converted_file):
            try:
                cleanup_temp_file(converted_file)
            except Exception:
                pass
        raise AudioProcessingError(
            message="Failed to preprocess audio",
            original_error=str(e),
            processing_step="preprocess_audio",
        )
