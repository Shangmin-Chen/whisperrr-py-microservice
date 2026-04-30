"""Utility functions for file handling and audio processing."""

import os
import uuid
import subprocess
import json
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable
import librosa
import soundfile as sf

# Suppress librosa's PySoundFile fallback warning (it's expected behavior)
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

from .config import settings
from .exceptions import (
    InvalidAudioFormat,
    FileTooLarge,
    AudioProcessingError,
    FileSystemError
)


def get_file_extension(filename: str) -> str:
    """Extract file extension from filename."""
    return Path(filename).suffix.lower().lstrip('.')


def validate_file_format(filename: str) -> bool:
    """Validate if file format is supported."""
    extension = get_file_extension(filename)
    return extension in settings.supported_formats_set


def validate_file_size(file_size: int) -> bool:
    """Validate if file size is within limits."""
    return file_size <= settings.max_file_size_bytes


def detect_audio_format(file_path: str) -> Optional[str]:
    """Detect audio/video format using file signature and ffprobe."""
    try:
        # Ensure file is readable and has content
        if not os.path.exists(file_path):
            return None
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return None  # Empty file
    except Exception:
        return None
    
    try:
        # First try ffprobe for more accurate detection (especially for M4A)
        try:
            probe_cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                file_path
            ]
            result = subprocess.run(
                probe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10
            )
            
            if result.returncode == 0:
                probe_data = json.loads(result.stdout.decode('utf-8'))
                format_name = probe_data.get('format', {}).get('format_name', '').lower()
                
                # Map ffprobe format names to our extensions
                if 'm4a' in format_name or 'mp4' in format_name:
                    ext = get_file_extension(file_path)
                    # Prefer extension if it's m4a, otherwise use format_name hint
                    if ext == 'm4a':
                        return 'm4a'
                    elif ext in ['mp4', 'mov', 'm4v']:
                        return ext
                    # If format_name suggests audio, check if it's m4a
                    if 'audio' in format_name.lower() or 'aac' in format_name.lower():
                        return 'm4a'
                    return 'mp4'
                elif 'matroska' in format_name:
                    ext = get_file_extension(file_path)
                    return ext if ext in ['mkv', 'webm'] else 'mkv'
                elif 'avi' in format_name:
                    return 'avi'
                elif 'flv' in format_name:
                    return 'flv'
                elif 'wmv' in format_name:
                    return 'wmv'
                elif '3gp' in format_name:
                    return '3gp'
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            pass
        
        # Fallback to signature-based detection
        with open(file_path, 'rb') as f:
            header = f.read(16)
        
        # Check for common audio format signatures
        if header.startswith(b'ID3') or header[1:4] == b'ID3':
            return 'mp3'
        elif header.startswith(b'RIFF') and header[8:12] == b'WAVE':
            return 'wav'
        elif header.startswith(b'OggS'):
            return 'ogg'
        elif header.startswith(b'fLaC'):
            return 'flac'
        elif header[4:8] == b'ftyp':  # MP4/M4A container
            # Read more bytes to check for M4A signature
            f.seek(0)
            more_data = f.read(32)
            # Check for M4A-specific signatures
            if b'M4A' in more_data[8:16] or b'qt' in more_data[8:16]:
                ext = get_file_extension(file_path)
                return 'm4a' if ext == 'm4a' else ('mov' if ext == 'mov' else 'mp4')
            ext = get_file_extension(file_path)
            if ext == 'm4a':
                return 'm4a'
            elif ext in ['mp4', 'mov', 'm4v', '3gp']:
                return ext
        elif header.startswith(b'\x30\x26\xB2\x75'):
            return 'wma'
        elif header.startswith(b'RIFF') and header[8:12] == b'AVI ':
            return 'avi'
        elif header.startswith(b'\x1a\x45\xdf\xa3'):  # Matroska (MKV, WEBM)
            ext = get_file_extension(file_path)
            if ext in ['mkv', 'webm']:
                return ext
        elif header.startswith(b'FLV'):
            return 'flv'
        elif header.startswith(b'\x30\x26\xB2\x75\x8E\x66\xCF\x11'):
            return 'wmv'
        elif header.startswith(b'\xFF\xF1') or header.startswith(b'\xFF\xF9'):  # AAC ADTS
            return 'aac'
        
        # Fallback to extension-based detection
        return get_file_extension(file_path)
    
    except Exception as e:
        return get_file_extension(file_path)


def validate_audio_file_integrity(file_path: str) -> Tuple[bool, Optional[str]]:
    """Validate if an audio file is valid and can be processed."""
    try:
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",  # Check for at least one audio stream
            "-show_entries", "stream=codec_type",
            "-of", "json",
            file_path
        ]
        
        from .config import settings
        
        result = subprocess.run(
            probe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=settings.ffprobe_timeout_seconds
        )
        
        if result.returncode != 0:
            # Try without audio stream requirement (might be video-only, which is OK)
            probe_cmd_video = [
                "ffprobe",
                "-v", "error",
                "-show_format",
                "-of", "json",
                file_path
            ]
            from .config import settings
            
            result_video = subprocess.run(
                probe_cmd_video,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=settings.ffprobe_timeout_seconds
            )
            
            if result_video.returncode != 0:
                error_msg = result_video.stderr.decode('utf-8', errors='ignore')
                return False, f"File validation failed: {error_msg[:200]}"
        
        # Try to load with librosa to verify it's actually readable
        try:
            librosa.get_duration(path=file_path)
        except Exception as e:
            return False, f"File cannot be read as audio: {str(e)[:200]}"
        
        return True, None
    
    except subprocess.TimeoutExpired:
        return False, "File validation timed out"
    except FileNotFoundError:
        # FFprobe not available, skip validation
        return True, None
    except Exception as e:
        # Don't fail on validation errors
        return True, None


def create_temp_file(suffix: str = None) -> Tuple[str, str]:
    """Create a temporary file and return path and filename."""
    try:
        # Create unique filename
        unique_id = str(uuid.uuid4())
        if suffix:
            filename = f"whisperrr_{unique_id}.{suffix}"
        else:
            filename = f"whisperrr_{unique_id}"
        
        file_path = os.path.join(settings.upload_dir, filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        return file_path, filename
    
    except Exception as e:
        raise FileSystemError(
            message="Failed to create temporary file",
            operation="create_temp_file",
            original_error=str(e)
        )


def cleanup_temp_file(file_path: str) -> None:
    """Safely cleanup temporary file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        pass


def get_audio_info(file_path: str) -> Dict[str, Any]:
    """Get audio file information."""
    try:
        # Use librosa to get audio info
        duration = librosa.get_duration(path=file_path)
        sr = librosa.get_samplerate(file_path)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        return {
            "duration": duration,
            "sample_rate": sr,
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2)
        }
    
    except Exception as e:
        raise AudioProcessingError(
            message="Failed to get audio information",
            original_error=str(e),
            processing_step="get_audio_info"
        )


def convert_audio_file(input_path: str, output_path: Optional[str] = None, progress_callback: Optional[Callable[[float, str], None]] = None) -> str:
    """Convert audio file to clean WAV format using ffmpeg."""
    try:
        if progress_callback:
            progress_callback(0.0, "Starting audio file conversion...")
        
        # Generate output path if not provided
        if output_path is None:
            output_path, _ = create_temp_file("wav")
        
        if progress_callback:
            progress_callback(10.0, "Extracting audio stream...")
        
        from .config import settings
        
        # Use ffmpeg to convert audio to clean WAV
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", input_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit audio
            "-ar", str(settings.target_sample_rate),  # Sample rate from config
            "-ac", str(settings.audio_channels),  # Audio channels from config
            "-y",  # Overwrite output
            output_path
        ]
        
        if progress_callback:
            progress_callback(30.0, "Processing audio conversion...")
        
        # Run ffmpeg conversion
        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=settings.ffmpeg_conversion_timeout_seconds
        )
        
        if progress_callback:
            progress_callback(80.0, "Finalizing converted audio...")
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown ffmpeg error"
            raise AudioProcessingError(
                message="Failed to convert audio file",
                original_error=error_msg,
                processing_step="convert_audio_file"
            )
        
        # Verify output file exists and has content
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise AudioProcessingError(
                message="Conversion produced empty or missing output file",
                original_error="Output file verification failed",
                processing_step="convert_audio_file"
            )
        
        if progress_callback:
            progress_callback(100.0, "Audio conversion completed")
        
        return output_path
    
    except subprocess.TimeoutExpired:
        from .config import settings
        raise AudioProcessingError(
            message=f"Audio conversion timed out (exceeded {settings.ffmpeg_conversion_timeout_seconds} seconds)",
            original_error="FFmpeg conversion timeout",
            processing_step="convert_audio_file"
        )
    except FileNotFoundError:
        raise AudioProcessingError(
            message="FFmpeg not found. Please ensure ffmpeg is installed.",
            original_error="FFmpeg executable not found",
            processing_step="convert_audio_file"
        )
    except Exception as e:
        raise AudioProcessingError(
            message="Failed to convert audio file",
            original_error=str(e),
            processing_step="convert_audio_file"
        )


def convert_video_to_audio(input_path: str, output_path: Optional[str] = None, progress_callback: Optional[Callable[[float, str], None]] = None) -> str:
    """Convert video or music file to audio format using ffmpeg."""
    try:
        if progress_callback:
            progress_callback(0.0, "Starting video to audio conversion...")
        
        # Generate output path if not provided
        if output_path is None:
            output_path, _ = create_temp_file("wav")
        
        if progress_callback:
            progress_callback(10.0, "Extracting audio from video...")
        
        from .config import settings
        
        # Extract audio using ffmpeg
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", input_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit audio
            "-ar", str(settings.target_sample_rate),  # Sample rate from config
            "-ac", str(settings.audio_channels),  # Audio channels from config
            "-y",  # Overwrite output
            output_path
        ]
        
        if progress_callback:
            progress_callback(30.0, "Processing video conversion...")
        
        # Run ffmpeg conversion
        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=settings.ffmpeg_conversion_timeout_seconds
        )
        
        if progress_callback:
            progress_callback(80.0, "Finalizing audio extraction...")
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown ffmpeg error"
            raise AudioProcessingError(
                message="Failed to convert video/music file to audio",
                original_error=error_msg,
                processing_step="convert_video_to_audio"
            )
        
        # Verify output file exists and has content
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise AudioProcessingError(
                message="Conversion produced empty or missing output file",
                original_error="Output file verification failed",
                processing_step="convert_video_to_audio"
            )
        
        if progress_callback:
            progress_callback(100.0, "Video to audio conversion completed")
        
        return output_path
    
    except subprocess.TimeoutExpired:
        from .config import settings
        raise AudioProcessingError(
            message=f"Video conversion timed out (exceeded {settings.ffmpeg_conversion_timeout_seconds} seconds)",
            original_error="FFmpeg conversion timeout",
            processing_step="convert_video_to_audio"
        )
    except FileNotFoundError:
        raise AudioProcessingError(
            message="FFmpeg not found. Please ensure ffmpeg is installed.",
            original_error="FFmpeg executable not found",
            processing_step="convert_video_to_audio"
        )
    except Exception as e:
        raise AudioProcessingError(
            message="Failed to convert video/music file to audio",
            original_error=str(e),
            processing_step="convert_video_to_audio"
        )


def preprocess_audio(file_path: str, target_sr: Optional[int] = None, progress_callback: Optional[Callable[[float, str], None]] = None) -> str:
    """Preprocess audio file for Whisper with validation and conversion."""
    from .config import settings
    
    if target_sr is None:
        target_sr = settings.target_sample_rate
    
    converted_file = None
    original_file_path = file_path
    
    def update_progress(progress: float, message: str):
        """Helper to update progress within preprocessing stages (0-40%)"""
        if progress_callback:
            # Map preprocessing progress to 0-40% range
            mapped_progress = progress * (settings.preprocessing_progress_max / 100.0)
            progress_callback(mapped_progress, message)
    
    try:
        update_progress(0.0, "Validating file...")
        
        # Check if file requires conversion (video/music formats)
        file_extension = get_file_extension(file_path)
        is_video_format = file_extension in settings.formats_requiring_conversion_set
        is_audio_format = file_extension in ['mp3', 'wav', 'm4a', 'flac', 'ogg', 'wma', 'aac']
        
        update_progress(2.0, f"Detected file format: {file_extension}")
        
        # For video formats: always convert
        if is_video_format:
            update_progress(5.0, f"Converting {file_extension.upper()} video to audio...")
            converted_file = convert_video_to_audio(file_path, progress_callback=lambda p, m: update_progress(5.0 + p * 0.25, m))
            file_path = converted_file
            update_progress(30.0, "Video conversion completed")
        
        # For audio formats: validate integrity and convert if problematic
        elif is_audio_format:
            update_progress(5.0, "Validating audio file integrity...")
            is_valid, error_msg = validate_audio_file_integrity(file_path)
            
            if not is_valid or error_msg:
                update_progress(10.0, f"Audio file has issues, converting to clean format...")
                converted_file = convert_audio_file(file_path, progress_callback=lambda p, m: update_progress(10.0 + p * 0.20, m))
                file_path = converted_file
                update_progress(30.0, "Audio conversion completed")
            else:
                update_progress(10.0, "Audio file is valid")
        
        # Get audio info
        update_progress(30.0, "Analyzing audio properties...")
        audio_info = get_audio_info(file_path)
        
        update_progress(32.0, "Loading audio data...")
        # Load audio
        try:
            audio, sr = librosa.load(file_path, sr=None)
        except Exception as e:
            # If loading fails, try converting the file
            if not converted_file:  # Only convert if we haven't already
                update_progress(35.0, "Audio loading failed, converting file...")
                converted_file = convert_audio_file(file_path, progress_callback=lambda p, m: update_progress(35.0 + p * 0.05, m))
                file_path = converted_file
        audio, sr = librosa.load(file_path, sr=None)
        
        # Cleanup converted file after loading (no longer needed)
        if converted_file and os.path.exists(converted_file) and converted_file != file_path:
            try:
                cleanup_temp_file(converted_file)
                converted_file = None  # Mark as cleaned up
            except Exception as e:
                pass
        
        update_progress(36.0, "Processing audio...")
        # Resample if necessary
        if sr != target_sr:
            update_progress(37.0, f"Resampling audio from {sr}Hz to {target_sr}Hz...")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        # Normalize audio
        update_progress(38.0, "Normalizing audio levels...")
        audio = librosa.util.normalize(audio)
        
        # Create output file
        update_progress(39.0, "Saving preprocessed audio...")
        output_path, _ = create_temp_file("wav")
        
        # Save as WAV (Whisper prefers WAV format)
        sf.write(output_path, audio, sr, format='WAV', subtype='PCM_16')
        
        update_progress(40.0, "Audio preprocessing completed, ready for transcription")
        return output_path
    
    except Exception as e:
        # Cleanup converted file on error
        if converted_file and os.path.exists(converted_file):
            try:
                cleanup_temp_file(converted_file)
            except Exception:
                pass
        raise AudioProcessingError(
            message="Failed to preprocess audio",
            original_error=str(e),
            processing_step="preprocess_audio"
        )


def validate_audio_file(file_path: str) -> Dict[str, Any]:
    """Comprehensive audio file validation."""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileSystemError(
                message="File does not exist",
                file_path=file_path
            )
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Validate file size
        if not validate_file_size(file_size):
            raise FileTooLarge(
                file_size=file_size,
                max_size=settings.max_file_size_bytes
            )
        
        # Detect format
        detected_format = detect_audio_format(file_path)
        
        # Validate format - check detected format first, then extension
        extension = get_file_extension(file_path)
        format_to_check = detected_format or extension
        
        if not format_to_check or format_to_check not in settings.supported_formats_set:
            raise InvalidAudioFormat(
                file_format=format_to_check or "unknown",
                supported_formats=settings.supported_formats
            )
        
        # Try to get audio info (this will fail if file is corrupted)
        # For video files, skip detailed audio info since we'll convert them anyway
        is_video_format = detected_format in settings.formats_requiring_conversion_set
        
        if is_video_format:
            # For video files, just return basic info without trying to read audio
            # The actual audio extraction will happen during preprocessing
            return {
                "valid": True,
                "format": detected_format,
                "file_size": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "duration": None,  # Will be determined after conversion
                "sample_rate": None  # Will be determined after conversion
            }
        else:
            # For audio files, try to get detailed info
            try:
                audio_info = get_audio_info(file_path)
                return {
                    "valid": True,
                    "format": detected_format,
                    "file_size": file_size,
                    "file_size_mb": round(file_size / (1024 * 1024), 2),
                    "duration": audio_info.get("duration"),
                    "sample_rate": audio_info.get("sample_rate")
                }
            except Exception as e:
                # If we can't read the audio file, it might be corrupted
                # But don't fail here - let preprocessing handle conversion
                return {
                    "valid": True,
                    "format": detected_format,
                    "file_size": file_size,
                    "file_size_mb": round(file_size / (1024 * 1024), 2),
                    "duration": None,  # Will be determined during preprocessing
                    "sample_rate": None  # Will be determined during preprocessing
                }
    
    except (InvalidAudioFormat, FileTooLarge, FileSystemError):
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        raise AudioProcessingError(
            message="Audio file validation failed",
            original_error=str(e),
            processing_step="validate_audio_file"
        )


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return round(memory_info.rss / (1024 * 1024), 2)
    except ImportError:
        # psutil not available, return 0
        return 0.0
    except Exception:
        return 0.0




def safe_filename(filename: str) -> str:
    """Create a safe filename by removing dangerous characters."""
    import re
    
    if not filename:
        return "file"
    
    # Remove path separators and dangerous characters
    safe = os.path.basename(filename)
    
    # Remove path traversal patterns
    safe = re.sub(r'\.\./', '', safe)
    safe = re.sub(r'\.\.\\', '', safe)
    safe = re.sub(r'\.\.', '', safe)
    
    # Remove or replace unsafe characters (keep only alphanumeric, dot, dash, underscore)
    safe = re.sub(r'[^a-zA-Z0-9._-]', '_', safe)
    
    # Remove multiple consecutive underscores
    safe = re.sub(r'_+', '_', safe)
    
    # Remove leading/trailing dots, underscores, and spaces
    safe = safe.strip('. _')
    
    # Limit length
    if len(safe) > 255:
        safe = safe[:255]
    
    # Ensure it's not empty after sanitization
    if not safe:
        safe = "file"
    
    return safe


def get_correlation_id() -> str:
    """Generate a unique correlation ID for request tracking."""
    return str(uuid.uuid4())




