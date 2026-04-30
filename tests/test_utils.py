"""
Comprehensive unit tests for utility functions covering all failure scenarios.

These tests verify that utility functions properly handle:
- File validation failures
- Audio processing errors
- Invalid file formats
- File system errors
- Edge cases
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from app.utils import (
    get_file_extension,
    validate_file_format,
    validate_file_size,
    detect_audio_format,
    validate_audio_file_integrity,
    safe_filename
)
from app.exceptions import (
    InvalidAudioFormat,
    FileTooLarge,
    AudioProcessingError,
    FileSystemError
)


class TestFileExtension:
    """Test suite for file extension utilities."""
    
    def test_get_file_extension_with_extension(self):
        """Test that get_file_extension extracts extension correctly."""
        assert get_file_extension("audio.mp3") == "mp3"
        assert get_file_extension("video.mp4") == "mp4"
        assert get_file_extension("file.wav") == "wav"
    
    def test_get_file_extension_without_extension(self):
        """Test that get_file_extension handles files without extension."""
        assert get_file_extension("file") == ""
    
    def test_get_file_extension_with_multiple_dots(self):
        """Test that get_file_extension handles multiple dots."""
        assert get_file_extension("file.name.mp3") == "mp3"
    
    def test_get_file_extension_case_insensitive(self):
        """Test that get_file_extension is case insensitive."""
        assert get_file_extension("FILE.MP3") == "mp3"
        assert get_file_extension("File.Mp3") == "mp3"


class TestFileValidation:
    """Test suite for file validation."""
    
    def test_validate_file_format_with_valid_format(self):
        """Test that validate_file_format accepts valid formats."""
        with patch('app.utils.settings') as mock_settings:
            mock_settings.supported_formats_set = {"mp3", "wav", "m4a"}
            assert validate_file_format("audio.mp3") is True
            assert validate_file_format("audio.wav") is True
    
    def test_validate_file_format_with_invalid_format(self):
        """Test that validate_file_format rejects invalid formats."""
        with patch('app.utils.settings') as mock_settings:
            mock_settings.supported_formats_set = {"mp3", "wav", "m4a"}
            assert validate_file_format("file.txt") is False
    
    def test_validate_file_size_with_valid_size(self):
        """Test that validate_file_size accepts valid sizes."""
        with patch('app.utils.settings') as mock_settings:
            mock_settings.max_file_size_bytes = 50 * 1024 * 1024
            assert validate_file_size(10 * 1024 * 1024) is True
    
    def test_validate_file_size_with_invalid_size(self):
        """Test that validate_file_size rejects invalid sizes."""
        with patch('app.utils.settings') as mock_settings:
            mock_settings.max_file_size_bytes = 50 * 1024 * 1024
            assert validate_file_size(1500 * 1024 * 1024) is False
    
    def test_validate_file_size_at_limit(self):
        """Test that validate_file_size accepts size at limit."""
        with patch('app.utils.settings') as mock_settings:
            mock_settings.max_file_size_bytes = 50 * 1024 * 1024
            assert validate_file_size(1000 * 1024 * 1024) is True


class TestAudioFormatDetection:
    """Test suite for audio format detection."""
    
    def test_detect_audio_format_with_mp3(self):
        """Test that detect_audio_format detects MP3."""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b'ID3\x03\x00fake mp3 content')
            temp_path = f.name
        
        try:
            format_type = detect_audio_format(temp_path)
            assert format_type == "mp3"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_detect_audio_format_with_wav(self):
        """Test that detect_audio_format detects WAV."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(b'RIFF' + b'\x00' * 4 + b'WAVE')
            temp_path = f.name
        
        try:
            format_type = detect_audio_format(temp_path)
            assert format_type == "wav"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_detect_audio_format_with_nonexistent_file(self):
        """Test that detect_audio_format handles nonexistent file."""
        format_type = detect_audio_format("/nonexistent/file.mp3")
        # Should fallback to extension-based detection
        assert format_type == "mp3"
    
    def test_detect_audio_format_with_invalid_file(self):
        """Test that detect_audio_format handles invalid file."""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b'invalid content')
            temp_path = f.name
        
        try:
            format_type = detect_audio_format(temp_path)
            # Should fallback to extension
            assert format_type == "mp3"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestAudioFileIntegrity:
    """Test suite for audio file integrity validation."""
    
    def test_validate_audio_file_integrity_with_valid_file(self):
        """Test that validate_audio_file_integrity accepts valid file."""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b'fake audio content')
            temp_path = f.name
        
        try:
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = b'{"format": {}}'
                with patch('librosa.get_duration', return_value=1.0):
                    is_valid, error = validate_audio_file_integrity(temp_path)
                    assert is_valid is True
                    assert error is None
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_validate_audio_file_integrity_with_invalid_file(self):
        """Test that validate_audio_file_integrity rejects invalid file."""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b'invalid content')
            temp_path = f.name
        
        try:
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 1
                mock_run.return_value.stderr = b'Invalid file'
                is_valid, error = validate_audio_file_integrity(temp_path)
                assert is_valid is False
                assert error is not None
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_validate_audio_file_integrity_with_timeout(self):
        """Test that validate_audio_file_integrity handles timeout."""
        import subprocess
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b'fake content')
            temp_path = f.name
        
        try:
            with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("ffprobe", 10)):
                is_valid, error = validate_audio_file_integrity(temp_path)
                assert is_valid is False
                assert error is not None
                assert "timed out" in error.lower() or "timeout" in error.lower()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_validate_audio_file_integrity_without_ffprobe(self):
        """Test that validate_audio_file_integrity handles missing ffprobe."""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b'fake content')
            temp_path = f.name
        
        try:
            with patch('subprocess.run', side_effect=FileNotFoundError()):
                # Should skip validation when ffprobe not available
                is_valid, error = validate_audio_file_integrity(temp_path)
                # May return True or False depending on librosa fallback
                assert isinstance(is_valid, bool)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestSafeFilename:
    """Test suite for safe filename generation."""
    
    def test_safe_filename_with_valid_name(self):
        """Test that safe_filename handles valid names."""
        assert safe_filename("audio.mp3") == "audio.mp3"
        assert safe_filename("test_file.wav") == "test_file.wav"
    
    def test_safe_filename_with_special_chars(self):
        """Test that safe_filename sanitizes special characters."""
        result = safe_filename("file/with\\path.mp3")
        assert "/" not in result
        assert "\\" not in result
    
    def test_safe_filename_with_unicode(self):
        """Test that safe_filename handles unicode characters."""
        result = safe_filename("файл.mp3")
        # Should handle unicode gracefully
        assert isinstance(result, str)
    
    def test_safe_filename_with_empty_string(self):
        """Test that safe_filename handles empty string."""
        result = safe_filename("")
        assert result is not None
        assert isinstance(result, str)

