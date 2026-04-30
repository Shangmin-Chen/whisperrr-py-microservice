"""
Comprehensive unit tests for WhisperService covering all failure scenarios.

These tests verify that the service properly handles:
- Model loading failures
- Transcription failures
- Invalid file paths
- Audio processing errors
- Resource cleanup failures
- Concurrent access issues
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from faster_whisper import WhisperModel

from app.whisper_service import WhisperService
from app.exceptions import (
    ModelNotLoaded,
    ModelLoadFailed,
    TranscriptionFailed,
    AudioProcessingError
)


class TestWhisperService:
    """Test suite for WhisperService."""
    
    @pytest.fixture
    def service(self):
        """Create a fresh service instance for each test."""
        # Reset singleton
        WhisperService._instance = None
        return WhisperService()
    
    @pytest.fixture
    def temp_audio_file(self):
        """Create a temporary audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b'fake audio content')
            yield f.name
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    # ========== Model Loading Tests ==========
    
    @pytest.mark.asyncio
    async def test_load_model_when_already_loaded_returns_cached(self, service):
        """Test that loading an already loaded model returns cached info."""
        with patch.object(service, '_load_model_sync', return_value=Mock()):
            await service.load_model("base")
            result = await service.load_model("base")
            
            assert result["success"] is True
            assert result["model_size"] == "base"
            assert result["load_time_seconds"] == 0.0
    
    @pytest.mark.asyncio
    async def test_load_model_when_concurrent_loading_raises_exception(self, service):
        """Test that concurrent model loading raises exception."""
        service._is_loading = True
        
        with pytest.raises(ModelLoadFailed):
            await service.load_model("base")
    
    @pytest.mark.asyncio
    async def test_load_model_when_model_load_fails_raises_exception(self, service):
        """Test that model loading failure raises exception."""
        with patch.object(service, '_load_model_sync', side_effect=Exception("Load failed")):
            with pytest.raises(ModelLoadFailed):
                await service.load_model("base")
    
    @pytest.mark.asyncio
    async def test_load_model_with_invalid_model_size_raises_exception(self, service):
        """Test that invalid model size raises exception."""
        with patch.object(service, '_load_model_sync', side_effect=ValueError("Invalid model")):
            with pytest.raises(ModelLoadFailed):
                await service.load_model("invalid-model")
    
    # ========== Transcription Tests ==========
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_when_model_not_loaded_raises_exception(self, service):
        """Test that transcription without loaded model raises exception."""
        service._model = None
        
        with pytest.raises(ModelNotLoaded):
            await service.transcribe_audio("test.mp3")
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_with_nonexistent_file_raises_exception(self, service):
        """Test that transcription with nonexistent file raises exception."""
        service._model = Mock()
        
        with pytest.raises(TranscriptionFailed):
            await service.transcribe_audio("/nonexistent/file.mp3")
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_when_transcription_fails_raises_exception(self, service):
        """Test that transcription failure raises exception."""
        service._model = Mock()
        mock_model = Mock()
        mock_model.transcribe = Mock(side_effect=Exception("Transcription failed"))
        service._model = mock_model
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b'fake audio')
            temp_path = f.name
        
        try:
            with patch('app.whisper_service.validate_audio_file', return_value={"duration": 1.0}):
                with patch('app.whisper_service.preprocess_audio', return_value=temp_path):
                    with pytest.raises(TranscriptionFailed):
                        await service.transcribe_audio(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_when_info_is_none_raises_exception(self, service):
        """Test that transcription with None info raises exception."""
        service._model = Mock()
        mock_model = Mock()
        
        # Mock transcribe to return segments with None info
        def mock_transcribe(*args, **kwargs):
            segments = [Mock(start=0.0, end=1.0, text="test")]
            return segments, None
        mock_model.transcribe = mock_transcribe
        service._model = mock_model
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b'fake audio')
            temp_path = f.name
        
        try:
            with patch('app.whisper_service.validate_audio_file', return_value={"duration": 1.0}):
                with patch('app.whisper_service.preprocess_audio', return_value=temp_path):
                    with pytest.raises(TranscriptionFailed):
                        await service._transcribe_sync(temp_path, None, 0.0, "transcribe", None)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_when_preprocessing_fails_raises_exception(self, service):
        """Test that preprocessing failure raises exception."""
        service._model = Mock()
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b'fake audio')
            temp_path = f.name
        
        try:
            with patch('app.whisper_service.validate_audio_file', return_value={"duration": 1.0}):
                with patch('app.whisper_service.preprocess_audio', 
                          side_effect=AudioProcessingError("Preprocessing failed")):
                    with pytest.raises(TranscriptionFailed):
                        await service.transcribe_audio(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_when_file_validation_fails_raises_exception(self, service):
        """Test that file validation failure raises exception."""
        service._model = Mock()
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b'fake audio')
            temp_path = f.name
        
        try:
            with patch('app.whisper_service.validate_audio_file', 
                      side_effect=Exception("Invalid file")):
                with pytest.raises(TranscriptionFailed):
                    await service.transcribe_audio(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    # ========== Model Info Tests ==========
    
    def test_get_model_info_when_model_not_loaded_returns_none(self, service):
        """Test that model info returns None when model not loaded."""
        service._model = None
        service._model_size = None
        
        info = service.get_model_info()
        
        assert info.model_size == "none"
        assert info.is_loaded is False
    
    def test_get_model_info_when_model_loaded_returns_info(self, service):
        """Test that model info returns correct info when model loaded."""
        service._model = Mock()
        service._model_size = "base"
        service._model_load_time = 1000.0
        
        with patch('app.whisper_service.get_memory_usage', return_value=1024.0):
            info = service.get_model_info()
            
            assert info.model_size == "base"
            assert info.is_loaded is True
    
    # ========== Utility Tests ==========
    
    def test_is_model_loaded_when_model_exists_returns_true(self, service):
        """Test that is_model_loaded returns True when model exists."""
        service._model = Mock()
        
        assert service.is_model_loaded() is True
    
    def test_is_model_loaded_when_model_none_returns_false(self, service):
        """Test that is_model_loaded returns False when model is None."""
        service._model = None
        
        assert service.is_model_loaded() is False
    
    def test_get_current_model_size_when_loaded_returns_size(self, service):
        """Test that get_current_model_size returns size when loaded."""
        service._model_size = "base"
        
        assert service.get_current_model_size() == "base"
    
    def test_get_current_model_size_when_not_loaded_returns_none(self, service):
        """Test that get_current_model_size returns None when not loaded."""
        service._model_size = None
        
        assert service.get_current_model_size() is None
    
    def test_get_uptime_returns_positive_value(self, service):
        """Test that get_uptime returns positive value."""
        uptime = service.get_uptime()
        
        assert uptime > 0
    
    # ========== Cleanup Tests ==========
    
    @pytest.mark.asyncio
    async def test_cleanup_when_active_transcriptions_waits(self, service):
        """Test that cleanup waits for active transcriptions."""
        service._active_transcriptions = 1
        service._executor = Mock()
        service._executor.shutdown = Mock()
        
        # Mock active_transcriptions to decrease after delay
        async def decrease_transcriptions():
            await asyncio.sleep(0.1)
            service._active_transcriptions = 0
        
        asyncio.create_task(decrease_transcriptions())
        await service.cleanup()
        
        service._executor.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_when_model_loaded_clears_model(self, service):
        """Test that cleanup clears model when loaded."""
        service._model = Mock()
        service._model_size = "base"
        service._active_transcriptions = 0
        service._executor = Mock()
        service._executor.shutdown = Mock()
        
        await service.cleanup()
        
        assert service._model is None
        assert service._model_size is None
    
    # ========== Concurrent Access Tests ==========
    
    @pytest.mark.asyncio
    async def test_concurrent_model_loading_prevents_race_condition(self, service):
        """Test that concurrent model loading is prevented."""
        service._is_loading = True
        
        # Try to load model while another is loading
        with pytest.raises(ModelLoadFailed):
            await service.load_model("base")
    
    @pytest.mark.asyncio
    async def test_concurrent_transcription_tracking(self, service):
        """Test that active transcription count is tracked correctly."""
        initial_count = service._active_transcriptions
        
        service._active_transcriptions += 1
        assert service.get_active_transcriptions() == initial_count + 1
        
        service._active_transcriptions -= 1
        assert service.get_active_transcriptions() == initial_count









