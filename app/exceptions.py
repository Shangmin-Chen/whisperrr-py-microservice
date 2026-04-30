"""Custom exception classes for the Whisperrr FastAPI service."""

from typing import Optional, Dict, Any


class WhisperrrException(Exception):
    """Base exception class for all Whisperrr exceptions."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class InvalidAudioFormat(WhisperrrException):
    """Raised when audio format is not supported or corrupted."""
    
    def __init__(
        self,
        message: str = "Invalid or unsupported audio format",
        file_format: Optional[str] = None,
        supported_formats: Optional[list] = None
    ):
        details = {}
        if file_format:
            details["provided_format"] = file_format
        if supported_formats:
            details["supported_formats"] = supported_formats
        
        super().__init__(
            message=message,
            error_code="INVALID_AUDIO_FORMAT",
            details=details
        )


class FileTooLarge(WhisperrrException):
    """Raised when uploaded file exceeds size limits."""
    
    def __init__(
        self,
        message: str = "File size exceeds maximum allowed size",
        file_size: Optional[int] = None,
        max_size: Optional[int] = None
    ):
        details = {}
        if file_size:
            details["file_size_bytes"] = file_size
            details["file_size_mb"] = round(file_size / (1024 * 1024), 2)
        if max_size:
            details["max_size_bytes"] = max_size
            details["max_size_mb"] = round(max_size / (1024 * 1024), 2)
        
        super().__init__(
            message=message,
            error_code="FILE_TOO_LARGE",
            details=details
        )


class ModelNotLoaded(WhisperrrException):
    """Raised when Whisper model is not loaded."""
    
    def __init__(
        self,
        message: str = "Whisper model is not loaded",
        model_size: Optional[str] = None
    ):
        details = {}
        if model_size:
            details["requested_model"] = model_size
        
        super().__init__(
            message=message,
            error_code="MODEL_NOT_LOADED",
            details=details
        )


class TranscriptionFailed(WhisperrrException):
    """Raised when transcription process fails."""
    
    def __init__(
        self,
        message: str = "Transcription failed",
        original_error: Optional[str] = None,
        file_path: Optional[str] = None
    ):
        details = {}
        if original_error:
            details["original_error"] = str(original_error)
        if file_path:
            details["file_path"] = file_path
        
        super().__init__(
            message=message,
            error_code="TRANSCRIPTION_FAILED",
            details=details
        )


class ModelLoadFailed(WhisperrrException):
    """Raised when model loading fails."""
    
    def __init__(
        self,
        message: str = "Failed to load Whisper model",
        model_size: Optional[str] = None,
        original_error: Optional[str] = None
    ):
        details = {}
        if model_size:
            details["model_size"] = model_size
        if original_error:
            details["original_error"] = str(original_error)
        
        super().__init__(
            message=message,
            error_code="MODEL_LOAD_FAILED",
            details=details
        )


class AudioProcessingError(WhisperrrException):
    """Raised when audio preprocessing fails."""
    
    def __init__(
        self,
        message: str = "Audio processing failed",
        original_error: Optional[str] = None,
        processing_step: Optional[str] = None
    ):
        details = {}
        if original_error:
            details["original_error"] = str(original_error)
        if processing_step:
            details["processing_step"] = processing_step
        
        super().__init__(
            message=message,
            error_code="AUDIO_PROCESSING_ERROR",
            details=details
        )




class FileSystemError(WhisperrrException):
    """Raised when file system operations fail."""
    
    def __init__(
        self,
        message: str = "File system operation failed",
        operation: Optional[str] = None,
        file_path: Optional[str] = None,
        original_error: Optional[str] = None
    ):
        details = {}
        if operation:
            details["operation"] = operation
        if file_path:
            details["file_path"] = file_path
        if original_error:
            details["original_error"] = str(original_error)
        
        super().__init__(
            message=message,
            error_code="FILE_SYSTEM_ERROR",
            details=details
        )
