from fastapi import APIRouter

from ..models import HealthResponse
from ..whisper_service import whisper_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if whisper_service.is_model_loaded() else "degraded",
        model_loaded=whisper_service.is_model_loaded(),
        model_size=whisper_service.get_current_model_size(),
        uptime=round(whisper_service.get_uptime(), 2),
    )
