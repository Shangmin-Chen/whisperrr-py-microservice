"""FastAPI lifespan: model load and background tasks."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import settings
from .job_manager import job_manager
from .whisper_service import whisper_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = None
    try:
        await whisper_service.load_model(settings.model_size)

        async def periodic_cleanup():
            while True:
                try:
                    await asyncio.sleep(settings.job_cleanup_interval_seconds)
                    job_manager.cleanup_old_jobs()
                    logger.info("Completed periodic job cleanup")
                except asyncio.CancelledError:
                    logger.info("Job cleanup task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in job cleanup: {e}", exc_info=True)

        cleanup_task = asyncio.create_task(periodic_cleanup())
        logger.info(f"Started periodic job cleanup (interval: {settings.job_cleanup_interval_seconds}s)")

        yield
    finally:
        if cleanup_task:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        await whisper_service.cleanup()
