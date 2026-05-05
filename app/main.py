import uvicorn

from .config import settings
from .factory import create_app
from .logging_setup import get_uvicorn_log_level

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.server_reload,
        log_level=get_uvicorn_log_level(),
    )
