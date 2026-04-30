# Python service Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies including ffmpeg for video/audio conversion
# Install gosu for proper user switching in entrypoint
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /tmp/whisperrr_uploads && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /app /tmp/whisperrr_uploads /home/appuser/.cache

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh

# Change ownership
RUN chown -R appuser:appuser /app && \
    chmod +x /docker-entrypoint.sh

# Expose port
EXPOSE 5001

# Set environment variables for production and performance optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONOPTIMIZE=2
# Thread optimization for NumPy and OpenMP (will be set based on CPU count at runtime)
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4
# Uvicorn worker configuration (default to 4 workers for 4 CPUs)
ENV UVICORN_WORKERS=4

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5001/health || exit 1

# Set entrypoint (runs as root, then switches to appuser)
ENTRYPOINT ["/docker-entrypoint.sh"]
# Don't switch USER here - entrypoint handles it

# Run the application
# CRITICAL: Single worker mode is required for in-memory job manager
# Multi-worker requires shared state (Redis/database)
# Keepalive must be longer than backend read timeout
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 5001 --workers ${UVICORN_WORKERS:-1} --timeout-keep-alive ${UVICORN_TIMEOUT_KEEP_ALIVE:-65} --timeout-graceful-shutdown ${UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN:-30} --limit-concurrency 100 --backlog 2048"]
