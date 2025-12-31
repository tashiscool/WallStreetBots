# Production Dockerfile for WallStreetBots Trading System
# Multi-stage build for smaller final image

# =============================================================================
# Stage 1: Build dependencies
# =============================================================================
FROM python:3.12-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libc6-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Production image
# =============================================================================
FROM python:3.12-slim-bookworm AS production

# Build arguments
ARG WSB_GIT_SHA=unknown

# Labels
LABEL maintainer="WallStreetBots Team" \
      version="1.0" \
      description="Production WallStreetBots Trading Platform"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    DJANGO_SETTINGS_MODULE=backend.settings \
    WSB_GIT_SHA=${WSB_GIT_SHA} \
    # Gunicorn settings
    GUNICORN_WORKERS=4 \
    GUNICORN_THREADS=2 \
    GUNICORN_TIMEOUT=120 \
    GUNICORN_KEEPALIVE=5 \
    # Application settings
    PORT=8000

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 wsb && \
    useradd --uid 1000 --gid wsb --shell /bin/bash --create-home wsb

# Create application directories
WORKDIR /app
RUN mkdir -p /app/logs /app/staticfiles /app/.state /app/data && \
    chown -R wsb:wsb /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=wsb:wsb . /app/

# Copy and set entrypoint
COPY --chown=wsb:wsb docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy gunicorn config
COPY --chown=wsb:wsb docker/gunicorn.conf.py /app/gunicorn.conf.py

# Switch to non-root user
USER wsb

# Collect static files (done at build time for faster startup)
RUN python manage.py collectstatic --noinput --clear 2>/dev/null || true

# Expose port
EXPOSE 8000

# Health check using the dedicated liveness endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health/live/ || exit 1

# Entrypoint handles migrations and startup
ENTRYPOINT ["/entrypoint.sh"]

# Default command: gunicorn with config file
CMD ["gunicorn", "--config", "/app/gunicorn.conf.py", "backend.wsgi:application"]
