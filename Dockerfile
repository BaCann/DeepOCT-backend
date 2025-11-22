
# STAGE 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install to user directory
RUN pip install --user --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Clean up pip cache
RUN rm -rf /root/.cache/pip

FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    netcat-openbsd \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Set PATH
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy application files
COPY wait-for-it.sh /app/
RUN chmod +x /app/wait-for-it.sh

COPY app/ /app/app/
COPY models/ /app/models/
COPY .env /app/.env

# Create uploads directory
RUN mkdir -p /app/uploads

# Clean up unnecessary files to reduce image size
RUN find /root/.local \( -type f -name '*.pyc' -o -name '*.pyo' \) -delete \
    && find /root/.local -type d -name '__pycache__' -delete \
    && find /root/.local -type d -name 'tests' -delete 2>/dev/null || true \
    && find /root/.local -type d -name 'test' -delete 2>/dev/null || true \
    && find /root/.local -name '*.dist-info' -exec sh -c 'rm -rf "$1"/RECORD "$1"/INSTALLER' _ {} \; \
    && du -sh /root/.local

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]