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

# ============================================
# STAGE 2: Runtime
# ============================================
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    netcat-openbsd \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Set PATH and environment variables
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy wait-for-it script
COPY wait-for-it.sh /app/
RUN chmod +x /app/wait-for-it.sh

# Copy application files
COPY app/ /app/app/
COPY models/ /app/models/

# THAY ĐỔI - KHÔNG copy .env vào image (sẽ dùng env_file từ docker-compose)
# COPY .env /app/.env  # XÓA dòng này

# Create directories for uploads
RUN mkdir -p /app/uploads && \
    chmod 755 /app/uploads

# Clean up unnecessary files to reduce image size
RUN find /root/.local \( -type f -name '*.pyc' -o -name '*.pyo' \) -delete \
    && find /root/.local -type d -name '__pycache__' -delete \
    && find /root/.local -type d -name 'tests' -delete 2>/dev/null || true \
    && find /root/.local -type d -name 'test' -delete 2>/dev/null || true \
    && find /root/.local -name '*.dist-info' -exec sh -c 'rm -rf "$1"/RECORD "$1"/INSTALLER' _ {} \; \
    && du -sh /root/.local

# Expose port
EXPOSE 8000

# Health check - Cải tiến để kiểm tra cả DB connection
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]