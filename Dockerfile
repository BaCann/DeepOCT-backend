# ============================================
# STAGE 1: Builder
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages to user directory
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

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    netcat-openbsd \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Set environment variables
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy wait-for-it script
# COPY wait-for-it.sh /app/
# RUN chmod +x /app/wait-for-it.sh

# Copy application code
COPY app/ /app/app/
COPY models/ /app/models/

# NOTE: .env files are NOT copied to image (security best practice)
# They will be provided via env_file in docker-compose or -e flags

# Create temp directory (optional - for temporary file operations)
RUN mkdir -p /tmp && chmod 1777 /tmp

# Clean up unnecessary files to reduce image size
RUN find /root/.local \( -type f -name '*.pyc' -o -name '*.pyo' \) -delete \
    && find /root/.local -type d -name '__pycache__' -delete \
    && find /root/.local -type d -name 'tests' -delete 2>/dev/null || true \
    && find /root/.local -type d -name 'test' -delete 2>/dev/null || true \
    && find /root/.local -name '*.dist-info' -exec sh -c 'rm -rf "$1"/RECORD "$1"/INSTALLER' _ {} \; \
    && echo "Python packages size:" && du -sh /root/.local

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]