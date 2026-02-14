
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --user --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

RUN rm -rf /root/.cache/pip


FROM python:3.11-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    postgresql-client \
    netcat-openbsd \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


COPY --from=builder /root/.local /root/.local


ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1


COPY app/ /app/app/
COPY models/ /app/models/

RUN mkdir -p /tmp && chmod 1777 /tmp


RUN find /root/.local \( -type f -name '*.pyc' -o -name '*.pyo' \) -delete \
    && find /root/.local -type d -name '__pycache__' -delete \
    && find /root/.local -type d -name 'tests' -delete 2>/dev/null || true \
    && find /root/.local -type d -name 'test' -delete 2>/dev/null || true \
    && find /root/.local -name '*.dist-info' -exec sh -c 'rm -rf "$1"/RECORD "$1"/INSTALLER' _ {} \; \
    && echo "Python packages size:" && du -sh /root/.local

EXPOSE 8000


HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]