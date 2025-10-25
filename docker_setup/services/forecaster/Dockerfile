FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/srv/app

WORKDIR /srv/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY app/forecast/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir pyyaml

# Copy application code (preserve package layout)
COPY app /srv/app/app

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs

# Expose API port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command: run the API server module
CMD ["python", "-m", "app.forecast.forecast_api"]
