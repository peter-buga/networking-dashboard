FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY app/forecast/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/forecast/*.py ./

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs

# Expose ports
EXPOSE 5000 19101

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command: run the API server
CMD ["python", "forecast_api.py"]
