FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose ports for UDP and HTTP metrics
EXPOSE 6000/udp 8002

# Run the application
CMD ["python", "app.py"]