FROM python:3.11-slim

WORKDIR /app

# Install requirements (no external dependencies needed for this service)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY app.py .

# Create directory for base files
RUN mkdir -p /app/base_files

# Expose the port
EXPOSE 8001

# Run the application
CMD ["python", "app.py"]