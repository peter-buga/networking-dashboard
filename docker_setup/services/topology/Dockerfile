FROM python:3.11-slim

# Install system dependencies for Graphviz
RUN apt-get update && apt-get install -y \
    graphviz \
    graphviz-dev \
    pkg-config \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py topology_parser.py topology_visualizer.py ./

# Create directory for temporary topology files
RUN mkdir -p /tmp/topologies

# Expose Flask port
EXPOSE 8001

# Run the Flask application
CMD ["python", "app.py"]
