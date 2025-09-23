FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Graphviz
RUN apt-get update \
	&& apt-get install -y --no-install-recommends graphviz \
	&& rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Django project
COPY topology_service ./topology_service

# Create directory for base files
RUN mkdir -p /app/base_files

# Expose the port
EXPOSE 8001

# Run the Django development server
ENV DJANGO_DEBUG=0
ENV TOPOLOGY_PORT=8001
CMD ["sh", "-lc", "python topology_service/manage.py runserver 0.0.0.0:${TOPOLOGY_PORT}"]