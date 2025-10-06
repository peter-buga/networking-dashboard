#!/bin/bash

echo "=== Network Monitoring Setup Test ==="
echo ""

echo "1. Starting socat tunnel (if not already running)..."
# Check if socat is already running
if ! pgrep -f "socat.*19100" > /dev/null; then
    echo "Starting socat tunnel: localhost:19100 -> 10.200.0.1:9100"
    sudo socat TCP4-LISTEN:19100,fork,reuseaddr TCP:10.200.0.1:9100 &
    SOCAT_PID=$!
    sleep 2
    echo "Socat started with PID: $SOCAT_PID"
else
    echo "Socat tunnel already running"
fi

echo ""
echo "2. Testing metrics endpoint..."
curl -s http://localhost:19100/metrics | head -20
if [ $? -eq 0 ]; then
    echo "✓ Metrics endpoint is accessible"
else
    echo "✗ Failed to access metrics endpoint"
    echo "Make sure your UDP receiver is running with metrics enabled"
fi

echo ""
echo "3. Starting Docker Compose services..."
cd /home/peti/networking-dashboard/docker_setup
docker-compose up -d

echo ""
echo "4. Waiting for services to start..."
sleep 10

echo ""
echo "5. Testing Prometheus targets..."
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.job=="udp-receiver") | {job: .job, health: .health, lastError: .lastError}'

echo ""
echo "6. Testing Grafana..."
echo "Grafana should be available at: http://localhost:3000"
echo "The Network Monitoring Dashboard should be automatically provisioned."

echo ""
echo "=== Setup Complete ==="
echo "Access points:"
echo "- Grafana: http://localhost:3000"
echo "- Prometheus: http://localhost:9090"
echo "- Metrics endpoint: http://localhost:19100/metrics"