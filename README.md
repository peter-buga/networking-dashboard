# networking-dashboard

Spin up a local monitoring stack using Prometheus for metrics collection and Grafana for visualization, with a synthetic load generator.

Quickstart
- Requirements: Docker and Docker Compose
- Start services:
	- `docker compose up -d`
- Open Grafana: http://localhost:3000
	- Dashboard: "Networking Dashboard"

Services
- Generator: http://localhost:8000/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

Notes
- Prometheus scrapes the generator for synthetic metrics.
- Grafana is pre-provisioned with datasources and a dashboard to visualize the metrics.