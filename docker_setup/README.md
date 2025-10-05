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
- Topology (Django): http://localhost:8001
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

Notes
- Prometheus scrapes the generator for synthetic metrics.
- Grafana is pre-provisioned with datasources and a dashboard to visualize the metrics.

Topology Service
- Endpoints:
	- `GET /health` → basic health check
	- `GET /topologyJson?file=/app/base_files/testnet.py` → parse a Mininet file and return nodes, edges (with interfaces), and IPs.
	- `GET /topologyImage?file=/app/base_files/testnet.py&width=1200&height=800` → render a PNG of the topology.
- The compose file mounts `./base_files` to `/app/base_files` in the container. Use container paths for the `file` query parameter.
- To rebuild only this service: `docker compose build topology && docker compose up -d topology`