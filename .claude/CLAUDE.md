# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

This is a containerized networking monitoring dashboard with 4 main services:

1. **Generator** (`services/generator/`) - Python service that generates synthetic Prometheus metrics (requests, temperature, latency) at configurable intervals
2. **Topology** (`services/topology/`) - Django service that parses Mininet topology files and provides:
   - `/health` - health check endpoint
   - `/topologyJson?file=...` - returns network topology as JSON (nodes, edges, IPs)
   - `/topologyImage?file=...&width=...&height=...` - renders topology as PNG using matplotlib/networkx
3. **Prometheus** - Scrapes metrics from generator service
4. **Grafana** - Visualizes metrics with pre-provisioned dashboards and datasources

## Common Commands

### Development
- Start all services: `docker compose up -d`
- Rebuild specific service: `docker compose build <service> && docker compose up -d <service>`
- View logs: `docker compose logs -f <service>`

### Service URLs
- Generator metrics: http://localhost:8000/metrics
- Topology service: http://localhost:8001
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Key Implementation Details

### Topology Service
- Built with Django, minimal settings in `topology_service/settings.py`
- Modular architecture with separate files:
  - `views.py` - HTTP route handlers only
  - `topology_parser.py` - Mininet file parsing logic using regex
  - `topology_visualizer.py` - NetworkX/matplotlib visualization logic
  - `topology_utils.py` - Helper functions for text positioning and collision detection
  - `config.py` - Environment-based configuration management
- Parses hosts, links (with interface names), and IP configurations from `.py` files in `/app/base_files/`
- Uses NetworkX for graph operations and matplotlib for PNG rendering with smart label positioning
- IP labels positioned below nodes with collision detection to prevent overlaps

### Configuration
- Topology visualization can be configured via environment variables
- See `services/topology/.env.example` for available settings:
  - Node size, font sizes, label distances
  - Image dimensions, DPI, format
  - Text styling and collision detection thresholds
- All settings have sensible defaults if not specified

### File Mounting
- `./base_files` is mounted to `/app/base_files` in topology container
- Use container paths (e.g., `/app/base_files/testnet.py`) when calling topology endpoints

### Dependencies
- Generator: `prometheus-client==0.20.0`
- Topology: `django>=4.2,<5.0`, `networkx==3.1`, `matplotlib>=3.7.0`