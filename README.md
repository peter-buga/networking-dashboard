# Networking Dashboard

This project provides a comprehensive solution for monitoring and visualizing network traffic. It includes a Dockerized monitoring stack with Prometheus and Grafana, a topology visualization service, and a set of tools for handling and testing notifications.

## Features

- **Dockerized Monitoring Stack:** Easily spin up a local monitoring environment with Prometheus and Grafana using Docker Compose.
- **Topology Visualization:** A service that parses Mininet network topology files and generates visualizations.
- **Notification Handling:** A suite of Python tools for receiving, processing, and testing notifications.
- **Example Scenario:** A Mininet-based test scenario for demonstrating the notification framework in action.

## Getting Started

### Prerequisites

- Docker
- Docker Compose

### Quickstart

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd networking-dashboard
    ```

2.  **Start the monitoring stack:**
    ```bash
    cd docker_setup
    docker compose up -d
    ```

3.  **Access the services:**
    - **Grafana:** [http://localhost:3000](http://localhost:3000) (pre-configured with a "Networking Dashboard")
    - **Prometheus:** [http://localhost:9090](http://localhost:9090)
    - **Topology Service:** [http://localhost:8001](http://localhost:8001)

## Components

### Dockerized Monitoring Stack (`docker_setup`)

This directory contains the Docker Compose setup for the monitoring stack.

- **`docker-compose.yml`:** Defines the services for Grafana, Prometheus, and the topology visualizer.
- **`grafana/`:** Contains provisioning configurations for Grafana, including datasources and dashboards.
- **`prometheus/`:** Contains the Prometheus configuration file (`prometheus.yml`).

### Notification Framework (`app`)

This directory contains the Python tools and scenarios for the notification framework.

- **`json_receiver/`:** A Python library and examples for receiving and handling notifications sent over UDP.
  - `notification_receiver.py`: A library for reassembling fragmented JSON messages.
  - `multipart_json_udp_receiver.py`: An example receiver that uses the `notification_receiver` library.
- **`scenario_notification/`:** A Mininet-based test scenario that demonstrates the notification framework.
  - `testnet.py`: The Mininet script for creating the network topology.
  - `edge1.ini` & `edge2.ini`: configuration files for the edge nodes.

### Topology Visualization Service (`docker_setup/services/topology`)

This service provides an API for visualizing Mininet network topologies.

- **`app.py`:** The Flask application that provides the API endpoints.
- **`topology_parser.py`:** A script for parsing Mininet topology files.
- **`topology_visualizer.py`:** A script for generating topology visualizations.

**API Endpoints:**

- `GET /health`: Basic health check.
- `GET /topologyJson?file=<path-to-mininet-file>`: Parses a Mininet file and returns the topology in JSON format.
- `GET /topologyImage?file=<path-to-mininet-file>&width=<width>&height=<height>`: Generates and returns a PNG image of the topology.

## Test Scenario

The `app/scenario_notification` directory contains a test scenario for the notification framework. The scenario uses Mininet to create a network topology with two instances. The `testnet.py` script sets up the topology and starts the instances.

For a detailed explanation of the scenario and how to interact with it, please refer to the `app/scenario_notification/README.md` file.

## Forecasting

The `app/forecast` directory contains a set of scripts for time-series forecasting of network metrics. It uses the ARIMA model to predict future values of metrics collected from Prometheus.

- `forecasting_service.py`: The main service that orchestrates the forecasting process.
- `forecast_api.py`: A Flask API for serving the forecasts.
- `arima_forecaster.py`: The implementation of the ARIMA forecaster.
- `prometheus_query.py`: A script for querying metrics from Prometheus.
- `forecast_collector.py`: A script for collecting and storing forecasts.

## Testing the Setup

The `test-monitoring-setup.ps` script can be used to test the monitoring setup. It performs the following steps:

1.  Starts a `socat` tunnel to forward traffic from `localhost:19100` to the UDP receiver.
2.  Tests the metrics endpoint to ensure that it is accessible.
3.  Starts the Docker Compose services.
4.  Tests the Prometheus targets to ensure that they are being scraped correctly.
5.  Tests the Grafana endpoint to ensure that it is accessible.
