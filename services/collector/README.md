# Network Node Collector Service

## Overview

The collector service is designed to receive UDP notifications from network nodes and convert them into Prometheus metrics. Each network node sends JSON-formatted messages to a shared UDP endpoint, and the collector processes these messages to extract relevant metrics for monitoring and visualization.

## Architecture

```
Network Nodes → UDP (port 6000) → Collector Service → Prometheus Metrics (port 8002) → Prometheus → Grafana
```

## Supported Notification Types

### 1. Transaction Notifications
Contains information about network configuration changes:
```json
{
  "notif_hostname": "edge1",
  "notif_msg": {
    "push_level": "INFO",
    "transaction": {
      "add_ifaces": 10,
      "add_objects": 3,
      "add_streams": 3,
      "committed": "edge1.ini"
    }
  },
  "notif_seq": 0,
  "notif_tstamp": 1744037778.721275
}
```

**Generated Metrics:**
- `network_node_transaction_operations_total{hostname, operation_type}` - Counter of transaction operations

### 2. R2DTWO Status Notifications
Contains status information about node components:
```json
{
  "notif_hostname": "edge1",
  "notif_msg": {
    "push_level": "INFO",
    "r2dtwo": {
      "status": "startup completed"
    }
  },
  "notif_seq": 1,
  "notif_tstamp": 1744037778.7405703
}
```

**Generated Metrics:**
- `network_node_status_info{hostname}` - Info metric with current node status

### 3. Triggered Source Notifications
Contains detailed MEP (Maintenance End Point) and PRF (Packet Replication Function) data:
```json
{
  "notif_hostname": "edge1",
  "notif_msg": {
    "push_level": "INFO",
    "triggered_source": {
      "level": 3,
      "source": "c_edge1_tx",
      "target": "c_edge2_rx",
      "stream": "stream_uni",
      "mep": [...]
    }
  }
}
```

**Generated Metrics:**
- `network_triggered_source_level{hostname, source, target, stream}` - Gauge for trigger level
- `network_triggered_source_session{hostname, source, target, stream}` - Gauge for session ID
- `network_triggered_source_seq{hostname, source, target, stream}` - Gauge for sequence number
- `network_mep_octets_passed{hostname, mep_name, traffic_type}` - Gauge for MEP octet counters
- `network_mep_packets_passed{hostname, mep_name, traffic_type}` - Gauge for MEP packet counters
- `network_mep_mask_signal_state_info{hostname, mep_name}` - Info metric for MEP state
- `network_prf_octets_passed{hostname, prf_name}` - Gauge for PRF octet counters
- `network_prf_packets_passed{hostname, prf_name}` - Gauge for PRF packet counters
- `network_prf_pipeline_action_count{hostname, prf_name, pipeline_name}` - Gauge for pipeline actions
- `network_prf_pipeline_mask_state_info{hostname, prf_name, pipeline_name}` - Info metric for pipeline state

## Configuration

### Environment Variables

- `UDP_PORT` (default: 6000) - UDP port to listen on for notifications
- `UDP_HOST` (default: 0.0.0.0) - UDP host interface to bind to
- `METRICS_PORT` (default: 8002) - HTTP port for Prometheus metrics endpoint
- `LOG_LEVEL` (default: INFO) - Logging level (DEBUG, INFO, WARNING, ERROR)

### Docker Compose Configuration

The collector service is integrated into the docker-compose.yml:

```yaml
collector:
  build:
    context: ./services/collector
  container_name: nd-collector
  environment:
    - UDP_PORT=6000
    - UDP_HOST=0.0.0.0
    - METRICS_PORT=8002
    - LOG_LEVEL=INFO
  ports:
    - "6000:6000/udp"
    - "8002:8002"
  restart: unless-stopped
```

### Prometheus Configuration

The collector is added as a scrape target in `databases/prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'collector'
    static_configs:
      - targets: ['collector:8002']
```

## Usage

### Starting the Service

```bash
# Build and start the collector service
docker compose build collector
docker compose up collector -d

# Or start all services
docker compose up -d
```

### Sending Test Data

Use the included test script to send sample notifications:

```bash
python3 test_udp_client.py
```

### Viewing Metrics

1. **Raw Prometheus metrics:** http://localhost:8002/metrics
2. **Prometheus UI:** http://localhost:9090
3. **Grafana dashboards:** http://localhost:3000

### Example Queries

```promql
# Total notifications by hostname
sum by (hostname) (network_node_notifications_total)

# MEP traffic rates
rate(network_mep_octets_passed[5m])

# Transaction operations over time
increase(network_node_transaction_operations_total[1h])

# Current node statuses
network_node_status_info

# PRF pipeline states
network_prf_pipeline_mask_state_info
```

## Network Node Integration

To integrate your network nodes with this collector:

1. **Configure nodes to send UDP packets to:** `<collector-host>:6000`
2. **Ensure JSON format matches** the expected notification structure
3. **Include required fields:**
   - `notif_hostname` - Unique identifier for the node
   - `notif_msg` - Message content with notification type
   - `notif_seq` - Sequence number for tracking
   - `notif_tstamp` - Timestamp of the notification

## Monitoring and Troubleshooting

### Health Check
```bash
curl http://localhost:8002/health
```

### Service Logs
```bash
docker logs nd-collector
```

### Check Prometheus Targets
Visit http://localhost:9090/targets to verify the collector is being scraped successfully.

### Common Issues

1. **No metrics appearing:** Check if notifications are reaching the UDP port and if the JSON format is valid
2. **Connection refused:** Ensure the collector service is running and ports are properly exposed
3. **Invalid JSON:** Check node notification format matches expected structure

## Extending the Collector

To add support for new notification types:

1. Add new metric definitions in `app.py`
2. Extend the `NotificationProcessor` class with new processing methods
3. Update the `process_notification` method to handle the new type
4. Rebuild and restart the service