# Network Nodes Grafana Dashboard

## Overview

The Network Nodes Monitoring dashboard provides comprehensive visualization of network node metrics collected via UDP notifications. The dashboard includes multiple panels organized to show different aspects of network node performance and status.

## Dashboard Features

### Top Row - Overview Statistics

1. **Active Network Nodes** - Shows the count of nodes currently sending notifications
2. **Total Notifications (Last 5m)** - Sum of all notifications received in the past 5 minutes
3. **Transaction Operations (Last 5m)** - Sum of all transaction operations in the past 5 minutes
4. **Node Status Overview** - Table showing current status of each node

### Template Variables

The dashboard includes two template variables for filtering:

- **hostname** - Filter by specific network nodes (supports multi-select and "All")
- **mep_name** - Filter by specific MEP (Maintenance End Point) names

### Main Panels

#### Notification Monitoring
- **Notification Rate by Type** - Time series showing rate of different notification types (transaction, r2dtwo, triggered_source)
- **Notification Rate by Node** - Time series showing notification rates per node

#### MEP (Maintenance End Point) Metrics
- **MEP Octets Passed (Data Traffic)** - Data traffic volume through MEPs
- **MEP Octets Passed (OAM Traffic)** - OAM (Operations, Administration, and Maintenance) traffic volume
- **MEP Packets Passed (Data/OAM Traffic)** - Packet counts for both traffic types
- **MEP Status Table** - Current status and signal states of all MEPs

#### PRF (Packet Replication Function) Metrics
- **PRF Octets/Packets Passed** - Traffic volume and packet counts through PRF
- **PRF Pipeline Action Count** - Number of actions performed by each pipeline
- **PRF Pipeline Status** - Current mask states of PRF pipelines

#### Advanced Metrics
- **Triggered Source Metrics** - Shows level, session, and sequence information for triggered source events
- **Transaction Operations** - Breakdown of transaction operations by type and node

## Key Metrics Explained

### Notification Types

1. **Transaction Notifications**
   - `add_ifaces` - Number of interfaces added
   - `add_objects` - Number of objects added  
   - `add_streams` - Number of streams added

2. **R2DTWO Status Notifications**
   - Node operational status updates
   - Common statuses: "startup completed", "operational", "maintenance mode"

3. **Triggered Source Notifications**
   - Contains detailed MEP and PRF performance data
   - Includes traffic statistics and component states

### Traffic Types

- **Data Traffic** - User data packets and octets
- **OAM Traffic** - Operations, Administration, and Maintenance packets

### Component States

- **Mask Signal State** - "masked" or "unmasked" status for MEPs
- **Mask State** - Pipeline mask states for PRF components

## Usage Tips

1. **Time Range Selection** - Use the time picker to focus on specific time periods
2. **Node Filtering** - Use the hostname dropdown to focus on specific nodes
3. **MEP Filtering** - Filter by specific MEP names to analyze particular components
4. **Zoom and Pan** - Click and drag on time series charts to zoom into specific time ranges
5. **Legend Interaction** - Click legend items to show/hide specific series

## Alerting Recommendations

Consider setting up alerts for:

- **Low notification rates** - May indicate node connectivity issues
- **High error rates** - Transaction failures or status problems
- **Traffic anomalies** - Unusual MEP or PRF traffic patterns
- **Component state changes** - MEP or PRF mask state transitions

## Troubleshooting

### No Data Visible

1. Check if the collector service is running: `docker logs nd-collector`
2. Verify UDP notifications are being sent to port 6000
3. Confirm Prometheus is scraping the collector: http://localhost:9090/targets
4. Check the time range - data may be outside the selected time window

### Missing Metrics

1. Verify notification format matches expected JSON structure
2. Check collector logs for parsing errors
3. Ensure all required fields are present in notifications

### Performance Issues

1. Consider adjusting refresh rate if dashboard is slow
2. Use template variables to filter data and reduce query load
3. Adjust time ranges to focus on relevant periods

## Dashboard Customization

The dashboard can be customized by:

1. **Adding new panels** - Create additional visualizations for specific metrics
2. **Modifying queries** - Adjust PromQL queries to show different aggregations
3. **Creating alerts** - Set up Grafana alerts based on metric thresholds
4. **Exporting/Importing** - Save dashboard configuration for backup or sharing

## Related Components

- **Collector Service** - UDP receiver and metrics processor (`services/collector/`)
- **Prometheus** - Metrics storage and querying (http://localhost:9090)
- **Test Clients** - Data generators (`test_udp_client.py`, `continuous_test_client.py`)

## Access Information

- **Dashboard URL**: http://localhost:3000/d/network-nodes
- **Dashboard UID**: `network-nodes`
- **Prometheus Data Source**: `Prometheus` (http://prometheus:9090)
- **Refresh Rate**: 10 seconds (configurable)