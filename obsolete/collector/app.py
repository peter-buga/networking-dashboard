import json
import socket
import threading
import time
import logging
import os
from typing import Dict, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler

from prometheus_client import (
    Counter, Gauge, Histogram, Info, 
    CollectorRegistry, generate_latest, 
    CONTENT_TYPE_LATEST
)

# Configuration
UDP_PORT = int(os.getenv("UDP_PORT", "6000"))
UDP_HOST = os.getenv("UDP_HOST", "0.0.0.0")
METRICS_PORT = int(os.getenv("METRICS_PORT", "8002"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus registry
registry = CollectorRegistry()

# Network node metrics
node_notifications_total = Counter(
    'network_node_notifications_total',
    'Total notifications received from network nodes',
    ['hostname', 'notification_type'],
    registry=registry
)

node_transaction_operations = Counter(
    'network_node_transaction_operations_total',
    'Transaction operations performed by nodes',
    ['hostname', 'operation_type'],
    registry=registry
)

node_status_info = Info(
    'network_node_status',
    'Current status of network nodes',
    ['hostname'],
    registry=registry
)

# MEP (Maintenance End Point) metrics
mep_octets_passed = Gauge(
    'network_mep_octets_passed',
    'Octets passed through MEP',
    ['hostname', 'mep_name', 'traffic_type'],
    registry=registry
)

mep_packets_passed = Gauge(
    'network_mep_packets_passed',
    'Packets passed through MEP',
    ['hostname', 'mep_name', 'traffic_type'],
    registry=registry
)

mep_mask_signal_state = Info(
    'network_mep_mask_signal_state',
    'MEP mask signal state',
    ['hostname', 'mep_name'],
    registry=registry
)

# Replicate/PRF metrics
prf_octets_passed = Gauge(
    'network_prf_octets_passed',
    'Octets passed through PRF (Packet Replication Function)',
    ['hostname', 'prf_name'],
    registry=registry
)

prf_packets_passed = Gauge(
    'network_prf_packets_passed',
    'Packets passed through PRF (Packet Replication Function)',
    ['hostname', 'prf_name'],
    registry=registry
)

prf_pipeline_action_count = Gauge(
    'network_prf_pipeline_action_count',
    'Action count for PRF pipelines',
    ['hostname', 'prf_name', 'pipeline_name'],
    registry=registry
)

prf_pipeline_mask_state = Info(
    'network_prf_pipeline_mask_state',
    'Mask state for PRF pipelines',
    ['hostname', 'prf_name', 'pipeline_name'],
    registry=registry
)

# Triggered source metrics
triggered_source_level = Gauge(
    'network_triggered_source_level',
    'Level of triggered source events',
    ['hostname', 'source', 'target', 'stream'],
    registry=registry
)

triggered_source_session = Gauge(
    'network_triggered_source_session',
    'Session ID of triggered source events',
    ['hostname', 'source', 'target', 'stream'],
    registry=registry
)

triggered_source_seq = Gauge(
    'network_triggered_source_seq',
    'Sequence number of triggered source events',
    ['hostname', 'source', 'target', 'stream'],
    registry=registry
)


class NotificationProcessor:
    """Process incoming network node notifications and update Prometheus metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process_notification(self, notification: Dict[str, Any]) -> None:
        """Process a single notification and update relevant metrics."""
        try:
            hostname = notification.get('notif_hostname', 'unknown')
            notif_msg = notification.get('notif_msg', {})
            notif_seq = notification.get('notif_seq', 0)
            notif_tstamp = notification.get('notif_tstamp', 0)
            
            self.logger.debug(f"Processing notification from {hostname}, seq={notif_seq}")
            
            # Determine notification type and process accordingly
            if 'transaction' in notif_msg:
                self._process_transaction(hostname, notif_msg['transaction'])
                node_notifications_total.labels(hostname=hostname, notification_type='transaction').inc()
                
            elif 'r2dtwo' in notif_msg:
                self._process_r2dtwo_status(hostname, notif_msg['r2dtwo'])
                node_notifications_total.labels(hostname=hostname, notification_type='r2dtwo_status').inc()
                
            elif 'triggered_source' in notif_msg:
                self._process_triggered_source(hostname, notif_msg['triggered_source'])
                node_notifications_total.labels(hostname=hostname, notification_type='triggered_source').inc()
                
            else:
                self.logger.warning(f"Unknown notification type from {hostname}: {list(notif_msg.keys())}")
                node_notifications_total.labels(hostname=hostname, notification_type='unknown').inc()
                
        except Exception as e:
            self.logger.error(f"Error processing notification: {e}")
    
    def _process_transaction(self, hostname: str, transaction: Dict[str, Any]) -> None:
        """Process transaction notification."""
        for operation, count in transaction.items():
            if operation != 'committed' and isinstance(count, int):
                node_transaction_operations.labels(
                    hostname=hostname, 
                    operation_type=operation
                ).inc(count)
    
    def _process_r2dtwo_status(self, hostname: str, r2dtwo: Dict[str, Any]) -> None:
        """Process r2dtwo status notification."""
        status = r2dtwo.get('status', 'unknown')
        node_status_info.labels(hostname=hostname).info({'status': status})
    
    def _process_triggered_source(self, hostname: str, triggered_source: Dict[str, Any]) -> None:
        """Process triggered source notification with MEP data."""
        level = triggered_source.get('level', 0)
        source = triggered_source.get('source', 'unknown')
        target = triggered_source.get('target', 'unknown')
        stream = triggered_source.get('stream', 'unknown')
        session = triggered_source.get('session', 0)
        seq = triggered_source.get('seq', 0)
        
        # Update triggered source metrics
        triggered_source_level.labels(
            hostname=hostname, source=source, target=target, stream=stream
        ).set(level)
        
        triggered_source_session.labels(
            hostname=hostname, source=source, target=target, stream=stream
        ).set(session)
        
        triggered_source_seq.labels(
            hostname=hostname, source=source, target=target, stream=stream
        ).set(seq)
        
        # Process MEP data
        mep_list = triggered_source.get('mep', [])
        for mep in mep_list:
            self._process_mep_data(hostname, mep)
    
    def _process_mep_data(self, hostname: str, mep: Dict[str, Any]) -> None:
        """Process MEP (Maintenance End Point) data."""
        mep_name = mep.get('name', 'unknown')
        mep_type = mep.get('type', 'unknown')
        
        if mep_type == 'mep_state':
            # Process MEP state data
            mask_signal_state = mep.get('mask_signal_state', 'unknown')
            oam_octets = mep.get('oam_octets_passed', 0)
            oam_packets = mep.get('oam_packets_passed', 0)
            octets = mep.get('octets_passed', 0)
            packets = mep.get('packets_passed', 0)
            
            mep_mask_signal_state.labels(hostname=hostname, mep_name=mep_name).info({
                'mask_signal_state': mask_signal_state
            })
            
            mep_octets_passed.labels(hostname=hostname, mep_name=mep_name, traffic_type='oam').set(oam_octets)
            mep_packets_passed.labels(hostname=hostname, mep_name=mep_name, traffic_type='oam').set(oam_packets)
            mep_octets_passed.labels(hostname=hostname, mep_name=mep_name, traffic_type='data').set(octets)
            mep_packets_passed.labels(hostname=hostname, mep_name=mep_name, traffic_type='data').set(packets)
            
        elif mep_type == 'replicate':
            # Process PRF (Packet Replication Function) data
            octets = mep.get('octets_passed', 0)
            packets = mep.get('packets_passed', 0)
            pipelines = mep.get('pipelines', [])
            
            prf_octets_passed.labels(hostname=hostname, prf_name=mep_name).set(octets)
            prf_packets_passed.labels(hostname=hostname, prf_name=mep_name).set(packets)
            
            for pipeline in pipelines:
                pipeline_name = pipeline.get('name', 'unknown')
                action_count = pipeline.get('action_count', 0)
                mask_state = pipeline.get('mask_state', 'unknown')
                
                prf_pipeline_action_count.labels(
                    hostname=hostname, prf_name=mep_name, pipeline_name=pipeline_name
                ).set(action_count)
                
                prf_pipeline_mask_state.labels(
                    hostname=hostname, prf_name=mep_name, pipeline_name=pipeline_name
                ).info({'mask_state': mask_state})


class UDPReceiver:
    """UDP server to receive JSON notifications from network nodes."""
    
    def __init__(self, host: str, port: int, processor: NotificationProcessor):
        self.host = host
        self.port = port
        self.processor = processor
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sock = None
        self.running = False
    
    def start(self):
        """Start the UDP receiver."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.host, self.port))
            self.running = True
            
            self.logger.info(f"UDP receiver started on {self.host}:{self.port}")
            
            while self.running:
                try:
                    data, addr = self.sock.recvfrom(65536)  # Max UDP packet size
                    self.logger.debug(f"Received {len(data)} bytes from {addr}")
                    
                    # Parse JSON
                    try:
                        notification = json.loads(data.decode('utf-8'))
                        self.processor.process_notification(notification)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Invalid JSON from {addr}: {e}")
                    except UnicodeDecodeError as e:
                        self.logger.error(f"Unicode decode error from {addr}: {e}")
                        
                except socket.error as e:
                    if self.running:  # Only log if we're still supposed to be running
                        self.logger.error(f"Socket error: {e}")
                        
        except Exception as e:
            self.logger.error(f"Failed to start UDP receiver: {e}")
            raise
    
    def stop(self):
        """Stop the UDP receiver."""
        self.running = False
        if self.sock:
            self.sock.close()
        self.logger.info("UDP receiver stopped")


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""
    
    def do_GET(self):
        if self.path == "/metrics":
            data = generate_latest(registry)
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default HTTP logging
        pass


def start_metrics_server():
    """Start the Prometheus metrics HTTP server."""
    server = HTTPServer(("0.0.0.0", METRICS_PORT), MetricsHandler)
    logger.info(f"Metrics server started on port {METRICS_PORT}")
    server.serve_forever()


def main():
    """Main application entry point."""
    logger.info("Starting network node collector service")
    logger.info(f"Configuration: UDP={UDP_HOST}:{UDP_PORT}, Metrics={METRICS_PORT}")
    
    # Create notification processor
    processor = NotificationProcessor()
    
    # Start metrics server in background thread
    metrics_thread = threading.Thread(target=start_metrics_server, daemon=True)
    metrics_thread.start()
    
    # Start UDP receiver (this will block)
    udp_receiver = UDPReceiver(UDP_HOST, UDP_PORT, processor)
    
    try:
        udp_receiver.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        udp_receiver.stop()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()