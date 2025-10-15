from prometheus_client import Counter, Gauge, start_http_server
from typing import Optional, Dict, Any

class MetricsExporter:
    """Encapsulate Prometheus metrics registration and updates."""
    def __init__(self, host: str, port: int, disabled: bool):
        self.enabled = False
        self._last_timestamp_gauge: Optional[Gauge] = None

        if disabled:
            print("Prometheus metrics exporter disabled by configuration.")
            return

        self.enabled = True
        # Basic notification metrics
        self.received = Counter(
            "json_notifications_received_total",
            "Total JSON notification datagrams received (including fragments)",
            ["hostname"]
        )
        self.processed = Counter(
            "json_notifications_processed_total",
            "Total JSON notifications successfully reassembled",
            ["hostname"]
        )
        self.bytes = Counter(
            "json_notification_bytes_total",
            "Total bytes of JSON notification payloads",
            ["hostname"]
        )
        self.failures = Counter(
            "json_notification_failures_total",
            "Total JSON notifications that failed to parse or validate",
            ["hostname"]
        )

        # Network interface metrics
        self.interface_packets_total = Gauge(
            "network_interface_packets_total",
            "Total packets on network interfaces",
            ["hostname", "interface", "direction"]
        )
        self.interface_octets_total = Gauge(
            "network_interface_octets_total", 
            "Total octets (bytes) on network interfaces",
            ["hostname", "interface", "direction"]
        )

        # MIP (Message Interface Point) metrics
        self.mip_packets_total = Counter(
            "mip_packets_total",
            "Total packets processed by MIP components",
            ["hostname", "component", "direction", "stream"]
        )
        self.mip_component_level = Gauge(
            "mip_component_level",
            "Current level setting of MIP components",
            ["hostname", "component"]
        )

        # Sequence recovery metrics
        self.seqrec_discarded_packets = Gauge(
            "seqrec_discarded_packets",
            "Number of discarded packets in sequence recovery",
            ["hostname", "component"]
        )
        self.seqrec_passed_packets = Gauge(
            "seqrec_passed_packets", 
            "Number of passed packets in sequence recovery",
            ["hostname", "component"]
        )
        self.seqrec_latent_errors = Gauge(
            "seqrec_latent_errors",
            "Number of latent errors in sequence recovery",
            ["hostname", "component"]
        )
        self.seqrec_recovery_resets = Gauge(
            "seqrec_recovery_resets",
            "Number of sequence recovery resets",
            ["hostname", "component"]
        )
        self.seqrec_history_length = Gauge(
            "seqrec_history_length",
            "History length setting for sequence recovery",
            ["hostname", "component"]
        )
        self.seqrec_latent_error_paths = Gauge(
            "seqrec_latent_error_paths",
            "Number of latent error paths in sequence recovery",
            ["hostname", "component"]
        )
        self.seqrec_latent_error_resets = Gauge(
            "seqrec_latent_error_resets",
            "Number of latent error resets in sequence recovery",
            ["hostname", "component"]
        )
        self.seqrec_reset_msec = Gauge(
            "seqrec_reset_msec",
            "Reset timeout in milliseconds for sequence recovery",
            ["hostname", "component"]
        )

        # Packet replication metrics
        self.replicate_octets_passed = Gauge(
            "replicate_octets_passed",
            "Octets passed through packet replication",
            ["hostname", "component"]
        )
        self.replicate_packets_passed = Gauge(
            "replicate_packets_passed",
            "Packets passed through packet replication", 
            ["hostname", "component"]
        )
        self.replicate_pipeline_actions = Gauge(
            "replicate_pipeline_actions",
            "Action count for replication pipelines",
            ["hostname", "component", "pipeline"]
        )

        # Parser metrics
        self.parser_no_match_packets = Gauge(
            "parser_no_match_packets",
            "Packets that didn't match any parser rules",
            ["hostname", "parser"]
        )
        self.parser_no_match_octets = Gauge(
            "parser_no_match_octets",
            "Octets that didn't match any parser rules", 
            ["hostname", "parser"]
        )
        self.parser_stream_packets = Gauge(
            "parser_stream_packets",
            "Packets matched to specific streams",
            ["hostname", "parser", "stream"]
        )
        self.parser_stream_octets = Gauge(
            "parser_stream_octets",
            "Octets matched to specific streams",
            ["hostname", "parser", "stream"]
        )

        if Gauge is not None:
            self._last_timestamp_gauge = Gauge(
                "json_notification_last_timestamp_seconds",
                "Timestamp of the last successful JSON notification (notif_tstamp field)",
            )

        start_http_server(port, addr=host)
        print(f"Prometheus metrics exporter listening on http://{host}:{port}/metrics")

    def inc_received(self, hostname: str = "unknown"):
        if self.enabled:
            self.received.labels(hostname=hostname).inc()

    def inc_processed(self, hostname: str = "unknown"):
        if self.enabled:
            self.processed.labels(hostname=hostname).inc()

    def inc_failures(self, hostname: str = "unknown"):
        if self.enabled:
            self.failures.labels(hostname=hostname).inc()

    def add_bytes(self, byte_count: int, hostname: str = "unknown"):
        if self.enabled:
            self.bytes.labels(hostname=hostname).inc(byte_count)

    def set_last_timestamp(self, timestamp: Optional[float]):
        if self.enabled and self._last_timestamp_gauge is not None and timestamp is not None:
            try:
                self._last_timestamp_gauge.set(float(timestamp))
            except (TypeError, ValueError):
                # Ignore malformed timestamps
                pass

    def update_notification_metrics(self, notification: Dict[str, Any]):
        """Update metrics based on parsed notification message."""
        if not self.enabled:
            return

        hostname = notification.get("notif_hostname", "unknown")
        notif_msg = notification.get("notif_msg", {})

        for component_name, component_data in notif_msg.items():
            self._update_component_metrics(hostname, component_name, component_data)

    def _update_component_metrics(self, hostname: str, component_name: str, component_data: Dict[str, Any]):
        """Update metrics for a specific component."""
        # Handle interface notifications (ifNotify_*)
        if component_name.startswith("ifNotify_"):
            interface_name = component_name.replace("ifNotify_", "")
            self._update_interface_metrics(hostname, interface_name, component_data)
        
        # Handle NNI interfaces
        elif "nni" in component_name and "parser" not in component_name:
            self._update_interface_metrics(hostname, component_name, component_data)
        
        # Handle UNI interfaces
        elif component_name.startswith("uni_"):
            self._update_interface_metrics(hostname, component_name, component_data)
        
        # Handle parser components
        elif "parser" in component_name:
            self._update_parser_metrics(hostname, component_name, component_data)
        
        # Handle MIP components
        elif isinstance(component_data, dict) and component_data.get("type") == "MIP":
            self._update_mip_metrics(hostname, component_name, component_data)
        
        # Handle standalone components (gen, pef, prf)
        elif component_name in ["gen", "pef", "prf"]:
            self._update_standalone_component_metrics(hostname, component_name, component_data)

    def _update_interface_metrics(self, hostname: str, interface: str, data: Dict[str, Any]):
        """Update interface-related metrics."""
        recv_packets = data.get("recv_packets", 0)
        send_packets = data.get("send_packets", 0)
        recv_octets = data.get("recv_octets", 0) 
        send_octets = data.get("send_octets", 0)

        # Set gauge values directly (these are cumulative counters from equipment)
        self.interface_packets_total.labels(hostname=hostname, interface=interface, direction="recv").set(recv_packets)
        self.interface_packets_total.labels(hostname=hostname, interface=interface, direction="send").set(send_packets)
        self.interface_octets_total.labels(hostname=hostname, interface=interface, direction="recv").set(recv_octets)
        self.interface_octets_total.labels(hostname=hostname, interface=interface, direction="send").set(send_octets)

    def _update_mip_metrics(self, hostname: str, component: str, data: Dict[str, Any]):
        """Update MIP component metrics."""
        level = data.get("level", 0)
        recv_count = data.get("recv", 0)
        send_count = data.get("send", 0)
        stream_name = data.get("stream_name", "unknown")

        self.mip_component_level.labels(hostname=hostname, component=component).set(level)
        
        if recv_count > 0:
            self.mip_packets_total.labels(hostname=hostname, component=component, direction="recv", stream=stream_name).inc(recv_count)
        if send_count > 0:
            self.mip_packets_total.labels(hostname=hostname, component=component, direction="send", stream=stream_name).inc(send_count)

        # Handle object-specific metrics
        obj_data = data.get("object", {})
        obj_type = obj_data.get("type")

        if obj_type == "seqrec":
            self._update_seqrec_metrics(hostname, component, obj_data)
        elif obj_type == "replicate":
            self._update_replicate_metrics(hostname, component, obj_data)

    def _update_seqrec_metrics(self, hostname: str, component: str, data: Dict[str, Any]):
        """Update sequence recovery metrics."""
        self.seqrec_discarded_packets.labels(hostname=hostname, component=component).set(data.get("discarded_packets", 0))
        self.seqrec_passed_packets.labels(hostname=hostname, component=component).set(data.get("passed_packets", 0))
        self.seqrec_latent_errors.labels(hostname=hostname, component=component).set(data.get("latent_errors", 0))
        self.seqrec_recovery_resets.labels(hostname=hostname, component=component).set(data.get("seq_recovery_resets", 0))
        self.seqrec_history_length.labels(hostname=hostname, component=component).set(data.get("history_length", 0))
        self.seqrec_latent_error_paths.labels(hostname=hostname, component=component).set(data.get("latent_error_paths", 0))
        self.seqrec_latent_error_resets.labels(hostname=hostname, component=component).set(data.get("latent_error_resets", 0))
        self.seqrec_reset_msec.labels(hostname=hostname, component=component).set(data.get("reset_msec", 0))

    def _update_replicate_metrics(self, hostname: str, component: str, data: Dict[str, Any]):
        """Update packet replication metrics."""
        self.replicate_octets_passed.labels(hostname=hostname, component=component).set(data.get("octets_passed", 0))
        self.replicate_packets_passed.labels(hostname=hostname, component=component).set(data.get("packets_passed", 0))

        # Handle pipeline metrics
        pipelines = data.get("pipelines", [])
        for pipeline in pipelines:
            pipeline_name = pipeline.get("name", "unknown")
            action_count = pipeline.get("action_count", 0)
            self.replicate_pipeline_actions.labels(hostname=hostname, component=component, pipeline=pipeline_name).set(action_count)

    def _update_parser_metrics(self, hostname: str, parser: str, data: Dict[str, Any]):
        """Update parser component metrics."""
        no_match_packets = data.get("no match packets", 0)
        no_match_octets = data.get("no match octets", 0)

        self.parser_no_match_packets.labels(hostname=hostname, parser=parser).set(no_match_packets)
        self.parser_no_match_octets.labels(hostname=hostname, parser=parser).set(no_match_octets)

        # Handle stream-specific metrics
        for key, value in data.items():
            if key.endswith(" packets") and not key.startswith("no match"):
                stream_name = key.replace(" packets", "")
                self.parser_stream_packets.labels(hostname=hostname, parser=parser, stream=stream_name).set(value)
            elif key.endswith(" octets") and not key.startswith("no match"):
                stream_name = key.replace(" octets", "")
                self.parser_stream_octets.labels(hostname=hostname, parser=parser, stream=stream_name).set(value)

    def _update_standalone_component_metrics(self, hostname: str, component: str, data: Dict[str, Any]):
        """Update metrics for standalone components like gen, pef, prf."""
        component_type = data.get("type")
        
        if component_type == "seqrec":
            self._update_seqrec_metrics(hostname, component, data)
        elif component_type == "replicate":
            self._update_replicate_metrics(hostname, component, data)