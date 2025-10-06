import json
from collections import deque
from typing import Optional, Dict, Any


class NotificationReceiver:
    def __init__(self, seq_history_size=20):
        """
        Initialize NotificationReceiver. Optional parameter is
        @seq_history_size, the size of the history window, default value is 20.
        """
        self.SEQ_HISTORY_SIZE = seq_history_size
        self.last_seqnums = deque(maxlen=self.SEQ_HISTORY_SIZE)
        self.mpart_collector = {}

    def process_notification(self, host, port, data):
        """
        Performs the reassembly of message fragments and also elimination
        of duplicate messages. Returns only full json messages.
        @host, @port - sender host and port, needed for elimination
        @data - the received json fragment
        @return None if message is partial, or the full json message.
        """
        try:
            jsonReceived = self._parse_json(data)
            self._validate_input(jsonReceived)

            seq_num = jsonReceived.get("notif_seq")
            hostname = jsonReceived.get("notif_hostname")
            frag_id = jsonReceived.get("notif_fragment")
            timestamp = jsonReceived.get("notif_tstamp")

            if self._is_duplicate(hostname, seq_num, str(frag_id)):
                return None

            self.last_seqnums.append((hostname, seq_num, str(frag_id)))

            if frag_id is None or frag_id == "1/1":
                return self._handle_single_part(jsonReceived)

            return self._handle_multipart(jsonReceived, hostname, seq_num, frag_id, timestamp, port)

        except json.decoder.JSONDecodeError:
            print("JSON decoder error: received message is not JSON or buffer is too small")
        except ValueError as ve:
            print(f"Invalid input: {ve}")

        return None

    def _parse_json(self, data):
        return json.loads(data)

    def _validate_input(self, message):
        required_fields = ["notif_seq", "notif_hostname", "notif_msg", "notif_tstamp"]
        for field in required_fields:
            if field not in message:
                raise ValueError(f"Missing required field: {field}")

    def _is_duplicate(self, hostname, seq_num, frag_id):
        return (hostname, seq_num, frag_id) in self.last_seqnums

    def _handle_single_part(self, message):
        message["notif_msg"] = json.loads(message["notif_msg"])
        return message

    def _handle_multipart(self, jsonReceived, hostname, seq_num, frag_id, timestamp, port):
        if hostname not in self.mpart_collector:
            self.mpart_collector[hostname] = {
                "message": {},
                "last_seqnum": -1,
                "frag_count": 0,
            }

        idx = int(frag_id.split('/')[0])
        items = int(frag_id.split('/')[1])
        collector = self.mpart_collector[hostname]

        if collector["last_seqnum"] != seq_num:
            if collector["frag_count"] > 0:
                print(f"\nWARNING: Missing part(s) of multipart message from {hostname} : {port} with sequence number {collector['last_seqnum']}")
            collector["last_seqnum"] = seq_num
            collector["message"].clear()
            collector["frag_count"] = 0

        collector["message"][idx] = jsonReceived.get("notif_msg")
        collector["frag_count"] += 1

        if collector["frag_count"] == items:
            concat_string = ""
            for i in range(1, items + 1):
                part = collector["message"].get(i)
                if part is None:
                    print(f"\nWARNING: Missing part(s) of multipart message reassembly from {hostname} : {port} with sequence number {seq_num}")
                    return None
                concat_string += part

            fullmsg_with_header = {
                "notif_seq": seq_num,
                "notif_hostname": hostname,
                "notif_tstamp": timestamp,
                "notif_msg": json.loads(concat_string)
            }

            collector["last_seqnum"] = -1
            collector["message"].clear()
            collector["frag_count"] = 0

            return fullmsg_with_header

        return None

    def extract_notification_summary(self, notification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract useful summary information from a notification for logging/monitoring.
        Returns a dictionary with key statistics and health indicators.
        """
        if not notification or "notif_msg" not in notification:
            return {}

        hostname = notification.get("notif_hostname", "unknown")
        seq_num = notification.get("notif_seq", 0)
        timestamp = notification.get("notif_tstamp", 0)
        notif_msg = notification.get("notif_msg", {})

        summary = {
            "hostname": hostname,
            "sequence": seq_num,
            "timestamp": timestamp,
            "interfaces": {},
            "mip_components": {},
            "error_indicators": {},
            "traffic_summary": {
                "total_recv_packets": 0,
                "total_send_packets": 0,
                "total_recv_octets": 0,
                "total_send_octets": 0
            }
        }

        # Process each component in the notification
        for component_name, component_data in notif_msg.items():
            if not isinstance(component_data, dict):
                continue

            # Extract interface statistics
            if (component_name.startswith("ifNotify_") or 
                component_name.startswith("nni") or 
                component_name.startswith("uni_")):
                self._extract_interface_stats(summary, component_name, component_data)

            # Extract MIP component information
            elif component_data.get("type") == "MIP":
                self._extract_mip_stats(summary, component_name, component_data)

            # Extract error indicators from recovery components
            elif component_data.get("type") in ["seqrec", "replicate"] or component_name in ["pef", "prf"]:
                self._extract_error_indicators(summary, component_name, component_data)

        return summary

    def _extract_interface_stats(self, summary: Dict[str, Any], interface_name: str, data: Dict[str, Any]):
        """Extract interface statistics for summary."""
        recv_packets = data.get("recv_packets", 0)
        send_packets = data.get("send_packets", 0) 
        recv_octets = data.get("recv_octets", 0)
        send_octets = data.get("send_octets", 0)

        summary["interfaces"][interface_name] = {
            "recv_packets": recv_packets,
            "send_packets": send_packets,
            "recv_octets": recv_octets,
            "send_octets": send_octets,
            "active": recv_packets > 0 or send_packets > 0
        }

        # Add to traffic summary
        summary["traffic_summary"]["total_recv_packets"] += recv_packets
        summary["traffic_summary"]["total_send_packets"] += send_packets
        summary["traffic_summary"]["total_recv_octets"] += recv_octets
        summary["traffic_summary"]["total_send_octets"] += send_octets

    def _extract_mip_stats(self, summary: Dict[str, Any], component_name: str, data: Dict[str, Any]):
        """Extract MIP component statistics for summary."""
        level = data.get("level", 0)
        recv_count = data.get("recv", 0)
        send_count = data.get("send", 0)
        stream_name = data.get("stream_name", "unknown")

        summary["mip_components"][component_name] = {
            "level": level,
            "recv": recv_count,
            "send": send_count,
            "stream": stream_name,
            "active": recv_count > 0 or send_count > 0
        }

        # Check for object-specific information
        obj_data = data.get("object", {})
        if obj_data:
            obj_type = obj_data.get("type")
            summary["mip_components"][component_name]["object_type"] = obj_type
            
            if obj_type == "seqrec":
                summary["mip_components"][component_name]["errors"] = {
                    "discarded_packets": obj_data.get("discarded_packets", 0),
                    "latent_errors": obj_data.get("latent_errors", 0)
                }
            elif obj_type == "replicate":
                summary["mip_components"][component_name]["replication"] = {
                    "packets_passed": obj_data.get("packets_passed", 0),
                    "octets_passed": obj_data.get("octets_passed", 0)
                }

    def _extract_error_indicators(self, summary: Dict[str, Any], component_name: str, data: Dict[str, Any]):
        """Extract error indicators from recovery components."""
        component_type = data.get("type")
        
        if component_type == "seqrec":
            discarded = data.get("discarded_packets", 0)
            latent_errors = data.get("latent_errors", 0)
            resets = data.get("seq_recovery_resets", 0)
            
            if discarded > 0 or latent_errors > 0 or resets > 0:
                summary["error_indicators"][component_name] = {
                    "type": "sequence_recovery",
                    "discarded_packets": discarded,
                    "latent_errors": latent_errors,
                    "recovery_resets": resets
                }
        
        elif component_type == "replicate":
            # Look for pipeline issues
            pipelines = data.get("pipelines", [])
            for pipeline in pipelines:
                if pipeline.get("mask_state") == "masked":
                    if component_name not in summary["error_indicators"]:
                        summary["error_indicators"][component_name] = {
                            "type": "replication", 
                            "masked_pipelines": []
                        }
                    summary["error_indicators"][component_name]["masked_pipelines"].append(pipeline.get("name"))
