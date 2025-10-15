from prometheus_client import Counter, Gauge, Enum, start_http_server
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
        start_http_server(port, addr=host)
        
        # Define metrics

        
        labels = ['hostname','component_name','component_type','stream_name','object_name', 'object_type']
        
        # SequenceRecovery
        self.seqrec_discarded_pacekets_counter = Counter('seqrec_discarded_packets', 'Number of discarded packets', labels)
        
        
        #MIP - sequenceRecovery
        # Gauges
        self.mip_seqrec_level_gauge = Gauge('mip_seqrec_level', 'Current SequenceRecovery level', labels)
        self.mip_seqrec_history_length_gauge = Gauge('mip_seqrec_history_length', 'History length', labels)
        self.mip_seqrec_latent_error_paths_gauge = Gauge('mip_seqrec_latent_error_paths', 'Number of latent error paths', labels)
        self.mip_seqrec_recovery_seq_num_gauge = Gauge('mip_seqrec_recovery_seq_num', 'Recovery sequence number', labels)
        self.mip_seqrec_reset_msec_gauge = Gauge('mip_seqrec_reset_msec', 'Reset time in milliseconds', labels)

        # Counters
        self.mip_seqrec_discarded_packets_counter = Counter('mip_seqrec_discarded_packets', 'Number of discarded packets', labels)
        self.mip_seqrec_latent_error_resets_counter = Counter('mip_seqrec_latent_error_resets', 'Number of latent error resets', labels)
        self.mip_seqrec_latent_errors_counter = Counter('mip_seqrec_latent_errors', 'Number of latent errors', labels)
        self.mip_seqrec_passed_packets_counter = Counter('mip_seqrec_passed_packets', 'Number of passed packets', labels)
        self.mip_seqrec_seq_recovery_resets_counter = Counter('mip_seqrec_seq_recovery_resets', 'Number of sequence recovery resets', labels)
        self.mip_seqrec_recv_counter = Counter('mip_seqrec_recv', 'Number of received items', labels)
        self.mip_seqrec_send_counter = Counter('mip_seqrec_send', 'Number of sent items', labels)

        # Enums
        self.mip_seqrec_recovery_algorithm = Enum('mip_seqrec_recovery_algorithm', 'Algorithm used for sequence recovery',
                                                  states=['vector'], labels=labels)
        self.mip_seqrec_use_init_flag = Enum('mip_seqrec_use_init_flag', "Use init flag",
                                             states=["true", "false"], labels=labels)
        self.mip_seqrec_use_reset_flag = Enum('mip_seqrec_use_reset_flag', "Use reset flag",
                                             states=["true", "false"], labels=labels)

        #MIP - Replicate
        replicate_labels = labels + ['pipeline_name']
        
        # Gauges for Replicate
        self.mip_replicate_level_gauge = Gauge('mip_replicate_level', 'Current Replicate level', labels)
        self.mip_replicate_pipeline_action_count_gauge = Gauge('mip_replicate_pipeline_action_count', 'Action count per pipeline', replicate_labels)

        # Counters for Replicate
        self.mip_replicate_octets_passed_counter = Counter('mip_replicate_octets_passed', 'Number of octets passed', labels)
        self.mip_replicate_packets_passed_counter = Counter('mip_replicate_packets_passed', 'Number of packets passed', labels)
        self.mip_replicate_recv_counter = Counter('mip_replicate_recv', 'Number of received items for replicate', labels)
        self.mip_replicate_send_counter = Counter('mip_replicate_send', 'Number of sent items for replicate', labels)


        # SequenceGenerator
        seqgen_labels = ['hostname','component_name','component_type']

        self.seqgen_use_init_flag = Enum('seqgen_use_init_flag', "Use init flag",
                                        states=["true", "false"], labels=seqgen_labels)
        self.seqgen_use_reset_flag = Enum('seqgen_use_reset_flag', "Use reset flag",
                                        states=["true", "false"], labels=seqgen_labels)
        
        #Interfaces
        interface_labels = ['hostname','interface_name']

        self.interface_recv_octets_gauge = Gauge('interface_recv_octets', 'Number of octets received on interface', interface_labels)
        self.interface_recv_packets_gauge = Gauge('interface_recv_packets', 'Number of packets received on interface', interface_labels)
        self.interface_send_octets_gauge = Gauge('interface_send_octets', 'Number of octets sent on interface', interface_labels)
        self.interface_send_packets_gauge = Gauge('interface_send_packets', 'Number of packets sent on interface', interface_labels)