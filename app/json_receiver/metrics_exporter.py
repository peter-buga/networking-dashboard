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
        # SequenceRecovery
        labels = ['hostname','component_name','component_type','object_name', 'object_type']
        self.seqrec_discarded_packets_counter = Counter('seqrec_discarded_packets', 'Number of discarded packets', labels)
        self.seqrec_history_length_gauge =  Gauge('seqrec_history_length', 'History length', labels)
        self.seqrec_latent_error_paths_gauge = Gauge('seqrec_latent_error_paths', 'Number of latent error paths', labels)
        self.seqrec_latent_error_resets_counter = Counter('seqrec_latent_error_resets', 'Number of latent error resets', labels)
        self.seqrec_latent_errors_counter = Counter('seqrec_latent_errors', 'Number of latent errors', labels)
        self.seqrec_passed_packets_counter = Counter('seqrec_passed_packets', 'Number of passed packets', labels)
        self.seqrec_recovery_algorithm = Enum('seqrec_recovery_algorithm', 'Algorithm used for sequence recovery',
                                              states=['vector', 'match'], labelnames=labels)
        self.seqrec_use_init_flag = Enum('seqrec_use_init_flag', "Use init flag",
                                        states=["true", "false"], labelnames=labels)
        self.seqrec_use_reset_flag = Enum('seqrec_use_reset_flag', "Use reset flag",
                                        states=["true", "false"], labelnames=labels)
        self.seqrec_seq_recovery_resets_counter = Counter('seqrec_seq_recovery_resets', 'Number of sequence recovery resets', labels)
        self.seqrec_recovery_seq_num_gauge = Gauge('seqrec_recovery_seq_num', 'Recovery sequence number', labels)
        self.seqrec_reset_msec_gauge = Gauge('seqrec_reset_msec', 'Reset time in milliseconds', labels)
        
        #MIP - sequenceRecovery
        seqrec_mip_labels = ['hostname','component_name','component_type','stream_name','object_name', 'object_type']
        self.mip_seqrec_level_gauge = Gauge('mip_seqrec_level', 'Current SequenceRecovery level', seqrec_mip_labels)
        self.mip_seqrec_history_length_gauge = Gauge('mip_seqrec_history_length', 'History length', seqrec_mip_labels)
        self.mip_seqrec_latent_error_paths_gauge = Gauge('mip_seqrec_latent_error_paths', 'Number of latent error paths', seqrec_mip_labels)
        self.mip_seqrec_recovery_seq_num_gauge = Gauge('mip_seqrec_recovery_seq_num', 'Recovery sequence number', seqrec_mip_labels)
        self.mip_seqrec_reset_msec_gauge = Gauge('mip_seqrec_reset_msec', 'Reset time in milliseconds', seqrec_mip_labels)

        self.mip_seqrec_discarded_packets_counter = Counter('mip_seqrec_discarded_packets', 'Number of discarded packets', seqrec_mip_labels)
        self.mip_seqrec_latent_error_resets_counter = Counter('mip_seqrec_latent_error_resets', 'Number of latent error resets', seqrec_mip_labels)
        self.mip_seqrec_latent_errors_counter = Counter('mip_seqrec_latent_errors', 'Number of latent errors', seqrec_mip_labels)
        self.mip_seqrec_passed_packets_counter = Counter('mip_seqrec_passed_packets', 'Number of passed packets', seqrec_mip_labels)
        self.mip_seqrec_seq_recovery_resets_counter = Counter('mip_seqrec_seq_recovery_resets', 'Number of sequence recovery resets', seqrec_mip_labels)
        self.mip_seqrec_recv_counter = Counter('mip_seqrec_recv', 'Number of received items', seqrec_mip_labels)
        self.mip_seqrec_send_counter = Counter('mip_seqrec_send', 'Number of sent items', seqrec_mip_labels)

    
        self.mip_seqrec_recovery_algorithm = Enum('mip_seqrec_recovery_algorithm', 'Algorithm used for sequence recovery',
                                                  states=['vector', 'match'], labelnames=seqrec_mip_labels)
        self.mip_seqrec_use_init_flag = Enum('mip_seqrec_use_init_flag', "Use init flag",
                                             states=["true", "false"], labelnames=seqrec_mip_labels)
        self.mip_seqrec_use_reset_flag = Enum('mip_seqrec_use_reset_flag', "Use reset flag",
                                             states=["true", "false"], labelnames=seqrec_mip_labels)


        #Replicate
        replicate_labels = labels + ['pipeline_name']

        self.replicate_octets_passed_counter = Counter('replicate_octets_passed', 'Number of octets passed', labels)
        self.replicate_packets_passed_counter = Counter('replicate_packets_passed', 'Number of packets passed', labels)
        self.replicate_pipeline_action_count_gauge = Gauge('replicate_pipeline_action_count', 'Action count per pipeline', replicate_labels)
        self.replicate_pipeline_mask_state = Enum('replicate_pipeline_mask_state', 'Mask state of the pipeline',
                                                  states=['masked', 'unmasked'], labelnames=replicate_labels)

        #MIP - Replicate
        replicate_mip_labels = ['hostname','component_name','component_type','stream_name','object_name', 'object_type']
        replicate_mip_pipeline_labels = replicate_mip_labels + ['pipeline_name']
        
        self.mip_replicate_level_gauge = Gauge('mip_replicate_level', 'Current Replicate level', replicate_mip_labels)
        self.mip_replicate_octets_passed_counter = Counter('mip_replicate_octets_passed', 'Number of octets passed', replicate_mip_labels)
        self.mip_replicate_packets_passed_counter = Counter('mip_replicate_packets_passed', 'Number of packets passed', replicate_mip_labels)
        self.mip_replicate_recv_counter = Counter('mip_replicate_recv', 'Number of received items for replicate', replicate_mip_labels)
        self.mip_replicate_send_counter = Counter('mip_replicate_send', 'Number of sent items for replicate', replicate_mip_labels)
        self.mip_replicate_pipeline_action_count_gauge = Gauge('mip_replicate_pipeline_action_count', 'Action count per pipeline', replicate_mip_pipeline_labels)
        self.mip_replicate_pipeline_mask_state = Enum('mip_replicate_pipeline_mask_state', 'Mask state of the pipeline',
                                                     states=['masked', 'unmasked'], labelnames=replicate_mip_pipeline_labels)

        # SequenceGenerator
        seqgen_labels = ['hostname','component_name','component_type']

        self.seqgen_use_init_flag = Enum('seqgen_use_init_flag', "Use init flag",
                                        states=["true", "false"], labelnames=seqgen_labels)
        self.seqgen_use_reset_flag = Enum('seqgen_use_reset_flag', "Use reset flag",
                                        states=["true", "false"], labelnames=seqgen_labels)
        
        #Interfaces
        interface_labels = ['hostname','interface_name']

        self.interface_recv_octets_gauge = Gauge('interface_recv_octets', 'Number of octets received on interface', interface_labels)
        self.interface_recv_packets_gauge = Gauge('interface_recv_packets', 'Number of packets received on interface', interface_labels)
        self.interface_send_octets_gauge = Gauge('interface_send_octets', 'Number of octets sent on interface', interface_labels)
        self.interface_send_packets_gauge = Gauge('interface_send_packets', 'Number of packets sent on interface', interface_labels)

    def update_notification_metrics(self, json_data: Dict[str, Any]) -> None:
        """Parse notification JSON and update Prometheus metrics."""
        if not self.enabled or json_data is None:
            return
        
        hostname = json_data.get('notif_hostname', 'unknown')
        notif_msg = json_data.get('notif_msg', {})
        
        for component_name, component_data in notif_msg.items():
            if not isinstance(component_data, dict):
                continue
            
            component_type = component_data.get('type', '')
            
            # Handle MIP components (contains nested object)
            if component_type == 'MIP':
                level = component_data.get('level', -1)
                stream_name = component_data.get('stream_name', '')
                recv = component_data.get('recv', 0)
                send = component_data.get('send', 0)
                obj = component_data.get('object', {})
                
                if not isinstance(obj, dict):
                    continue
                
                object_name = obj.get('name', '')
                object_type = obj.get('type', '')
                
                base_labels = {
                    'hostname': hostname,
                    'component_name': component_name,
                    'component_type': component_type,
                    'stream_name': stream_name,
                    'object_name': object_name,
                    'object_type': object_type
                }
                
                # Handle MIP with seqrec object
                if object_type == 'seqrec':
                    self.mip_seqrec_level_gauge.labels(**base_labels).set(level)
                    self.mip_seqrec_recv_counter.labels(**base_labels).inc(recv)
                    self.mip_seqrec_send_counter.labels(**base_labels).inc(send)
                    
                    # Gauges
                    self.mip_seqrec_history_length_gauge.labels(**base_labels).set(obj.get('history_length', 0))
                    self.mip_seqrec_latent_error_paths_gauge.labels(**base_labels).set(obj.get('latent_error_paths', 0))
                    self.mip_seqrec_recovery_seq_num_gauge.labels(**base_labels).set(obj.get('recovery_seq_num', 0))
                    self.mip_seqrec_reset_msec_gauge.labels(**base_labels).set(obj.get('reset_msec', 0))
                    
                    # Counters
                    self.mip_seqrec_discarded_packets_counter.labels(**base_labels).inc(obj.get('discarded_packets', 0))
                    self.mip_seqrec_latent_error_resets_counter.labels(**base_labels).inc(obj.get('latent_error_resets', 0))
                    self.mip_seqrec_latent_errors_counter.labels(**base_labels).inc(obj.get('latent_errors', 0))
                    self.mip_seqrec_passed_packets_counter.labels(**base_labels).inc(obj.get('passed_packets', 0))
                    self.mip_seqrec_seq_recovery_resets_counter.labels(**base_labels).inc(obj.get('seq_recovery_resets', 0))
                    
                    # Enums
                    recovery_algo = obj.get('recovery_algorithm', 'vector')
                    self.mip_seqrec_recovery_algorithm.labels(**base_labels).state(recovery_algo)
                    
                    use_init = 'true' if obj.get('use_init_flag', False) else 'false'
                    self.mip_seqrec_use_init_flag.labels(**base_labels).state(use_init)
                    
                    use_reset = 'true' if obj.get('use_reset_flag', False) else 'false'
                    self.mip_seqrec_use_reset_flag.labels(**base_labels).state(use_reset)
                
                # Handle MIP with replicate object
                elif object_type == 'replicate':
                    self.mip_replicate_level_gauge.labels(**base_labels).set(level)
                    self.mip_replicate_recv_counter.labels(**base_labels).inc(recv)
                    self.mip_replicate_send_counter.labels(**base_labels).inc(send)
                    
                    # Counters
                    self.mip_replicate_octets_passed_counter.labels(**base_labels).inc(obj.get('octets_passed', 0))
                    self.mip_replicate_packets_passed_counter.labels(**base_labels).inc(obj.get('packets_passed', 0))
                    
                    # Handle pipelines
                    pipelines = obj.get('pipelines', [])
                    for pipeline in pipelines:
                        if not isinstance(pipeline, dict):
                            continue
                        
                        pipeline_name = pipeline.get('name', '')
                        pipeline_labels = {**base_labels, 'pipeline_name': pipeline_name}
                        
                        action_count = pipeline.get('action_count', 0)
                        self.mip_replicate_pipeline_action_count_gauge.labels(**pipeline_labels).set(action_count)
                        
                        mask_state = pipeline.get('mask_state', 'unmasked')
                        self.mip_replicate_pipeline_mask_state.labels(**pipeline_labels).state(mask_state)
            
            # Handle standalone seqrec components
            elif component_type == 'seqrec':
                seqrec_labels = {
                    'hostname': hostname,
                    'component_name': component_name,
                    'component_type': component_type,
                    'object_name': component_data.get('name', ''),
                    'object_type': component_type
                }
                
                # Gauges
                self.seqrec_history_length_gauge.labels(**seqrec_labels).set(component_data.get('history_length', 0))
                self.seqrec_latent_error_paths_gauge.labels(**seqrec_labels).set(component_data.get('latent_error_paths', 0))
                self.seqrec_recovery_seq_num_gauge.labels(**seqrec_labels).set(component_data.get('recovery_seq_num', 0))
                self.seqrec_reset_msec_gauge.labels(**seqrec_labels).set(component_data.get('reset_msec', 0))
                
                # Counters #TODO: Make sure to increase by change not by full value
                self.seqrec_discarded_packets_counter.labels(**seqrec_labels).inc(component_data.get('discarded_packets', 0))
                self.seqrec_latent_error_resets_counter.labels(**seqrec_labels).inc(component_data.get('latent_error_resets', 0))
                self.seqrec_latent_errors_counter.labels(**seqrec_labels).inc(component_data.get('latent_errors', 0))
                self.seqrec_passed_packets_counter.labels(**seqrec_labels).inc(component_data.get('passed_packets', 0))
                self.seqrec_seq_recovery_resets_counter.labels(**seqrec_labels).inc(component_data.get('seq_recovery_resets', 0))
                
                # Enums
                recovery_algo = component_data.get('recovery_algorithm', 'vector')
                self.seqrec_recovery_algorithm.labels(**seqrec_labels).state(recovery_algo)
                
                use_init = 'true' if component_data.get('use_init_flag', False) else 'false'
                self.seqrec_use_init_flag.labels(**seqrec_labels).state(use_init)
                
                use_reset = 'true' if component_data.get('use_reset_flag', False) else 'false'
                self.seqrec_use_reset_flag.labels(**seqrec_labels).state(use_reset)
            
            # Handle standalone replicate components
            elif component_type == 'replicate':
                replicate_labels = {
                    'hostname': hostname,
                    'component_name': component_name,
                    'component_type': component_type,
                    'object_name': component_data.get('name', ''),
                    'object_type': component_type
                }
                
                # Counters
                self.replicate_octets_passed_counter.labels(**replicate_labels).inc(component_data.get('octets_passed', 0))
                self.replicate_packets_passed_counter.labels(**replicate_labels).inc(component_data.get('packets_passed', 0))
                
                # Handle pipelines
                pipelines = component_data.get('pipelines', [])
                for pipeline in pipelines:
                    if not isinstance(pipeline, dict):
                        continue
                    
                    pipeline_name = pipeline.get('name', '')
                    pipeline_labels = {**replicate_labels, 'pipeline_name': pipeline_name}
                    
                    action_count = pipeline.get('action_count', 0)
                    self.replicate_pipeline_action_count_gauge.labels(**pipeline_labels).set(action_count)
                    
                    mask_state = pipeline.get('mask_state', 'unmasked')
                    self.replicate_pipeline_mask_state.labels(**pipeline_labels).state(mask_state)
            
            # Handle seqgen components
            elif component_type == 'seqgen':
                seqgen_labels = {
                    'hostname': hostname,
                    'component_name': component_name,
                    'component_type': component_type
                }
                
                use_init = 'true' if component_data.get('use_init_flag', False) else 'false'
                self.seqgen_use_init_flag.labels(**seqgen_labels).state(use_init)
                
                use_reset = 'true' if component_data.get('use_reset_flag', False) else 'false'
                self.seqgen_use_reset_flag.labels(**seqgen_labels).state(use_reset)
            
            # Handle interface notifications (ifNotify_*)
            elif component_name.startswith('ifNotify_'):
                interface_name = component_name.replace('ifNotify_', '')
                interface_labels = {
                    'hostname': hostname,
                    'interface_name': interface_name
                }
                
                self.interface_recv_octets_gauge.labels(**interface_labels).set(component_data.get('recv_octets', 0))
                self.interface_recv_packets_gauge.labels(**interface_labels).set(component_data.get('recv_packets', 0))
                self.interface_send_octets_gauge.labels(**interface_labels).set(component_data.get('send_octets', 0))
                self.interface_send_packets_gauge.labels(**interface_labels).set(component_data.get('send_packets', 0))