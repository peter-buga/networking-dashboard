from prometheus_client import Counter, Gauge, Enum, start_http_server
from typing import Optional, Dict, Any

class MetricsExporter:
    """Encapsulate Prometheus metrics registration and updates."""
    def __init__(self, host: str, port: int, disabled: bool):
        self.enabled = False
        self._last_timestamp_gauge: Optional[Gauge] = None
        self._previous_values: Dict[str, Dict[str, int]] = {} 
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
        self.seqrec_recovery_algorithm = Gauge('seqrec_recovery_algorithm', 'Algorithm used for sequence recovery',
                                              labelnames=labels + ['algorithm'])
        self.seqrec_use_init_flag = Gauge('seqrec_use_init_flag', "Use init flag",
                                        labelnames=labels)
        self.seqrec_use_reset_flag = Gauge('seqrec_use_reset_flag', "Use reset flag",
                                        labelnames=labels)
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

    
        self.mip_seqrec_recovery_algorithm = Gauge('mip_seqrec_recovery_algorithm', 'Algorithm used for sequence recovery',
                                                  labelnames=seqrec_mip_labels + ['algorithm'])
        self.mip_seqrec_use_init_flag = Gauge('mip_seqrec_use_init_flag', "Use init flag",
                                             labelnames=seqrec_mip_labels)
        self.mip_seqrec_use_reset_flag = Gauge('mip_seqrec_use_reset_flag', "Use reset flag",
                                             labelnames=seqrec_mip_labels)


        #Replicate
        replicate_labels = labels + ['pipeline_name']

        self.replicate_octets_passed_counter = Counter('replicate_octets_passed', 'Number of octets passed', labels)
        self.replicate_packets_passed_counter = Counter('replicate_packets_passed', 'Number of packets passed', labels)
        self.replicate_pipeline_action_count_gauge = Gauge('replicate_pipeline_action_count', 'Action count per pipeline', replicate_labels)
        self.replicate_pipeline_mask_state = Gauge('replicate_pipeline_mask_state', 'Mask state of the pipeline',
                                                  labelnames=replicate_labels + ['state'])

        #MIP - Replicate
        replicate_mip_labels = ['hostname','component_name','component_type','stream_name','object_name', 'object_type']
        replicate_mip_pipeline_labels = replicate_mip_labels + ['pipeline_name']
        
        self.mip_replicate_level_gauge = Gauge('mip_replicate_level', 'Current Replicate level', replicate_mip_labels)
        self.mip_replicate_octets_passed_counter = Counter('mip_replicate_octets_passed', 'Number of octets passed', replicate_mip_labels)
        self.mip_replicate_packets_passed_counter = Counter('mip_replicate_packets_passed', 'Number of packets passed', replicate_mip_labels)
        self.mip_replicate_recv_counter = Counter('mip_replicate_recv', 'Number of received items for replicate', replicate_mip_labels)
        self.mip_replicate_send_counter = Counter('mip_replicate_send', 'Number of sent items for replicate', replicate_mip_labels)
        self.mip_replicate_pipeline_action_count_gauge = Gauge('mip_replicate_pipeline_action_count', 'Action count per pipeline', replicate_mip_pipeline_labels)
        self.mip_replicate_pipeline_mask_state = Gauge('mip_replicate_pipeline_mask_state', 'Mask state of the pipeline',
                                                     labelnames=replicate_mip_pipeline_labels + ['state'])

        # SequenceGenerator
        seqgen_labels = ['hostname','component_name','component_type']

        self.seqgen_use_init_flag = Gauge('seqgen_use_init_flag', "Use init flag",
                                        labelnames=seqgen_labels)
        self.seqgen_use_reset_flag = Gauge('seqgen_use_reset_flag', "Use reset flag",
                                        labelnames=seqgen_labels)
        
        #Interfaces
        interface_labels = ['hostname','interface_name']

        self.interface_recv_octets_gauge = Gauge('interface_recv_octets', 'Number of octets received on interface', interface_labels)
        self.interface_recv_packets_gauge = Gauge('interface_recv_packets', 'Number of packets received on interface', interface_labels)
        self.interface_send_octets_gauge = Gauge('interface_send_octets', 'Number of octets sent on interface', interface_labels)
        self.interface_send_packets_gauge = Gauge('interface_send_packets', 'Number of packets sent on interface', interface_labels)

        #Stream Parsers
        parser_labels = ['hostname','stream_name']
        self.parser_no_match_octets_gauge = Gauge('parser_no_match_octets',"Number of no match octets for stream", parser_labels)
        self.parser_no_match_packets_gauge = Gauge('parser_no_match_packets',"Number of no match packets for stream", parser_labels)
        self.parser_stream_octets_gauge = Gauge('parser_stream_octets',"Number of stream octets", parser_labels)
        self.parser_stream_packets_gauge = Gauge('parser_stream_packets',"Number of stream packets", parser_labels)
        
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
                    
                    # Calculate delta for recv/send
                    label_key = str(sorted(base_labels.items()))
                    recv_send_key = f"{label_key}:recv_send"
                    if recv_send_key not in self._previous_values:
                        self._previous_values[recv_send_key] = {'recv': recv, 'send': send}
                        recv_delta = 0
                        send_delta = 0
                    else:
                        recv_delta = max(0, recv - self._previous_values[recv_send_key]['recv'])
                        send_delta = max(0, send - self._previous_values[recv_send_key]['send'])
                        self._previous_values[recv_send_key] = {'recv': recv, 'send': send}
                    
                    self.mip_seqrec_recv_counter.labels(**base_labels).inc(recv_delta)
                    self.mip_seqrec_send_counter.labels(**base_labels).inc(send_delta)
                    
                    # Gauges
                    self.mip_seqrec_history_length_gauge.labels(**base_labels).set(obj.get('history_length', 0))
                    self.mip_seqrec_latent_error_paths_gauge.labels(**base_labels).set(obj.get('latent_error_paths', 0))
                    self.mip_seqrec_recovery_seq_num_gauge.labels(**base_labels).set(obj.get('recovery_seq_num', 0))
                    self.mip_seqrec_reset_msec_gauge.labels(**base_labels).set(obj.get('reset_msec', 0))
                    
                    # Counters - track deltas for rolling sums
                    counter_key = f"{label_key}:counters"
                    if counter_key not in self._previous_values:
                        self._previous_values[counter_key] = {
                            'discarded_packets': obj.get('discarded_packets', 0),
                            'latent_error_resets': obj.get('latent_error_resets', 0),
                            'latent_errors': obj.get('latent_errors', 0),
                            'passed_packets': obj.get('passed_packets', 0),
                            'seq_recovery_resets': obj.get('seq_recovery_resets', 0)
                        }
                    else:
                        discarded_packets_delta = max(0, obj.get('discarded_packets', 0) - self._previous_values[counter_key]['discarded_packets'])
                        latent_error_resets_delta = max(0, obj.get('latent_error_resets', 0) - self._previous_values[counter_key]['latent_error_resets'])
                        latent_errors_delta = max(0, obj.get('latent_errors', 0) - self._previous_values[counter_key]['latent_errors'])
                        passed_packets_delta = max(0, obj.get('passed_packets', 0) - self._previous_values[counter_key]['passed_packets'])
                        seq_recovery_resets_delta = max(0, obj.get('seq_recovery_resets', 0) - self._previous_values[counter_key]['seq_recovery_resets'])
                        
                        self._previous_values[counter_key] = {
                            'discarded_packets': obj.get('discarded_packets', 0),
                            'latent_error_resets': obj.get('latent_error_resets', 0),
                            'latent_errors': obj.get('latent_errors', 0),
                            'passed_packets': obj.get('passed_packets', 0),
                            'seq_recovery_resets': obj.get('seq_recovery_resets', 0)
                        }
                        
                        self.mip_seqrec_discarded_packets_counter.labels(**base_labels).inc(discarded_packets_delta)
                        self.mip_seqrec_latent_error_resets_counter.labels(**base_labels).inc(latent_error_resets_delta)
                        self.mip_seqrec_latent_errors_counter.labels(**base_labels).inc(latent_errors_delta)
                        self.mip_seqrec_passed_packets_counter.labels(**base_labels).inc(passed_packets_delta)
                        self.mip_seqrec_seq_recovery_resets_counter.labels(**base_labels).inc(seq_recovery_resets_delta)
                    
                    # Enums
                    recovery_algo = obj.get('recovery_algorithm', 'vector')
                    self.mip_seqrec_recovery_algorithm.labels(**base_labels, algorithm=recovery_algo).set(1)
                    
                    self.mip_seqrec_use_init_flag.labels(**base_labels).set(1 if obj.get('use_init_flag', False) else 0)
                    self.mip_seqrec_use_reset_flag.labels(**base_labels).set(1 if obj.get('use_reset_flag', False) else 0)
                
                # Handle MIP with replicate object
                elif object_type == 'replicate':
                    self.mip_replicate_level_gauge.labels(**base_labels).set(level)
                    
                    # Calculate delta for recv/send (rolling sum -> delta increment)
                    label_key = str(sorted(base_labels.items()))
                    recv_send_key = f"{label_key}:recv_send"
                    if recv_send_key not in self._previous_values:
                        self._previous_values[recv_send_key] = {'recv': recv, 'send': send}
                        recv_delta = 0
                        send_delta = 0
                    else:
                        recv_delta = max(0, recv - self._previous_values[recv_send_key]['recv'])
                        send_delta = max(0, send - self._previous_values[recv_send_key]['send'])
                        self._previous_values[recv_send_key] = {'recv': recv, 'send': send}
                    
                    self.mip_replicate_recv_counter.labels(**base_labels).inc(recv_delta)
                    self.mip_replicate_send_counter.labels(**base_labels).inc(send_delta)
                    
                    # Counters - track deltas for rolling sums
                    counter_key = f"{label_key}:counters"
                    if counter_key not in self._previous_values:
                        self._previous_values[counter_key] = {
                            'octets_passed': obj.get('octets_passed', 0),
                            'packets_passed': obj.get('packets_passed', 0)
                        }
                    else:
                        octets_passed_delta = max(0, obj.get('octets_passed', 0) - self._previous_values[counter_key]['octets_passed'])
                        packets_passed_delta = max(0, obj.get('packets_passed', 0) - self._previous_values[counter_key]['packets_passed'])
                        
                        self._previous_values[counter_key] = {
                            'octets_passed': obj.get('octets_passed', 0),
                            'packets_passed': obj.get('packets_passed', 0)
                        }
                        
                        self.mip_replicate_octets_passed_counter.labels(**base_labels).inc(octets_passed_delta)
                        self.mip_replicate_packets_passed_counter.labels(**base_labels).inc(packets_passed_delta)
                    
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
                        self.mip_replicate_pipeline_mask_state.labels(**pipeline_labels, state=mask_state).set(1)
            
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
                
                # Counters - track deltas for rolling sums
                label_key = str(sorted(seqrec_labels.items()))
                counter_key = f"{label_key}:counters"
                if counter_key not in self._previous_values:
                    self._previous_values[counter_key] = {
                        'discarded_packets': component_data.get('discarded_packets', 0),
                        'latent_error_resets': component_data.get('latent_error_resets', 0),
                        'latent_errors': component_data.get('latent_errors', 0),
                        'passed_packets': component_data.get('passed_packets', 0),
                        'seq_recovery_resets': component_data.get('seq_recovery_resets', 0)
                    }
                else:
                    discarded_packets_delta = max(0, component_data.get('discarded_packets', 0) - self._previous_values[counter_key]['discarded_packets'])
                    latent_error_resets_delta = max(0, component_data.get('latent_error_resets', 0) - self._previous_values[counter_key]['latent_error_resets'])
                    latent_errors_delta = max(0, component_data.get('latent_errors', 0) - self._previous_values[counter_key]['latent_errors'])
                    passed_packets_delta = max(0, component_data.get('passed_packets', 0) - self._previous_values[counter_key]['passed_packets'])
                    seq_recovery_resets_delta = max(0, component_data.get('seq_recovery_resets', 0) - self._previous_values[counter_key]['seq_recovery_resets'])
                    
                    self._previous_values[counter_key] = {
                        'discarded_packets': component_data.get('discarded_packets', 0),
                        'latent_error_resets': component_data.get('latent_error_resets', 0),
                        'latent_errors': component_data.get('latent_errors', 0),
                        'passed_packets': component_data.get('passed_packets', 0),
                        'seq_recovery_resets': component_data.get('seq_recovery_resets', 0)
                    }
                    
                    self.seqrec_discarded_packets_counter.labels(**seqrec_labels).inc(discarded_packets_delta)
                    self.seqrec_latent_error_resets_counter.labels(**seqrec_labels).inc(latent_error_resets_delta)
                    self.seqrec_latent_errors_counter.labels(**seqrec_labels).inc(latent_errors_delta)
                    self.seqrec_passed_packets_counter.labels(**seqrec_labels).inc(passed_packets_delta)
                    self.seqrec_seq_recovery_resets_counter.labels(**seqrec_labels).inc(seq_recovery_resets_delta)
                
                # Enums
                recovery_algo = component_data.get('recovery_algorithm', '')
                self.seqrec_recovery_algorithm.labels(**seqrec_labels, algorithm=recovery_algo).set(1)
                
                self.seqrec_use_init_flag.labels(**seqrec_labels).set(1 if component_data.get('use_init_flag', False) else 0)
                self.seqrec_use_reset_flag.labels(**seqrec_labels).set(1 if component_data.get('use_reset_flag', False) else 0)
            
            # Handle standalone replicate components
            elif component_type == 'replicate':
                replicate_labels = {
                    'hostname': hostname,
                    'component_name': component_name,
                    'component_type': component_type,
                    'object_name': component_data.get('name', ''),
                    'object_type': component_type
                }
                
                # Counters - track deltas for rolling sums
                label_key = str(sorted(replicate_labels.items()))
                counter_key = f"{label_key}:counters"
                if counter_key not in self._previous_values:
                    self._previous_values[counter_key] = {
                        'octets_passed': component_data.get('octets_passed', 0),
                        'packets_passed': component_data.get('packets_passed', 0)
                    }
                else:
                    octets_passed_delta = max(0, component_data.get('octets_passed', 0) - self._previous_values[counter_key]['octets_passed'])
                    packets_passed_delta = max(0, component_data.get('packets_passed', 0) - self._previous_values[counter_key]['packets_passed'])
                    
                    self._previous_values[counter_key] = {
                        'octets_passed': component_data.get('octets_passed', 0),
                        'packets_passed': component_data.get('packets_passed', 0)
                    }
                    
                    self.replicate_octets_passed_counter.labels(**replicate_labels).inc(octets_passed_delta)
                    self.replicate_packets_passed_counter.labels(**replicate_labels).inc(packets_passed_delta)
                
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
                    self.replicate_pipeline_mask_state.labels(**pipeline_labels, state=mask_state).set(1)
            
            # Handle seqgen components
            elif component_type == 'seqgen':
                seqgen_labels = {
                    'hostname': hostname,
                    'component_name': component_name,
                    'component_type': component_type
                }
                
                self.seqgen_use_init_flag.labels(**seqgen_labels).set(1 if component_data.get('use_init_flag', False) else 0)
                self.seqgen_use_reset_flag.labels(**seqgen_labels).set(1 if component_data.get('use_reset_flag', False) else 0)
            
            # Handle parsers
            elif "parser" in component_name:
                stream_name = component_name.replace(" parser","")
                parser_labels = {
                    'hostname': hostname,
                    'stream_name': stream_name
                }
                self.parser_no_match_octets_gauge.labels(**parser_labels).set(component_data.get('no_match_octets', 0))
                self.parser_no_match_packets_gauge.labels(**parser_labels).set(component_data.get('no_match_packets', 0))
                self.parser_stream_octets_gauge.labels(**parser_labels).set(component_data.get('stream_octets', 0))
                self.parser_stream_packets_gauge.labels(**parser_labels).set(component_data.get('stream_packets', 0))

            # Handle interfaces
            else:
                interface_name = component_name
                interface_labels = {
                    'hostname': hostname,
                    'interface_name': interface_name
                }
                
                self.interface_recv_octets_gauge.labels(**interface_labels).set(component_data.get('recv_octets', 0))
                self.interface_recv_packets_gauge.labels(**interface_labels).set(component_data.get('recv_packets', 0))
                self.interface_send_octets_gauge.labels(**interface_labels).set(component_data.get('send_octets', 0))
                self.interface_send_packets_gauge.labels(**interface_labels).set(component_data.get('send_packets', 0))