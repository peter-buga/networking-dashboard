"""
Metrics filter that extracts metric names from MetricsExporter definition.
This ensures the forecaster only trains on metrics that are actually exported.
"""

# All metric names defined in MetricsExporter
# Source: app/json_receiver/metrics_exporter.py
DEFINED_METRICS = {
    # SequenceRecovery metrics
    'seqrec_discarded_packets',
    'seqrec_history_length',
    'seqrec_latent_error_paths',
    'seqrec_latent_error_resets',
    'seqrec_latent_errors',
    'seqrec_passed_packets',
    'seqrec_recovery_algorithm',
    'seqrec_use_init_flag',
    'seqrec_use_reset_flag',
    'seqrec_seq_recovery_resets',
    'seqrec_recovery_seq_num',
    'seqrec_reset_msec',
    
    # MIP - SequenceRecovery metrics
    'mip_seqrec_level',
    'mip_seqrec_history_length',
    'mip_seqrec_latent_error_paths',
    'mip_seqrec_recovery_seq_num',
    'mip_seqrec_reset_msec',
    'mip_seqrec_discarded_packets',
    'mip_seqrec_latent_error_resets',
    'mip_seqrec_latent_errors',
    'mip_seqrec_passed_packets',
    'mip_seqrec_seq_recovery_resets',
    'mip_seqrec_recv',
    'mip_seqrec_send',
    'mip_seqrec_recovery_algorithm',
    'mip_seqrec_use_init_flag',
    'mip_seqrec_use_reset_flag',
    
    # Replicate metrics
    'replicate_octets_passed',
    'replicate_packets_passed',
    'replicate_pipeline_action_count',
    'replicate_pipeline_mask_state',
    
    # MIP - Replicate metrics
    'mip_replicate_level',
    'mip_replicate_octets_passed',
    'mip_replicate_packets_passed',
    'mip_replicate_recv',
    'mip_replicate_send',
    'mip_replicate_pipeline_action_count',
    'mip_replicate_pipeline_mask_state',
    
    # SequenceGenerator metrics
    'seqgen_use_init_flag',
    'seqgen_use_reset_flag',
    
    # Interface metrics
    'interface_recv_octets',
    'interface_recv_packets',
    'interface_send_octets',
    'interface_send_packets',
}


def get_defined_metrics() -> list:
    """
    Get list of metrics defined in MetricsExporter.
    
    Returns:
        List of metric names that should be forecasted
    """
    return sorted(list(DEFINED_METRICS))


def is_metric_defined(metric_name: str) -> bool:
    """
    Check if a metric is defined in MetricsExporter.
    
    Args:
        metric_name: Name of the metric to check
    
    Returns:
        True if metric is defined, False otherwise
    """
    # Handle potential suffixes added by Prometheus (e.g., _total for counters)
    metric_base = metric_name.rstrip('_total')
    return metric_base in DEFINED_METRICS


def filter_metrics(all_metrics: list) -> list:
    """
    Filter metrics to only include those defined in MetricsExporter.
    
    Args:
        all_metrics: List of all available metrics from Prometheus
    
    Returns:
        Filtered list of metrics that are defined in MetricsExporter
    """
    filtered = []
    for metric in all_metrics:
        if is_metric_defined(metric):
            filtered.append(metric)
    return filtered
