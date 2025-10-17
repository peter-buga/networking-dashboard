"""
Prometheus data client for fetching time series data.
"""
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np


class PrometheusClient:
    """Client for querying Prometheus time series data."""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        """
        Initialize Prometheus client.
        
        Args:
            prometheus_url: URL of the Prometheus server
        """
        self.base_url = prometheus_url
        self.query_url = f"{self.base_url}/api/v1/query_range"
    
    def fetch_time_series(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        step: str = "15s",
        filters: Optional[Dict[str, str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Fetch time series data from Prometheus.
        
        Args:
            metric_name: Name of the metric to fetch
            start_time: Start time for the range (default: 24 hours ago)
            end_time: End time for the range (default: now)
            step: Step size for the time series (default: 15s)
            filters: Optional dict of label filters (e.g., {'hostname': 'edge1'})
        
        Returns:
            Dictionary mapping label values to numpy arrays of [timestamp, value] pairs
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        # Build the query with filters
        query = self._build_query(metric_name, filters)
        
        params = {
            'query': query,
            'start': int(start_time.timestamp()),
            'end': int(end_time.timestamp()),
            'step': step
        }
        
        try:
            response = requests.get(self.query_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data['status'] != 'success':
                raise ValueError(f"Prometheus query failed: {data.get('error', 'Unknown error')}")
            
            return self._parse_response(data)
        
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Prometheus: {e}")
    
    def _build_query(self, metric_name: str, filters: Optional[Dict[str, str]] = None) -> str:
        """Build PromQL query with optional filters."""
        if not filters:
            return metric_name
        
        filter_parts = [f'{key}="{value}"' for key, value in filters.items()]
        return f"{metric_name}{{{','.join(filter_parts)}}}"
    
    def _parse_response(self, response: Dict) -> Dict[str, np.ndarray]:
        """
        Parse Prometheus response into structured data.
        
        Returns:
            Dictionary mapping label combinations to (timestamps, values) arrays
        """
        result = {}
        
        for time_series in response.get('data', {}).get('result', []):
            # Create a label key for this time series
            labels = time_series.get('metric', {})
            label_key = self._create_label_key(labels)
            
            # Extract timestamps and values
            values = time_series.get('values', [])
            if values:
                timestamps = np.array([float(v[0]) for v in values])
                data_values = np.array([float(v[1]) if v[1] != 'NaN' else np.nan for v in values])
                
                result[label_key] = {
                    'timestamps': timestamps,
                    'values': data_values,
                    'labels': labels
                }
        
        return result
    
    def _create_label_key(self, labels: Dict[str, str]) -> str:
        """Create a unique key from labels."""
        if not labels:
            return "default"
        
        # Sort labels for consistent key generation
        sorted_labels = sorted(labels.items())
        return "|".join([f"{k}={v}" for k, v in sorted_labels if k != '__name__'])
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics from Prometheus."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/label/__name__/values",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'success':
                return data.get('data', [])
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch available metrics: {e}")
        
        return []
    
    def get_metric_labels(self, metric_name: str) -> Dict[str, List[str]]:
        """Get available label values for a metric."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/series",
                params={'match[]': metric_name},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'success':
                labels_dict = {}
                for series in data.get('data', []):
                    for label_name, label_value in series.items():
                        if label_name != '__name__':
                            if label_name not in labels_dict:
                                labels_dict[label_name] = set()
                            labels_dict[label_name].add(label_value)
                
                # Convert sets to sorted lists
                return {k: sorted(list(v)) for k, v in labels_dict.items()}
        
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch metric labels: {e}")
        
        return {}
