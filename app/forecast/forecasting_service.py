"""
Main forecasting application combining Prometheus data retrieval and LSTM prediction.
"""
import json
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from prometheus_client import start_http_server, Gauge
import logging

from prometheus_api import PrometheusClient
from lstm_forecaster import LSTMForecaster
from metrics_filter import filter_metrics, get_defined_metrics


logger = logging.getLogger(__name__)


class MetricsForecastService:
    """Service for forecasting network metrics using LSTM."""
    
    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        metrics_export_port: int = 19101,
        model_cache_dir: str = "./models"
    ):
        """
        Initialize the forecasting service.
        
        Args:
            prometheus_url: URL of Prometheus server
            metrics_export_port: Port for exporting forecast metrics
            model_cache_dir: Directory for caching trained models
        """
        self.prometheus_client = PrometheusClient(prometheus_url)
        self.metrics_export_port = metrics_export_port
        self.model_cache_dir = model_cache_dir
        self.models: Dict[str, LSTMForecaster] = {}
        self.forecast_metrics: Dict[str, Gauge] = {}
        
        # Create export metrics
        self._setup_export_metrics()
    
    def _setup_export_metrics(self):
        """Setup Prometheus metrics for exporting forecasts."""
        try:
            start_http_server(self.metrics_export_port)
            logger.info(f"Started metrics export server on port {self.metrics_export_port}")
        except OSError as e:
            logger.warning(f"Could not start metrics server on port {self.metrics_export_port}: {e}")
    
    def discover_metrics(self, filter_pattern: Optional[str] = None) -> List[str]:
        """
        Discover available metrics from Prometheus that are defined in MetricsExporter.
        
        Only metrics explicitly defined in MetricsExporter will be returned.
        
        Args:
            filter_pattern: Optional regex pattern to further filter metrics
        
        Returns:
            List of metric names defined in MetricsExporter
        """
        try:
            # Get all metrics from Prometheus
            all_metrics = self.prometheus_client.get_available_metrics()
            
            # Filter to only include metrics defined in MetricsExporter
            defined_only = filter_metrics(all_metrics)
            
            # Apply additional filter pattern if provided
            if filter_pattern:
                import re
                pattern = re.compile(filter_pattern)
                defined_only = [m for m in defined_only if pattern.search(m)]
            
            logger.info(f"Discovered {len(defined_only)} defined metrics from Prometheus "
                       f"(filtered from {len(all_metrics)} total metrics)")
            return defined_only
        
        except Exception as e:
            logger.error(f"Failed to discover metrics: {e}")
            return []
    
    def train_forecaster(
        self,
        metric_name: str,
        filters: Optional[Dict[str, str]] = None,
        history_hours: int = 24,
        sequence_length: int = 24,
        forecast_horizon: int = 6,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict:
        """
        Train an LSTM forecaster for a specific metric.
        
        Args:
            metric_name: Name of the Prometheus metric
            filters: Optional label filters
            history_hours: Hours of historical data to use
            sequence_length: Length of input sequences
            forecast_horizon: Number of steps to forecast
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training forecaster for metric: {metric_name}")
        
        # Fetch data from Prometheus
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=history_hours)
        
        try:
            data = self.prometheus_client.fetch_time_series(
                metric_name,
                start_time=start_time,
                end_time=end_time,
                step="15s",
                filters=filters
            )
        except Exception as e:
            logger.error(f"Failed to fetch data for {metric_name}: {e}")
            return {"status": "failed", "error": str(e)}
        
        if not data:
            logger.warning(f"No data available for metric: {metric_name}")
            return {"status": "no_data", "error": "No data available from Prometheus"}
        
        results = {}
        
        # Train a model for each time series
        for label_key, time_series in data.items():
            try:
                values = time_series['values']
                logger.info(f"Training on {label_key} with {len(values)} data points")
                
                # Create and train forecaster
                forecaster = LSTMForecaster(
                    sequence_length=sequence_length,
                    forecast_horizon=forecast_horizon
                )
                forecaster.build_model()
                
                metrics = forecaster.fit(
                    values,
                    epochs=epochs,
                    batch_size=batch_size,
                    early_stopping=True,
                    verbose=0
                )
                
                # Store the model
                model_key = f"{metric_name}:{label_key}"
                self.models[model_key] = forecaster
                
                results[label_key] = {
                    "status": "success",
                    "metrics": metrics,
                    "model_key": model_key
                }
                
                logger.info(f"Successfully trained model for {label_key}")
                
            except Exception as e:
                logger.error(f"Failed to train model for {label_key}: {e}")
                results[label_key] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        return results
    
    def forecast(
        self,
        metric_name: str,
        label_key: str,
        steps_ahead: Optional[int] = None,
        filters: Optional[Dict[str, str]] = None
    ) -> Optional[np.ndarray]:
        """
        Generate forecast for a specific metric.
        
        Args:
            metric_name: Name of the metric
            label_key: Label combination key
            steps_ahead: Number of steps to forecast
            filters: Label filters for data retrieval
        
        Returns:
            Array of forecast values or None if failed
        """
        model_key = f"{metric_name}:{label_key}"
        
        if model_key not in self.models:
            logger.warning(f"No trained model for {model_key}")
            return None
        
        try:
            # Fetch recent data for context
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            
            data = self.prometheus_client.fetch_time_series(
                metric_name,
                start_time=start_time,
                end_time=end_time,
                filters=filters
            )
            
            if label_key not in data:
                logger.warning(f"No recent data for {label_key}")
                return None
            
            values = data[label_key]['values']
            forecaster = self.models[model_key]
            forecast = forecaster.predict(values, steps_ahead)
            
            logger.info(f"Generated forecast for {model_key}: {forecast}")
            return forecast
        
        except Exception as e:
            logger.error(f"Failed to generate forecast for {model_key}: {e}")
            return None
    
    def get_available_metrics(self) -> List[str]:
        """Get list of metrics defined in MetricsExporter that are available in Prometheus."""
        try:
            metrics = self.discover_metrics()
            return metrics
        except Exception as e:
            logger.error(f"Failed to get available metrics: {e}")
            return []
    
    def get_defined_metrics(self) -> List[str]:
        """Get list of all metrics defined in MetricsExporter (regardless of Prometheus availability)."""
        return get_defined_metrics()
    
    def forecast_batch(
        self,
        metric_configs: List[Dict],
        steps_ahead: Optional[int] = None
    ) -> Dict:
        """
        Generate forecasts for multiple metrics in batch.
        
        Args:
            metric_configs: List of dicts with 'metric_name' and optional 'filters'
            steps_ahead: Number of steps to forecast
        
        Returns:
            Dictionary with forecast results
        """
        results = {}
        
        for config in metric_configs:
            metric_name = config.get('metric_name')
            filters = config.get('filters')
            
            if not metric_name:
                logger.warning("Skipping config without metric_name")
                continue
            
            logger.info(f"Processing forecasts for metric: {metric_name}")
            
            # Train if not already trained
            if not any(k.startswith(f"{metric_name}:") for k in self.models.keys()):
                train_results = self.train_forecaster(
                    metric_name,
                    filters=filters
                )
                results[f"{metric_name}:training"] = train_results
            
            # Generate forecasts
            metric_forecasts = {}
            for model_key in self.models.keys():
                if model_key.startswith(f"{metric_name}:"):
                    label_key = model_key.replace(f"{metric_name}:", "")
                    forecast = self.forecast(
                        metric_name,
                        label_key,
                        steps_ahead=steps_ahead,
                        filters=filters
                    )
                    
                    if forecast is not None:
                        metric_forecasts[label_key] = {
                            "forecast": forecast.tolist(),
                            "timestamp": datetime.now().isoformat()
                        }
            
            results[metric_name] = metric_forecasts
        
        return results
    
    def save_models(self, base_path: str = None):
        """Save all trained models."""
        import os
        
        if base_path is None:
            base_path = self.model_cache_dir
        
        os.makedirs(base_path, exist_ok=True)
        
        for model_key, forecaster in self.models.items():
            try:
                filepath = os.path.join(base_path, f"{model_key.replace(':', '_')}.h5")
                forecaster.save_model(filepath)
                logger.info(f"Saved model to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save model {model_key}: {e}")
    
    def load_models(self, base_path: str = None):
        """Load all trained models."""
        import os
        
        if base_path is None:
            base_path = self.model_cache_dir
        
        if not os.path.exists(base_path):
            logger.warning(f"Model cache directory does not exist: {base_path}")
            return
        
        for filename in os.listdir(base_path):
            if filename.endswith('.h5'):
                try:
                    model_key = filename[:-3].replace('_', ':')
                    filepath = os.path.join(base_path, filename)
                    
                    forecaster = LSTMForecaster()
                    forecaster.load_model(filepath)
                    
                    self.models[model_key] = forecaster
                    logger.info(f"Loaded model from {filepath}")
                except Exception as e:
                    logger.error(f"Failed to load model {filename}: {e}")


def main():
    """Example usage of the forecasting service."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Network metrics forecasting service')
    parser.add_argument(
        '--prometheus-url',
        default='http://localhost:9090',
        help='Prometheus server URL'
    )
    parser.add_argument(
        '--export-port',
        type=int,
        default=19101,
        help='Port for exporting forecast metrics'
    )
    parser.add_argument(
        '--metric',
        help='Specific metric to forecast'
    )
    parser.add_argument(
        '--list-metrics',
        action='store_true',
        help='List available metrics from Prometheus'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train forecasters for the specified metric'
    )
    parser.add_argument(
        '--config',
        help='Path to JSON config file for batch forecasting'
    )
    
    args = parser.parse_args()
    
    service = MetricsForecastService(
        prometheus_url=args.prometheus_url,
        metrics_export_port=args.export_port
    )
    
    if args.list_metrics:
        metrics = service.get_available_metrics()
        print("Available metrics:")
        for metric in sorted(metrics):
            print(f"  - {metric}")
    
    elif args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        results = service.forecast_batch(config.get('metrics', []))
        print(json.dumps(results, indent=2, default=str))
        
        service.save_models()
    
    elif args.metric:
        if args.train:
            results = service.train_forecaster(args.metric)
            print(f"Training results: {json.dumps(results, indent=2, default=str)}")
            service.save_models()


if __name__ == '__main__':
    main()
