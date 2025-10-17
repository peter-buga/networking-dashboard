"""
Flask API server for forecasting service.
Provides endpoints for predictions and metrics that Prometheus can scrape.
"""

import os
import json
import logging
from flask import Flask, jsonify, request
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from datetime import datetime

from forecasting_service import MetricsForecastService
from forecast_scheduler import ForecastScheduler


logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Prometheus metrics
forecast_predictions = Gauge(
    'forecast_prediction',
    'Forecasted values',
    labelnames=['metric', 'label', 'step']
)

forecast_accuracy = Gauge(
    'forecast_accuracy_r2',
    'R² score of forecast model',
    labelnames=['metric', 'label']
)

forecast_errors = Counter(
    'forecast_errors_total',
    'Total forecasting errors',
    labelnames=['metric', 'error_type']
)

training_duration = Histogram(
    'model_training_duration_seconds',
    'Time taken to train model',
    labelnames=['metric']
)

last_forecast_timestamp = Gauge(
    'forecast_last_timestamp',
    'Timestamp of last forecast generation',
    labelnames=['metric']
)

last_training_timestamp = Gauge(
    'model_last_training_timestamp',
    'Timestamp of last model training',
    labelnames=['metric']
)

scheduler_status = Gauge(
    'forecast_scheduler_status',
    'Scheduler status (1=running, 0=stopped)'
)

# Global service and scheduler
service: MetricsForecastService = None
scheduler: ForecastScheduler = None


def init_service(prometheus_url: str, metrics_export_port: int, model_cache_dir: str):
    """Initialize the forecasting service."""
    global service
    service = MetricsForecastService(
        prometheus_url=prometheus_url,
        metrics_export_port=metrics_export_port,
        model_cache_dir=model_cache_dir
    )
    return service


def init_scheduler(retrain_interval: str, forecast_interval: int):
    """Initialize and start the scheduler (auto-discovers metrics from Prometheus)."""
    global scheduler
    max_workers = int(os.getenv('MAX_WORKERS', '4'))
    scheduler = ForecastScheduler(
        service=service,
        retrain_interval=retrain_interval,
        forecast_interval=forecast_interval,
        max_workers=max_workers
    )
    scheduler.start()
    scheduler_status.set(1)
    return scheduler


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service_running": service is not None,
        "scheduler_running": scheduler.is_running if scheduler else False,
        "models_loaded": len(service.models) if service else 0,
    }), 200


@app.route('/status', methods=['GET'])
def status():
    """Get detailed status information."""
    if scheduler is None:
        return jsonify({"error": "Scheduler not initialized"}), 503
    
    status_info = scheduler.get_status()
    return jsonify(status_info), 200


# ============================================================================
# Forecast Endpoints
# ============================================================================

@app.route('/forecast/<metric_name>', methods=['GET'])
def get_forecast(metric_name):
    """
    Get forecast for a specific metric.
    
    Query parameters:
        - label_key: Label combination key (required)
        - steps_ahead: Number of steps to forecast (optional, default: 6)
    """
    if service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    label_key = request.args.get('label_key')
    steps_ahead = request.args.get('steps_ahead', type=int)
    
    if not label_key:
        return jsonify({"error": "label_key parameter required"}), 400
    
    try:
        forecast = service.forecast(
            metric_name=metric_name,
            label_key=label_key,
            steps_ahead=steps_ahead
        )
        
        if forecast is None:
            return jsonify({
                "error": f"No forecast available for {metric_name}:{label_key}"
            }), 404
        
        # Update Prometheus metrics
        for i, pred in enumerate(forecast, 1):
            forecast_predictions.labels(
                metric=metric_name,
                label=label_key,
                step=str(i)
            ).set(float(pred))
        
        last_forecast_timestamp.labels(metric=metric_name).set(datetime.now().timestamp())
        
        return jsonify({
            "metric": metric_name,
            "label_key": label_key,
            "forecast": forecast.tolist(),
            "timestamp": datetime.now().isoformat(),
            "steps": len(forecast)
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting forecast for {metric_name}: {e}")
        forecast_errors.labels(metric=metric_name, error_type="forecast").inc()
        return jsonify({"error": str(e)}), 500


@app.route('/forecast/batch', methods=['POST'])
def forecast_batch():
    """
    Generate forecasts for multiple metrics.
    
    JSON body:
        {
            "metrics": [
                {"metric_name": "interface_recv_packets"},
                {"metric_name": "interface_send_packets"}
            ],
            "steps_ahead": 6
        }
    """
    if service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        data = request.get_json()
        metrics_config = data.get('metrics', [])
        steps_ahead = data.get('steps_ahead')
        
        if not metrics_config:
            return jsonify({"error": "metrics list required"}), 400
        
        results = service.forecast_batch(metrics_config, steps_ahead)
        
        return jsonify({
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "count": len(metrics_config)
        }), 200
    
    except Exception as e:
        logger.error(f"Error in batch forecast: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Training Endpoints
# ============================================================================

@app.route('/train/<metric_name>', methods=['POST'])
def train_metric(metric_name):
    """
    Train a forecaster for a specific metric.
    
    JSON body (optional):
        {
            "filters": {"hostname": "edge1"},
            "history_hours": 24,
            "epochs": 50,
            "batch_size": 32
        }
    """
    if service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        data = request.get_json() or {}
        
        with training_duration.labels(metric=metric_name).time():
            results = service.train_forecaster(
                metric_name=metric_name,
                filters=data.get('filters'),
                history_hours=data.get('history_hours', 1),
                epochs=data.get('epochs', 20),
                batch_size=data.get('batch_size', 128)
            )
        
        # Update metrics
        last_training_timestamp.labels(metric=metric_name).set(datetime.now().timestamp())
        
        # Count successes/failures
        success_count = sum(1 for r in results.values() if r.get('status') == 'success')
        failed_count = len(results) - success_count
        
        if failed_count > 0:
            forecast_errors.labels(metric=metric_name, error_type="training").inc(failed_count)
        
        return jsonify({
            "metric": metric_name,
            "results": results,
            "summary": {
                "total": len(results),
                "success": success_count,
                "failed": failed_count
            },
            "timestamp": datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error training {metric_name}: {e}")
        forecast_errors.labels(metric=metric_name, error_type="training").inc()
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Model Management Endpoints
# ============================================================================

@app.route('/models', methods=['GET'])
def list_models():
    """List all loaded models."""
    if service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    return jsonify({
        "models": list(service.models.keys()),
        "count": len(service.models),
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/models/save', methods=['POST'])
def save_models():
    """Save all trained models."""
    if service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        service.save_models()
        return jsonify({
            "status": "success",
            "models_saved": len(service.models),
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/models/load', methods=['POST'])
def load_models():
    """Load pre-trained models."""
    if service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        service.load_models()
        return jsonify({
            "status": "success",
            "models_loaded": len(service.models),
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Scheduler Management Endpoints
# ============================================================================

@app.route('/scheduler/status', methods=['GET'])
def scheduler_status_endpoint():
    """Get scheduler status."""
    if scheduler is None:
        return jsonify({"error": "Scheduler not initialized"}), 503
    
    return jsonify(scheduler.get_status()), 200


@app.route('/scheduler/stop', methods=['POST'])
def stop_scheduler():
    """Stop the scheduler."""
    if scheduler is None:
        return jsonify({"error": "Scheduler not initialized"}), 503
    
    scheduler.stop()
    scheduler_status.set(0)
    return jsonify({
        "status": "stopped",
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/scheduler/start', methods=['POST'])
def start_scheduler():
    """Start the scheduler."""
    if scheduler is None:
        return jsonify({"error": "Scheduler not initialized"}), 503
    
    if scheduler.is_running:
        return jsonify({"error": "Scheduler already running"}), 400
    
    scheduler.start()
    scheduler_status.set(1)
    return jsonify({
        "status": "started",
        "timestamp": datetime.now().isoformat()
    }), 200


# ============================================================================
# Metrics Endpoint
# ============================================================================

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}


# ============================================================================
# Utility Endpoints
# ============================================================================

@app.route('/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    if scheduler is None:
        return jsonify({"error": "Scheduler not initialized"}), 503
    
    return jsonify({
        "retrain_interval": scheduler.retrain_interval,
        "forecast_interval": scheduler.forecast_interval,
        "metrics_configured": len(scheduler.metrics_config),
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/available-metrics', methods=['GET'])
def available_metrics():
    """Get available metrics from Prometheus."""
    if service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        metrics = service.get_available_metrics()
        return jsonify({
            "metrics": sorted(metrics),
            "count": len(metrics),
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error getting available metrics: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/defined-metrics', methods=['GET'])
def defined_metrics():
    """Get all metrics defined in MetricsExporter."""
    if service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        defined = service.get_defined_metrics()
        return jsonify({
            "metrics": sorted(defined),
            "count": len(defined),
            "description": "All metrics defined in MetricsExporter that will be forecasted",
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error getting defined metrics: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Endpoint not found",
        "path": request.path,
        "method": request.method
    }), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    logger.error(f"Server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration from environment
    prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
    api_port = int(os.getenv('API_PORT', 5000))
    api_host = os.getenv('API_HOST', '0.0.0.0')
    metrics_export_port = int(os.getenv('METRICS_EXPORT_PORT', 19101))
    model_cache_dir = os.getenv('MODEL_CACHE_DIR', './models')
    retrain_interval = os.getenv('RETRAIN_INTERVAL', 'weekly')
    forecast_interval = int(os.getenv('FORECAST_INTERVAL', '300'))
    
    # Initialize service and scheduler (metrics auto-discovered from Prometheus)
    logger.info(f"Initializing forecasting service (Prometheus: {prometheus_url})")
    init_service(prometheus_url, metrics_export_port, model_cache_dir)
    
    logger.info(f"Initializing scheduler (retrain: {retrain_interval}, forecast: {forecast_interval}s)")
    logger.info("Metrics will be auto-discovered from Prometheus")
    init_scheduler(retrain_interval, forecast_interval)
    
    # Start Flask app
    logger.info(f"Starting API server on {api_host}:{api_port}")
    app.run(host=api_host, port=api_port, debug=False)
