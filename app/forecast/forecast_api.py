"""HTTP interface exposing forecast operations for orchestration and scrapes."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from flask import Flask, Response, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST

from .forecast_scheduler import ForecastScheduler
from .forecasting_service import MetricsForecastService
from .lstm_forecaster import LSTMForecaster
from .metrics_filter import MetricsFilter
from .prometheus_query import PrometheusClient


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger('app.forecast')
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(name)s %(message)s',
        )
    level_name = os.getenv('FORECAST_LOG_LEVEL') or 'INFO'
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    logger.info('Forecast logging configured at %s', level_name.upper())
    return logger


logger = _configure_logging()

app = Flask(__name__)


def create_service() -> MetricsForecastService:
    logger.info('Initialising forecast service components')
    forecaster = LSTMForecaster()
    prometheus_url = os.getenv('PROMETHEUS_BASE_URL', 'http://localhost:9090')
    prometheus = PrometheusClient(prometheus_url)
    filter_path = Path(os.getenv('FORECAST_FILTER_CONFIG', Path(__file__).with_name('metrics_filter.yaml')))
    metrics_filter = MetricsFilter(filter_path)
    service = MetricsForecastService(forecaster, prometheus, metrics_filter)
    logger.info('Forecast service ready (Prometheus=%s)', prometheus_url)
    return service


service = create_service()
scheduler = ForecastScheduler(
    service=service,
    prometheus=service.prometheus,
    metrics_filter=service.metrics_filter,
    max_concurrency=service.forecaster.max_parallel_jobs,
    retrain_days=service.forecaster.retrain_days,
)
scheduler.start()


@app.route('/forecast/predict', methods=['POST'])
def forecast_predict() -> Response:
    payload = request.get_json(force=True)
    metric = payload.get('metric')
    labels = payload.get('labels', {})
    if not metric or not isinstance(labels, dict):
        return jsonify({'error': 'metric and labels are required'}), 400
    logger.info("/forecast/predict metric=%s labels=%s force=%s", metric, labels, payload.get('force_retrain'))

    if payload.get('force_retrain'):
        try:
            service.trigger_retrain(metric, labels)
        except Exception as exc:  # pragma: no cover - external dependency
            logger.warning("Forced retrain failed for %s %s: %s", metric, labels, exc)

    try:
        result = service.get_forecast(metric, labels)
    except Exception as exc:  # pragma: no cover - external dependency
        logger.error("Forecast failed for %s %s: %s", metric, labels, exc)
        return jsonify({'error': str(exc), 'metric': metric, 'labels': labels}), 500
    return jsonify(result)


@app.route('/health', methods=['GET'])
def health() -> Response:
    healthy = service.prometheus.is_healthy()
    status_code = 200 if healthy else 503
    return jsonify({'status': 'ok' if healthy else 'degraded'}), status_code


@app.route('/metrics', methods=['GET'])
def metrics() -> Response:
    payload = service.prometheus_payload()
    return Response(payload, mimetype=CONTENT_TYPE_LATEST)


if __name__ == '__main__':  # pragma: no cover - manual execution entry point
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    scheduler.start()
    app.run(host='0.0.0.0', port=int(os.getenv('FORECAST_API_PORT', 8080)))

