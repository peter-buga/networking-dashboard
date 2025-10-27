"""Orchestration layer exposing forecast capabilities to the rest of the app."""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, Optional

import pandas as pd
from prometheus_client import generate_latest

from .arima_forecaster import ARIMAForecaster
from .forecast_collector import ForecastCollector, get_forecast_collector
from .metrics_filter import MetricsFilter
from .prometheus_query import PrometheusClient

logger = logging.getLogger(__name__)


class MetricsForecastService:
    def __init__(
        self,
        forecaster: ARIMAForecaster,
        prometheus: PrometheusClient,
        metrics_filter: MetricsFilter,
    ) -> None:
        self.forecaster = forecaster
        self.prometheus = prometheus
        self.metrics_filter = metrics_filter
        self._collector: ForecastCollector = get_forecast_collector()

    def get_forecast(self, metric: str, labels: Dict[str, str]) -> Dict[str, object]:
        if self.metrics_filter.filter_metrics([metric]) == []:
            logger.debug("Metric %s ignored by filter", metric)
            return {'status': 'filtered_out', 'metric': metric, 'labels': labels}

        try:
            history = self._fetch_history(metric, labels)
            logger.info("Fetched history for %s %s: %d points", metric, labels, len(history))
        except Exception as exc:
            logger.warning("Failed to fetch history for %s %s: %s", metric, labels, exc)

        if not self.forecaster.has_model(metric, labels):
            if len(history) < self.forecaster.lookback_points():
                logger.info("Metric %s %s warming up (history=%d)", metric, labels, len(history))
                return {'status': 'warming_up', 'metric': metric, 'labels': labels}
            try:
                self.forecaster.train(metric, labels, history)
            except ValueError as exc:
                logger.info("Not enough data to train %s %s: %s", metric, labels, exc)
                return {'status': 'warming_up', 'metric': metric, 'labels': labels}

        forecast = self.forecaster.predict(metric, labels, history)
        self._update_gauges(forecast)
        logger.debug("Forecast ready for %s %s -> %.3f", metric, labels, forecast['point_forecast'])
        return forecast

    def prometheus_payload(self) -> bytes:
        return generate_latest()

    def _fetch_history(self, metric: str, labels: Dict[str, str]) -> pd.DataFrame:
        return self.prometheus.get_metric_history(
            metric,
            labels,
            lookback_minutes=self.forecaster.lookback_minutes,
            step_seconds=self.forecaster.cadence_seconds,
            horizon_minutes=self.forecaster.horizon_minutes,
        )

    def _update_gauges(self, forecast: Dict[str, object], stale: bool = False) -> None:
        metric = forecast.get('metric')
        point = forecast.get('point_forecast')
        if metric is None or point is None:
            logger.debug("Skipping gauge update; metric=%s point=%s", metric, point)
            return

        labels_raw = forecast.get('labels', {})
        labels = {key: str(value) for key, value in labels_raw.items() if key != '__name__'}
        hostname = labels.get('hostname', 'unknown')
        component_name = labels.get('component_name') or labels.get('object_name') or 'unknown'
        model_version = str(forecast.get('model_version', 'unknown'))
        horizon = str(self.forecaster.horizon_minutes)
        timestamp_seconds = self._resolve_sample_timestamp(forecast)

        value = float(point)

        try:
            if metric in {'seqrec_passed_packets', 'seqrec_discarded_packets', 'replicate_packets_passed'}:
                metric_type = 'replicate' if metric.startswith('replicate_') else 'seqrec'
                valid = 'passed' if 'passed' in metric else 'discarded'
                self._collector.set_value(
                    'forecasted_seqrec_packets',
                    {
                        'hostname': hostname,
                        'component_name': component_name,
                        'type': metric_type,
                        'model_version': model_version,
                        'horizon_minutes': horizon,
                        'valid': valid,
                    },
                    value,
                    timestamp_seconds,
                )
            elif metric == 'replicate_octets_passed':
                self._collector.set_value(
                    'forecasted_replicate_octets_passed',
                    {
                        'hostname': hostname,
                        'component_name': component_name,
                        'type': 'replicate',
                        'model_version': model_version,
                        'horizon_minutes': horizon,
                        'valid': 'passed',
                    },
                    value,
                    timestamp_seconds,
                )
            elif metric == 'seqrec_recovery_seq_num':
                self._collector.set_value(
                    'forecasted_seq_num',
                    {
                        'hostname': hostname,
                        'component_name': component_name,
                        'type': 'seqrec',
                        'model_version': model_version,
                        'horizon_minutes': horizon,
                    },
                    value,
                    timestamp_seconds,
                )
            elif metric.startswith('interface_'):
                direction = 'recv' if '_recv_' in metric else 'send'
                interface_name = labels.get('interface_name') or labels.get('component_name') or 'unknown'
                metric_name = 'forecasted_interface_octets' if 'octets' in metric else 'forecasted_interface_packets'
                self._collector.set_value(
                    metric_name,
                    {
                        'hostname': hostname,
                        'interface_name': interface_name,
                        'direction': direction,
                        'model_version': model_version,
                        'horizon_minutes': horizon,
                    },
                    value,
                    timestamp_seconds,
                )
            elif metric.startswith('parser_'):
                parser_name = labels.get('parser_name') or labels.get('component_name') or metric.split('_', 1)[-1]
                stream_name = labels.get('stream_name') or labels.get('object_name') or 'unknown'
                metric_name = 'forecasted_parser_octets' if 'octets' in metric else 'forecasted_parser_packets'
                self._collector.set_value(
                    metric_name,
                    {
                        'hostname': hostname,
                        'parser_name': parser_name,
                        'stream_name': stream_name,
                        'model_version': model_version,
                        'horizon_minutes': horizon,
                    },
                    value,
                    timestamp_seconds,
                )
            else:
                logger.debug("No forecast gauge configured for metric %s", metric)
        except Exception as exc:
            logger.warning("Failed to update forecast metric for %s %s: %s", metric, labels, exc)

    def _resolve_sample_timestamp(self, forecast: Dict[str, object]) -> float:
        timestamp_ms: Optional[int] = None
        for key in ('timestamp', 'timestamp_ms', 'prediction_timestamp', 'prediction_timestamp_ms'):
            candidate = forecast.get(key)
            timestamp_ms = self._coerce_timestamp_ms(candidate)
            if timestamp_ms is not None:
                break
        if timestamp_ms is None:
            return time.time()
        return timestamp_ms / 1000.0

    @staticmethod
    def _coerce_timestamp_ms(raw: object) -> Optional[int]:
        if raw is None:
            return None
        if isinstance(raw, pd.Timestamp):
            ts = raw
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')
            else:
                ts = ts.tz_convert('UTC')
            return int(ts.value // 1_000_000)

        numeric: Optional[float] = None
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return None
            try:
                numeric = float(text)
            except ValueError:
                try:
                    ts = pd.Timestamp(text)
                except Exception:
                    return None
                if ts.tzinfo is None:
                    ts = ts.tz_localize('UTC')
                else:
                    ts = ts.tz_convert('UTC')
                return int(ts.value // 1_000_000)
        else:
            try:
                numeric = float(raw)
            except (TypeError, ValueError):
                return None

        if numeric is None or not math.isfinite(numeric):
            return None

        abs_numeric = abs(numeric)
        if abs_numeric >= 1e18:
            # treat as nanoseconds
            return int(numeric / 1_000_000)
        if abs_numeric >= 1e15:
            # treat as microseconds
            return int(numeric / 1_000)
        if abs_numeric >= 1e12:
            # treat as milliseconds
            return int(numeric)
        # default to seconds
        return int(numeric * 1000)
