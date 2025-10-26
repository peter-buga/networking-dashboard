"""Orchestration layer exposing forecast capabilities to the rest of the app."""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
from prometheus_client import Gauge, generate_latest, start_http_server

from .lstm_forecaster import LSTMForecaster
from .metrics_filter import MetricsFilter
from .prometheus_query import PrometheusClient

logger = logging.getLogger(__name__)


class MetricsForecastService:
    def __init__(
        self,
        forecaster: LSTMForecaster,
        prometheus: PrometheusClient,
        metrics_filter: MetricsFilter,
    ) -> None:
        self.forecaster = forecaster
        self.prometheus = prometheus
        self.metrics_filter = metrics_filter

        object_labels = [
            'hostname',  # hostname of the reporting device
            'component_name',  # name of the component reporting seqrec metrics
            'type',  # seqrec | replicate
            'model_version',
            'horizon_minutes',
        ]

        self.forecasted_packets = Gauge(
            'forecasted_seqrec_packets',
            'Number of packets predicted by the forecast model',
            object_labels + ['valid'],  # valid = discarded|passed
        )

        self.forecasted_octets = Gauge(
            'forecasted_replicate_octets_passed',
            'Number of replicate octets passed predicted by the forecast model',
            object_labels + ['valid'],  # valid = discarded|passed
        )

        self.forecasted_seq_num = Gauge(
            'forecasted_seq_num',
            'Predicted sequence number by the forecast model',
            object_labels,
        )

        interface_labels = ['hostname', 
                            'interface_name', 
                            'direction',  # recv | send
                            'model_version', 
                            'horizon_minutes']

        self.forecasted_interface_octets = Gauge(
            'forecasted_interface_octets',
            'Number of interface octets predicted by the forecast model',
            interface_labels,
        )

        self.forecasted_interface_packets = Gauge(
            'forecasted_interface_packets',
            'Number of interface packets predicted by the forecast model',
            interface_labels,
        )

        parser_labels = ['hostname', 
                         'parser_name',
                         'stream_name',
                         'model_version', 
                         'horizon_minutes']

        self.forecasted_parser_packets = Gauge(
            'forecasted_parser_packets',
            'Number of parser packets predicted by the forecast model',
            parser_labels,
        )

        self.forecasted_parser_octets = Gauge(
            'forecasted_parser_octets',
            'Number of parser octets predicted by the forecast model',
            parser_labels,
        )

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

        try:
            if metric in {'seqrec_passed_packets', 'seqrec_discarded_packets', 'replicate_packets_passed'}:
                metric_type = 'replicate' if metric.startswith('replicate_') else 'seqrec'
                valid = 'passed' if 'passed' in metric else 'discarded'
                self.forecasted_packets.labels(
                    hostname=hostname,
                    component_name=component_name,
                    type=metric_type,
                    model_version=model_version,
                    horizon_minutes=horizon,
                    valid=valid,
                ).set(float(point))
            elif metric == 'replicate_octets_passed':
                self.forecasted_octets.labels(
                    hostname=hostname,
                    component_name=component_name,
                    type='replicate',
                    model_version=model_version,
                    horizon_minutes=horizon,
                    valid='passed',
                ).set(float(point))
            elif metric == 'seqrec_recovery_seq_num':
                self.forecasted_seq_num.labels(
                    hostname=hostname,
                    component_name=component_name,
                    type='seqrec',
                    model_version=model_version,
                    horizon_minutes=horizon,
                ).set(float(point))
            elif metric.startswith('interface_'):
                direction = 'recv' if '_recv_' in metric else 'send'
                interface_name = labels.get('interface_name') or labels.get('component_name') or 'unknown'
                gauge = self.forecasted_interface_octets if 'octets' in metric else self.forecasted_interface_packets
                gauge.labels(
                    hostname=hostname,
                    interface_name=interface_name,
                    direction=direction,
                    model_version=model_version,
                    horizon_minutes=horizon,
                ).set(float(point))
            elif metric.startswith('parser_'):
                parser_name = labels.get('parser_name') or labels.get('component_name') or metric.split('_', 1)[-1]
                stream_name = labels.get('stream_name') or labels.get('object_name') or 'unknown'
                gauge = self.forecasted_parser_octets if 'octets' in metric else self.forecasted_parser_packets
                gauge.labels(
                    hostname=hostname,
                    parser_name=parser_name,
                    stream_name=stream_name,
                    model_version=model_version,
                    horizon_minutes=horizon,
                ).set(float(point))
            else:
                logger.debug("No forecast gauge configured for metric %s", metric)
        except Exception as exc:
            logger.warning("Failed to update forecast gauge for %s %s: %s", metric, labels, exc)
