"""Prometheus collector for forecast metrics with explicit timestamp support."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from prometheus_client.core import GaugeMetricFamily, REGISTRY


@dataclass(frozen=True)
class MetricSpec:
    documentation: str
    labels: Tuple[str, ...]


@dataclass(frozen=True)
class ForecastSample:
    labels: Tuple[str, ...]
    value: float
    timestamp: Optional[float]


class ForecastCollector:
    def __init__(self, metric_specs: Dict[str, MetricSpec]) -> None:
        self._metric_specs = metric_specs
        self._lock = threading.RLock()
        self._values: Dict[str, Dict[Tuple[str, ...], ForecastSample]] = {
            name: {} for name in metric_specs
        }

    def set_value(
        self,
        metric_name: str,
        labels: Dict[str, str],
        value: float,
        timestamp: Optional[float],
    ) -> None:
        spec = self._metric_specs[metric_name]
        missing = [label for label in spec.labels if label not in labels]
        if missing:
            raise KeyError(f"Missing labels {missing} for metric {metric_name}")
        ordered_labels = tuple(str(labels[label]) for label in spec.labels)
        sample = ForecastSample(
            labels=ordered_labels,
            value=float(value),
            timestamp=float(timestamp) if timestamp is not None else None,
        )
        with self._lock:
            self._values[metric_name][ordered_labels] = sample

    def collect(self) -> Iterable[GaugeMetricFamily]:
        with self._lock:
            snapshot = {
                name: list(samples.values()) for name, samples in self._values.items()
            }
        for metric_name, samples in snapshot.items():
            spec = self._metric_specs[metric_name]
            family = GaugeMetricFamily(
                metric_name,
                spec.documentation,
                labels=list(spec.labels),
            )
            for sample in samples:
                family.add_metric(list(sample.labels), sample.value, timestamp=sample.timestamp)
            yield family


OBJECT_LABELS = (
    'hostname',
    'component_name',
    'type',
    'model_version',
    'horizon_minutes',
)
OBJECT_LABELS_WITH_VALID = OBJECT_LABELS + ('valid',)
INTERFACE_LABELS = (
    'hostname',
    'interface_name',
    'direction',
    'model_version',
    'horizon_minutes',
)
PARSER_LABELS = (
    'hostname',
    'parser_name',
    'stream_name',
    'model_version',
    'horizon_minutes',
)

FORECAST_METRIC_SPECS = {
    'forecasted_seqrec_packets': MetricSpec(
        documentation='Number of packets predicted by the forecast model',
        labels=OBJECT_LABELS_WITH_VALID,
    ),
    'forecasted_replicate_octets_passed': MetricSpec(
        documentation='Number of replicate octets passed predicted by the forecast model',
        labels=OBJECT_LABELS_WITH_VALID,
    ),
    'forecasted_seq_num': MetricSpec(
        documentation='Predicted sequence number by the forecast model',
        labels=OBJECT_LABELS,
    ),
    'forecasted_interface_octets': MetricSpec(
        documentation='Number of interface octets predicted by the forecast model',
        labels=INTERFACE_LABELS,
    ),
    'forecasted_interface_packets': MetricSpec(
        documentation='Number of interface packets predicted by the forecast model',
        labels=INTERFACE_LABELS,
    ),
    'forecasted_parser_packets': MetricSpec(
        documentation='Number of parser packets predicted by the forecast model',
        labels=PARSER_LABELS,
    ),
    'forecasted_parser_octets': MetricSpec(
        documentation='Number of parser octets predicted by the forecast model',
        labels=PARSER_LABELS,
    ),
}

_FORECAST_COLLECTOR = ForecastCollector(FORECAST_METRIC_SPECS)
try:
    REGISTRY.register(_FORECAST_COLLECTOR)
except ValueError:
    # Collector already registered (likely due to module reload)
    pass


def get_forecast_collector() -> ForecastCollector:
    return _FORECAST_COLLECTOR
