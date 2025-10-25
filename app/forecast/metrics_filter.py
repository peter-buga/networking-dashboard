"""Utility helpers to filter Prometheus metrics based on managed configuration."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from time import time
from typing import Dict, Iterable, List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

logger = logging.getLogger(__name__)


@dataclass
class MetricRule:
    """Simple selector combining a metric name glob and optional label filters."""

    pattern: str
    labels: Dict[str, str] = field(default_factory=dict)

    def matches(self, metric: str, series_labels: Optional[Dict[str, str]] = None) -> bool:
        if not fnmatch(metric, self.pattern):
            return False
        if not self.labels:
            return True
        if not series_labels:
            return False
        for key, expected in self.labels.items():
            actual = series_labels.get(key)
            if actual is None:
                return False
            if not fnmatch(actual, expected):
                return False
        return True


class MetricsFilter:
    """Loads metric inclusion rules from disk and filters discovery results on demand."""

    def __init__(self, config_path: Path, reload_interval: float = 5.0) -> None:
        self._config_path = Path(config_path)
        self._reload_interval = reload_interval
        self._lock = threading.RLock()
        self._rules: List[MetricRule] = []
        self._last_reload: float = 0.0
        self._last_mtime: float = 0.0
        self._load_rules(force=True)

    def _load_rules(self, force: bool = False) -> None:
        now = time()
        if not force and now - self._last_reload < self._reload_interval:
            return

        self._last_reload = now
        if not self._config_path.exists():
            logger.debug("Metrics filter config %s missing; allowing no metrics", self._config_path)
            with self._lock:
                self._rules = []
            return

        mtime = self._config_path.stat().st_mtime
        if not force and mtime <= self._last_mtime:
            return

        try:
            raw_text = self._config_path.read_text()
            data = self._parse_config(raw_text)
            rules = [MetricRule(pattern=item['pattern'], labels=item.get('labels', {})) for item in data]
        except Exception as exc:  # pragma: no cover - config edge cases
            logger.warning("Failed to load metrics filter from %s: %s", self._config_path, exc)
            return

        with self._lock:
            self._rules = rules
            self._last_mtime = mtime
        logger.info("Loaded %d metric filter rules from %s", len(rules), self._config_path)

    @staticmethod
    def _parse_config(text: str) -> List[Dict[str, object]]:
        if not text.strip():
            return []
        if yaml is not None:
            parsed = yaml.safe_load(text)
        else:
            parsed = json.loads(text)
        if parsed is None:
            return []
        if isinstance(parsed, dict):
            metrics_section = parsed.get('metrics', [])
        else:
            metrics_section = parsed
        if not isinstance(metrics_section, list):
            raise ValueError('metrics_filter config must define a list of rules under "metrics"')
        rules: List[Dict[str, object]] = []
        for entry in metrics_section:
            if not isinstance(entry, dict):
                raise ValueError('each metrics_filter rule must be a mapping')
            pattern = entry.get('pattern') or entry.get('metric')
            if not pattern:
                raise ValueError('metrics_filter rule missing "pattern" field')
            labels = entry.get('labels', {})
            if labels and not isinstance(labels, dict):
                raise ValueError('metrics_filter rule labels must be a mapping')
            rules.append({'pattern': str(pattern), 'labels': {str(k): str(v) for k, v in labels.items()}})
        return rules

    def filter_metrics(self, metrics: Iterable[str]) -> List[str]:
        self._load_rules()
        with self._lock:
            if not self._rules:
                return sorted(set(metrics))
            return sorted({metric for metric in metrics if any(rule.matches(metric) for rule in self._rules)})

    def filter_series(self, metric: str, series: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
        self._load_rules()
        with self._lock:
            if not self._rules:
                return list(series)
            allowed = [rule for rule in self._rules if rule.matches(metric)]
        if not allowed:
            return []
        result: List[Dict[str, str]] = []
        for label_set in series:
            if any(rule.matches(metric, label_set) for rule in allowed):
                result.append(label_set)
        return result

    @staticmethod
    def validate_config(raw_text: str) -> None:
        MetricsFilter._parse_config(raw_text)


def load_default_filter() -> MetricsFilter:
    default_path = Path(__file__).with_name('metrics_filter.yaml')
    return MetricsFilter(default_path)

