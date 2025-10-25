"""Simple cooperative scheduler managing forecast inference and retraining."""

from __future__ import annotations

import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import schedule

from .forecasting_service import MetricsForecastService
from .metrics_filter import MetricsFilter
from .prometheus_query import PrometheusClient

logger = logging.getLogger(__name__)


class ForecastScheduler:
    def __init__(
        self,
        service: MetricsForecastService,
        prometheus: PrometheusClient,
        metrics_filter: MetricsFilter,
        max_concurrency: int,
        jitter_seconds: int = 5,
        discovery_interval_seconds: int = 300,
    ) -> None:
        self.service = service
        self.prometheus = prometheus
        self.metrics_filter = metrics_filter
        self.max_concurrency = max(1, max_concurrency)
        self.jitter_seconds = jitter_seconds
        self.discovery_interval_seconds = discovery_interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_discovery: datetime = datetime.min.replace(tzinfo=timezone.utc)
        self._known_targets: List[Tuple[str, Dict[str, str]]] = []

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.schedule_jobs()
        self._thread = threading.Thread(target=self._run_loop, name='forecast-scheduler', daemon=True)
        self._thread.start()
        logger.info("Forecast scheduler started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def schedule_jobs(self) -> None:
        cadence = max(1, self.service.forecaster.cadence_seconds)
        schedule.clear('forecast')
        schedule.every(cadence).seconds.do(self.run_inference_cycle).tag('forecast')
        schedule.every().day.at('03:00').do(self.run_retrain_cycle).tag('forecast')

    def run_inference_cycle(self) -> None:
        logger.debug("Running inference cycle")
        targets = self._discover_targets()
        if not targets:
            logger.debug("No eligible targets discovered")
            return
        logger.info("Scheduling %d forecast jobs", len(targets))
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = {executor.submit(self._run_single_forecast, metric, labels): (metric, labels) for metric, labels in targets}
            for future in as_completed(futures):
                metric, labels = futures[future]
                try:
                    future.result()
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning("Forecast job failed for %s %s: %s", metric, labels, exc)

    def run_retrain_cycle(self) -> None:
        logger.info("Running scheduled retrain cycle")
        targets = self._discover_targets(force=True)
        logger.info("Retraining %d targets", len(targets))
        for metric, labels in targets:
            try:
                metadata = self.service.trigger_retrain(metric, labels)
                logger.info("Retrained %s %s -> %s", metric, labels, metadata.get('version'))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Retrain skipped for %s %s: %s", metric, labels, exc)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            schedule.run_pending()
            time.sleep(1)

    def _run_single_forecast(self, metric: str, labels: Dict[str, str]) -> None:
        time.sleep(random.uniform(0, self.jitter_seconds))
        result = self.service.get_forecast(metric, labels)
        if result.get('forecast_stale'):
            logger.debug("Returned stale forecast for %s %s", metric, labels)
        else:
            logger.debug("Forecast job finished for %s %s status=%s", metric, labels, result.get('status', 'ok'))

    def _discover_targets(self, force: bool = False) -> List[Tuple[str, Dict[str, str]]]:
        now = datetime.now(timezone.utc)
        if not force and now - self._last_discovery < timedelta(seconds=self.discovery_interval_seconds):
            return self._known_targets
        self._last_discovery = now
        metrics = self.prometheus.discover_metrics()
        eligible_metrics = self.metrics_filter.filter_metrics(metrics)
        logger.debug("Discovery found %d metrics (eligible=%d)", len(metrics), len(eligible_metrics))
        targets: List[Tuple[str, Dict[str, str]]] = []
        for metric in eligible_metrics:
            series = self.prometheus.list_series(metric, lookback_minutes=60)
            filtered = self.metrics_filter.filter_series(metric, series)
            for labels in filtered:
                targets.append((metric, labels))
        logger.info("Discovered %d metric/label targets", len(targets))
        self._known_targets = targets
        return targets