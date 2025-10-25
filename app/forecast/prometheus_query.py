"""Thin Prometheus API client focused on forecast data retrieval."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class PrometheusClient:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._discovery_cache: Dict[str, object] = {}

    def _request(self, path: str, params: Dict[str, str]) -> Dict[str, object]:
        url = f"{self.base_url}{path}"
        logger.debug("Calling Prometheus %s with params %s", url, params)
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if payload.get('status') != 'success':
            raise RuntimeError(f"Prometheus API error: {payload}")
        return payload

    def discover_metrics(self, match_prefix: Optional[str] = None, cache_ttl: int = 300) -> List[str]:
        now = datetime.now(timezone.utc).timestamp()
        cache_entry = self._discovery_cache.get('metrics')
        if cache_entry and cache_entry['expires'] > now:
            metrics = cache_entry['value']
        else:
            payload = self._request('/api/v1/label/__name__/values', {})
            metrics = payload.get('data', [])
            self._discovery_cache['metrics'] = {'value': metrics, 'expires': now + cache_ttl}
        if match_prefix:
            return sorted(metric for metric in metrics if metric.startswith(match_prefix))
        return sorted(metrics)

    def list_series(self, metric: str, lookback_minutes: int = 60) -> List[Dict[str, str]]:
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=lookback_minutes)
        match_expr = f'{metric}'
        params = {
            'match[]': match_expr,
            'start': f"{start.timestamp():.0f}",
            'end': f"{end.timestamp():.0f}",
        }
        payload = self._request('/api/v1/series', params)
        return payload.get('data', [])

    def get_metric_history(
        self,
        metric: str,
        labels: Dict[str, str],
        lookback_minutes: int,
        step_seconds: int,
        horizon_minutes: int,
    ) -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=lookback_minutes + horizon_minutes)
        label_filters = ','.join(
            [
                f"{k}='{v}'"
                for k, v in sorted(labels.items())
                if k != '__name__'
            ]
        )
        query = f"{metric}{{{label_filters}}}" if label_filters else metric
        params = {
            'query': query,
            'start': f"{start.timestamp():.0f}",
            'end': f"{end.timestamp():.0f}",
            'step': f"{step_seconds}s",
        }
        payload = self._request('/api/v1/query_range', params)
        result = payload.get('data', {}).get('result', [])
        if not result:
            logger.info("No data returned for %s labels=%s", metric, labels)
            return pd.DataFrame(columns=['timestamp', 'value'])
        # Prometheus values come as strings; convert carefully.
        values = result[0]['values']
        data = {
            'timestamp': [datetime.fromtimestamp(float(ts), tz=timezone.utc) for ts, _ in values],
            'value': [float(val) for _, val in values],
        }
        df = pd.DataFrame(data)
        preview = df.head(3).to_dict(orient='records')
        logger.info(
            "Fetched %d samples for %s labels=%s preview=%s",
            len(values),
            metric,
            labels,
            preview,
        )
        return df

    def is_healthy(self) -> bool:
        try:
            url = f"{self.base_url}/-/healthy"
            response = requests.get(url, timeout=self.timeout)
            logger.debug("Prometheus health status %s", response.status_code)
            return response.status_code == 200
        except Exception as exc:  # pragma: no cover - health failure path
            logger.debug("Prometheus health check failed: %s", exc)
            return False
