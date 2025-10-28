"""ARIMA-based forecaster implementation replacing the previous LSTM model."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    results: ARIMAResults
    metadata: Dict[str, object]
    order: Tuple[int, int, int]


class ARIMAForecaster:
    def __init__(self) -> None:
        self.lookback_minutes = int(os.getenv('FORECAST_LOOKBACK_MINUTES', 180))
        self.horizon_minutes = int(os.getenv('FORECAST_HORIZON_MINUTES', 180))
        self.retrain_days = int(os.getenv('FORECAST_RETRAIN_DAYS', 7))
        self.cadence_seconds = int(os.getenv('FORECAST_CADENCE_SECONDS', 15))
        self.max_parallel_jobs = int(os.getenv('FORECAST_MAX_CONCURRENCY', 10))
        clip_sigma = os.getenv('FORECAST_CLIP_Z')
        self.clip_sigma = float(clip_sigma) if clip_sigma else None
        baseline_strategy = os.getenv('FORECAST_BASELINE_STRATEGY', 'last').lower()
        if baseline_strategy not in {'last', 'mean', 'median'}:
            logger.warning("Unsupported baseline strategy %s, defaulting to 'last'", baseline_strategy)
            baseline_strategy = 'last'
        self.baseline_strategy = baseline_strategy
        default_weight = os.getenv('FORECAST_BASELINE_WEIGHT')
        self._baseline_weight_override = default_weight is not None
        self.default_baseline_weight = float(default_weight) if default_weight else 0.0
        env_model_dir = os.getenv('FORECAST_MODEL_DIR')
        default_dir = Path(__file__).resolve().parents[2] / 'docker_setup' / 'forecaster_models'
        if env_model_dir:
            self.model_dir = self._prepare_model_dir(Path(env_model_dir).expanduser(), allow_fallback=False)
        else:
            self.model_dir = self._prepare_model_dir(default_dir, allow_fallback=True)
        self._cache: Dict[Tuple[str, str], ModelBundle] = {}
        self.arima_order = self._parse_order(os.getenv('FORECAST_ARIMA_ORDER'))
        self.arima_maxiter = int(os.getenv('FORECAST_ARIMA_MAXITER', 200))
        self.arima_fallback_orders = self._parse_order_list(os.getenv('FORECAST_ARIMA_FALLBACKS'))
        logger.info(
            "ARIMAForecaster initialised (order=%s lookback=%smin horizon=%smin cadence=%ssec models=%s)",
            self.arima_order,
            self.lookback_minutes,
            self.horizon_minutes,
            self.cadence_seconds,
            self.model_dir,
        )

    def train(self, metric: str, labels: Dict[str, str], history: pd.DataFrame) -> Dict[str, object]:
        label_hash = self._label_hash(labels)
        logger.info("Training ARIMA model for %s/%s (samples=%d)", metric, label_hash, len(history))
        preprocessed = self._preprocess(history)
        lookback_steps = self._lookback_steps()
        horizon_steps = self._horizon_steps()

        if preprocessed.empty:
            raise ValueError('No usable history points were provided')

        series = preprocessed['value'].astype(np.float32)
        min_required = lookback_steps + horizon_steps + 1
        if len(series) < min_required:
            raise ValueError(
                f'Not enough data to train (have={len(series)}, required>={min_required})'
            )

        # Apply stationarity check and differencing transformation
        series, diff_order, original_values = self._make_stationary(series)

        # After differencing, verify we still have enough data
        if len(series) < min_required:
            raise ValueError(
                f'Not enough data after differencing (have={len(series)}, required>={min_required})'
            )

        train_series = series.iloc[:-horizon_steps]
        eval_series = series.iloc[-horizon_steps:]

        eval_results, order = self._fit_model_with_fallback(train_series)

        eval_forecast = eval_results.get_forecast(steps=horizon_steps)
        predicted_eval = eval_forecast.predicted_mean.to_numpy(dtype=np.float32)
        baseline_window = series.iloc[-lookback_steps:].to_numpy(dtype=np.float32)
        baseline_eval = np.repeat(self._baseline_forecast(baseline_window), horizon_steps).astype(np.float32)

        candidate_weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        baseline_weight = self.default_baseline_weight
        if not self._baseline_weight_override and len(predicted_eval) == len(eval_series):
            best_rmse = float('inf')
            best_weight = baseline_weight
            for weight in candidate_weights:
                combined = (1 - weight) * predicted_eval + weight * baseline_eval
                rmse_candidate = float(np.sqrt(np.mean(np.square(eval_series.to_numpy(dtype=np.float32) - combined))))
                if np.isfinite(rmse_candidate) and rmse_candidate < best_rmse:
                    best_rmse = rmse_candidate
                    best_weight = float(weight)
            baseline_weight = best_weight

        combined_eval = (1 - baseline_weight) * predicted_eval + baseline_weight * baseline_eval
        residuals = eval_series.to_numpy(dtype=np.float32) - combined_eval
        rmse = float(np.sqrt(np.mean(np.square(residuals)))) if residuals.size else float('nan')
        residual_std = float(np.std(residuals)) if residuals.size else float('nan')
        residual_mean = float(np.mean(residuals)) if residuals.size else 0.0
        full_results, final_order = self._fit_model_with_fallback(series, preferred_order=order)
        if final_order != order:
            logger.info(
                "Recomputing evaluation metrics for %s/%s with fallback ARIMA order %s",
                metric,
                label_hash,
                final_order,
            )
            eval_results, _ = self._fit_model_with_fallback(train_series, preferred_order=final_order)
            eval_forecast = eval_results.get_forecast(steps=horizon_steps)
            predicted_eval = eval_forecast.predicted_mean.to_numpy(dtype=np.float32)
            combined_eval = (1 - baseline_weight) * predicted_eval + baseline_weight * baseline_eval
            residuals = eval_series.to_numpy(dtype=np.float32) - combined_eval
            rmse = float(np.sqrt(np.mean(np.square(residuals)))) if residuals.size else float('nan')
            residual_std = float(np.std(residuals)) if residuals.size else float('nan')
            residual_mean = float(np.mean(residuals)) if residuals.size else 0.0
            order = final_order

        metadata = {
            'metric': metric,
            'labels': labels,
            'label_hash': label_hash,
            'version': self._new_version_identifier(),
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'lookback_minutes': self.lookback_minutes,
            'horizon_minutes': self.horizon_minutes,
            'samples': len(series),
            'rmse': rmse,
            'residual_std': residual_std,
            'residual_mean': residual_mean,
            'baseline_weight': float(baseline_weight),
            'baseline_strategy': self.baseline_strategy,
            'arima_order': final_order,
            'diff_order': diff_order,
        }

        promoted = self._save_bundle(
            metric,
            metadata['label_hash'],
            ModelBundle(results=full_results, metadata=metadata, order=final_order),
        )
        logger.info(
            "Training completed for %s/%s version=%s rmse=%.4f baseline_w=%.2f promoted=%s",
            metric,
            label_hash,
            metadata['version'],
            metadata['rmse'],
            metadata['baseline_weight'],
            promoted,
        )
        metadata['promoted'] = promoted
        return metadata

    def predict(self, metric: str, labels: Dict[str, str], history: pd.DataFrame) -> Dict[str, object]:
        bundle = self._ensure_bundle(metric, labels)
        if bundle is None:
            raise FileNotFoundError('Model not trained for given metric/labels')

        preprocessed = self._preprocess(history)
        if preprocessed.empty:
            raise ValueError('No history available for prediction')

        series = preprocessed['value'].astype(np.float32)
        lookback_steps = self._lookback_steps()
        horizon_steps = self._horizon_steps()

        if len(series) < lookback_steps:
            raise ValueError('Not enough history points for inference')

        # Get differencing order from model metadata
        diff_order = int(bundle.metadata.get('diff_order', 0))

        # Keep original series reference for inverse differencing and baseline calculation
        original_series = series.copy()

        if diff_order > 0:
            # Apply differencing to input series for prediction
            if diff_order == 1:
                series = series.diff().dropna()
            elif diff_order == 2:
                series = original_series.diff().diff().dropna()
            logger.debug("Applied d=%d differencing to input series for prediction (samples: %d -> %d)",
                        diff_order, len(original_series), len(series))

        results = bundle.results.apply(series, refit=False)
        forecast = results.get_forecast(steps=horizon_steps)
        predicted = forecast.predicted_mean.to_numpy(dtype=np.float32)

        # Inverse transform predictions back to original scale if differencing was applied
        # Use fresh values from current prediction input, not stale training metadata
        if diff_order > 0:
            fresh_original_values = original_series.dropna().tolist()
            predicted = self._inverse_difference(predicted, diff_order, fresh_original_values).astype(np.float32)
            logger.debug("Applied inverse differencing to predictions (using fresh values from input)")

        point_forecast = float(predicted[-1])
        prediction_timestamp_ms = self._compute_prediction_timestamp(preprocessed.index, forecast, horizon_steps)

        conf_int = forecast.conf_int()
        if isinstance(conf_int, pd.DataFrame):
            lower, upper = map(float, conf_int.iloc[-1])
        else:
            lower = float(conf_int[-1][0])
            upper = float(conf_int[-1][1])

        # Inverse transform confidence intervals using fresh values
        if diff_order > 0:
            fresh_original_values = original_series.dropna().tolist()
            lower_array = np.array([lower])
            upper_array = np.array([upper])
            lower = float(self._inverse_difference(lower_array, diff_order, fresh_original_values)[0])
            upper = float(self._inverse_difference(upper_array, diff_order, fresh_original_values)[0])

        baseline_weight = float(bundle.metadata.get('baseline_weight', self.default_baseline_weight))
        if baseline_weight > 0:
            # Calculate baseline from original scale series, not differenced series
            raw_window = original_series.iloc[-lookback_steps:].to_numpy(dtype=np.float32)
            strategy = bundle.metadata.get('baseline_strategy', self.baseline_strategy)
            if strategy not in {'last', 'mean', 'median'}:
                strategy = self.baseline_strategy
            baseline_value = self._baseline_forecast(raw_window, strategy=strategy)
            point_forecast = (1 - baseline_weight) * point_forecast + baseline_weight * baseline_value

        residual_mean = float(bundle.metadata.get('residual_mean', 0.0))
        residual_std = float(bundle.metadata.get('residual_std', 0.0))
        if np.isnan(residual_std):
            residual_std = 0.0

        center = point_forecast + residual_mean
        margin = 1.96 * residual_std if residual_std > 0 else 0.0
        lower_bound = min(center - margin, lower)
        upper_bound = max(center + margin, upper)
        confidence_interval = [
            float(max(lower_bound, 0.0)),
            float(upper_bound),
        ]

        logger.debug(
            "Predicted value for %s/%s -> %.3f (ci=%s)",
            metric,
            bundle.metadata['label_hash'],
            point_forecast,
            confidence_interval,
        )

        return {
            'metric': metric,
            'labels': labels,
            'point_forecast': point_forecast,
            'confidence_interval': confidence_interval,
            'model_version': bundle.metadata['version'],
            'prediction_timestamp_ms': prediction_timestamp_ms,
        }

    def load_model(self, metric: str, labels: Dict[str, str]) -> Optional[Dict[str, object]]:
        bundle = self._ensure_bundle(metric, labels)
        if bundle is None:
            return None
        return bundle.metadata

    def has_model(self, metric: str, labels: Dict[str, str]) -> bool:
        return self._ensure_bundle(metric, labels) is not None

    def list_models(self, metric: Optional[str] = None) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        root = self.model_dir if metric is None else self.model_dir / metric
        if not root.exists():
            return results
        for metric_dir in (root.iterdir() if metric is None else [root]):
            if not metric_dir.is_dir():
                continue
            for label_dir in metric_dir.iterdir():
                if not label_dir.is_dir():
                    continue
                latest = label_dir / 'latest.json'
                if latest.exists():
                    try:
                        results.append(json.loads(latest.read_text()))
                    except Exception as exc:  # pragma: no cover - IO edge
                        logger.warning("Failed to read metadata for %s: %s", label_dir, exc)
        return results

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.debug("Forecaster cache cleared")

    def lookback_points(self) -> int:
        return self._lookback_steps()

    def horizon_points(self) -> int:
        return self._horizon_steps()

    # --- Internal helpers -----------------------------------------------
    def _check_stationarity(self, series: pd.Series, alpha: float = 0.05) -> Tuple[bool, Dict[str, object]]:
        """
        Check if a time series is stationary using both ADF and KPSS tests.

        A series is considered stationary if:
        - ADF test rejects H0 (non-stationary) at alpha significance level
        - KPSS test fails to reject H0 (stationary) at alpha significance level

        Returns:
            (is_stationary, test_stats_dict)
        """
        if series.empty or len(series) < 10:
            logger.warning("Series too short for stationarity testing (len=%d)", len(series))
            return False, {'error': 'Series too short'}

        # Check for constant series (no variance)
        if series.std() == 0:
            logger.warning("Series has zero variance (constant values)")
            return False, {'error': 'Zero variance series', 'mean': float(series.mean())}

        test_stats = {}
        try:
            # ADF Test: H0 = non-stationary, reject if p-value < alpha
            adf_result = adfuller(series, autolag='AIC')
            adf_pvalue = float(adf_result[1])
            test_stats['adf_pvalue'] = adf_pvalue
            test_stats['adf_statistic'] = float(adf_result[0])
            test_stats['adf_critical_5pct'] = float(adf_result[4]['5%'])
            adf_stationary = adf_pvalue < alpha
        except Exception as exc:
            logger.warning("ADF test failed: %s", exc)
            adf_stationary = False
            test_stats['adf_error'] = str(exc)

        try:
            # KPSS Test: H0 = stationary, fail to reject if p-value >= alpha
            kpss_result = kpss(series, regression='c', nlags='auto')
            kpss_pvalue = float(kpss_result[1])
            test_stats['kpss_pvalue'] = kpss_pvalue
            test_stats['kpss_statistic'] = float(kpss_result[0])
            test_stats['kpss_critical_5pct'] = float(kpss_result[3]['5%'])
            kpss_stationary = kpss_pvalue >= alpha
        except Exception as exc:
            logger.warning("KPSS test failed: %s", exc)
            kpss_stationary = False
            test_stats['kpss_error'] = str(exc)

        # Both tests must agree
        is_stationary = adf_stationary and kpss_stationary
        test_stats['stationary'] = is_stationary

        logger.info(
            "Stationarity test: ADF=%s (p=%.4f), KPSS=%s (p=%.4f) -> stationary=%s",
            adf_stationary,
            test_stats.get('adf_pvalue', -1),
            kpss_stationary,
            test_stats.get('kpss_pvalue', -1),
            is_stationary,
        )

        return is_stationary, test_stats

    def _make_stationary(self, series: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int, List[float]]:
        """
        Transform a time series to be stationary through differencing if needed.

        If the series is already stationary, returns it unchanged.
        Otherwise, applies first-order or second-order differencing as needed.

        Returns:
            (stationary_series, differencing_order, original_values_for_inverse)
        """
        original_series = series.copy()
        diff_order = 0

        is_stationary, test_stats = self._check_stationarity(series)

        if not is_stationary and max_diff >= 1:
            logger.info("Applying first-order differencing to make series stationary")
            series = series.diff().dropna()
            diff_order = 1

            is_stationary, test_stats = self._check_stationarity(series)

        if not is_stationary and max_diff >= 2:
            logger.info("Applying second-order differencing to make series stationary")
            series = original_series.diff().diff().dropna()
            diff_order = 2

            is_stationary, test_stats = self._check_stationarity(series)

        if not is_stationary:
            logger.warning(
                "Series could not be made stationary with d=%d differencing, proceeding anyway",
                max_diff,
            )

        logger.info("Stationarity transformation complete: diff_order=%d, samples=%d", diff_order, len(series))

        # Store original values for inverse transformation (needed for predictions)
        original_values = original_series.dropna().tolist()

        return series, diff_order, original_values

    def _inverse_difference(
        self, differenced_values: np.ndarray, diff_order: int, original_values: List[float]
    ) -> np.ndarray:
        """
        Reverse the differencing transformation to get predictions back to original scale.

        Args:
            differenced_values: The differenced predictions from ARIMA
            diff_order: The order of differencing applied (1 or 2)
            original_values: Original (non-differenced) values needed for inverse transform

        Returns:
            Array of values in original scale
        """
        if diff_order == 0 or len(original_values) == 0:
            return differenced_values

        values = differenced_values.copy()

        if diff_order == 1:
            # For d=1: original[t] = diff[t] + original[t-1]
            last_original = original_values[-1]
            reconstructed = np.empty_like(values)
            for i, diff_val in enumerate(values):
                last_original = last_original + diff_val
                reconstructed[i] = last_original
            return reconstructed

        elif diff_order == 2:
            # For d=2: apply inverse twice
            # First, reconstruct d=1 level using the last original value and its first difference
            # last_diff = original_values[-1] - original_values[-2] (if available)
            # Then reconstruct d=0 from d=1

            if len(original_values) >= 2:
                last_diff = original_values[-1] - original_values[-2]
            else:
                last_diff = 0.0

            # First inverse: reconstruct first-difference level
            level_1_reconstructed = np.empty_like(values)
            for i, second_diff_val in enumerate(values):
                last_diff = last_diff + second_diff_val
                level_1_reconstructed[i] = last_diff

            # Second inverse: reconstruct original level
            last_original = original_values[-1]
            reconstructed = np.empty_like(values)
            for i, first_diff_val in enumerate(level_1_reconstructed):
                last_original = last_original + first_diff_val
                reconstructed[i] = last_original

            return reconstructed

        return values

    def _fit_single(self, series: pd.Series, order: Tuple[int, int, int]) -> ARIMAResults:
        model = ARIMA(
            series,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('ignore', ConvergenceWarning)
            results = model.fit(method_kwargs={'maxiter': self.arima_maxiter})
        converged = True
        retvals = getattr(results, 'mle_retvals', None)
        if isinstance(retvals, dict):
            converged = bool(retvals.get('converged', True))
        if any(issubclass(w.category, ConvergenceWarning) for w in captured):
            converged = False
        if not converged:
            raise RuntimeError(f"ARIMA order {order} failed to converge (maxiter={self.arima_maxiter})")
        return results

    def _preprocess(self, history: pd.DataFrame) -> pd.DataFrame:
        if history is None or history.empty:
            return pd.DataFrame(columns=['timestamp', 'value'])

        df = history.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df = df.drop_duplicates(subset=['timestamp'])
            df = df.sort_values('timestamp')
            df = df.set_index('timestamp')
        else:
            df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
            df = df[~df.index.isna()]
            df = df[~df.index.duplicated(keep='last')]
            df = df.sort_index()

        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['value'])

        if df.empty:
            return df

        target_freq = f'{self.cadence_seconds}s'
        try:
            full_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=target_freq)
            if len(full_index) and (len(full_index) <= len(df) * 5):
                df = df.reindex(full_index)
        except Exception as exc:
            logger.debug("Failed to reindex history to cadence: %s", exc)

        if df['value'].isna().any():
            df['value'] = df['value'].interpolate(method='time')
            df['value'] = df['value'].ffill().bfill()

        if df['value'].isna().any():
            df = df.dropna(subset=['value'])

        if self.clip_sigma and not df.empty:
            mean = df['value'].mean()
            std = df['value'].std(ddof=0)
            if std > 0:
                limit = self.clip_sigma * std
                df['value'] = df['value'].clip(mean - limit, mean + limit)

        return df

    def _baseline_forecast(self, window: np.ndarray, strategy: Optional[str] = None) -> float:
        if window.size == 0:
            return 0.0
        mode = (strategy or self.baseline_strategy)
        if mode == 'mean':
            return float(np.mean(window))
        if mode == 'median':
            return float(np.median(window))
        return float(window[-1])

    def _candidate_orders(self, preferred: Optional[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        orders: List[Tuple[int, int, int]] = []

        def _add(candidate: Optional[Tuple[int, int, int]]) -> None:
            if candidate and candidate not in orders:
                orders.append(candidate)

        _add(preferred)
        _add(self.arima_order)
        for fallback_order in self.arima_fallback_orders:
            _add(fallback_order)
        for default in [(1, 1, 1), (1, 1, 0), (0, 1, 1), (0, 1, 0)]:
            _add(default)
        return orders

    def _fit_model_with_fallback(
        self,
        series: pd.Series,
        preferred_order: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[ARIMAResults, Tuple[int, int, int]]:
        candidates = self._candidate_orders(preferred_order)
        last_exc: Optional[Exception] = None
        for idx, order in enumerate(candidates):
            try:
                results = self._fit_single(series, order)
                if idx > 0:
                    logger.info("Using fallback ARIMA order %s (series=%d)", order, len(series))
                return results, order
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
                logger.warning("ARIMA fit failed for order %s: %s", order, exc)
                last_exc = exc
        raise RuntimeError(f"Failed to fit ARIMA model with candidate orders {candidates}") from last_exc

    def _parse_order(self, raw: Optional[str]) -> Tuple[int, int, int]:
        if not raw:
            # Changed default from (2,1,2) to (2,0,2) because differencing is now handled explicitly
            return (2, 0, 2)
        try:
            parts = [int(part.strip()) for part in raw.split(',')]
            if len(parts) != 3:
                raise ValueError
            p, d, q = parts
            return (max(0, p), max(0, d), max(0, q))
        except ValueError:
            logger.warning("Invalid FORECAST_ARIMA_ORDER %s, defaulting to (2, 0, 2)", raw)
            return (2, 0, 2)

    def _parse_order_list(self, raw: Optional[str]) -> List[Tuple[int, int, int]]:
        if not raw:
            return []
        orders: List[Tuple[int, int, int]] = []
        for chunk in raw.split(';'):
            chunk = chunk.strip()
            if not chunk:
                continue
            orders.append(self._parse_order(chunk))
        return orders

    def _ensure_bundle(self, metric: str, labels: Dict[str, str]) -> Optional[ModelBundle]:
        key = (metric, self._label_hash(labels))
        if key in self._cache:
            return self._cache[key]
        bundle = self._load_bundle(metric, labels)
        if bundle:
            self._cache[key] = bundle
        return bundle

    def _load_bundle(self, metric: str, labels: Dict[str, str]) -> Optional[ModelBundle]:
        label_hash = self._label_hash(labels)
        label_dir = self.model_dir / metric / label_hash
        latest_file = label_dir / 'latest.json'
        if not latest_file.exists():
            return None
        try:
            metadata = json.loads(latest_file.read_text())
        except json.JSONDecodeError:
            logger.warning("Corrupt metadata for %s", label_dir)
            return None
        version_dir = label_dir / metadata['version']
        if not version_dir.exists():
            logger.warning("Missing artifact directory %s", version_dir)
            return None
        model_path = version_dir / 'model.pkl'
        if not model_path.exists():
            logger.warning("Incomplete artifacts for %s", version_dir)
            return None
        results = ARIMAResults.load(str(model_path))
        raw_order = metadata.get('arima_order', self.arima_order)
        # Ensure raw_order is a tuple/list of three integers
        if isinstance(raw_order, (list, tuple)) and len(raw_order) == 3:
            order = tuple(int(part) for part in raw_order)
        elif isinstance(raw_order, str):
            order = self._parse_order(raw_order)
        else:
            order = self.arima_order
        if len(order) != 3:
            order = self.arima_order
        return ModelBundle(results=results, metadata=metadata, order=order)  # type: ignore[arg-type]

    def _save_bundle(self, metric: str, label_hash: str, bundle: ModelBundle) -> bool:
        label_dir = self.model_dir / metric / label_hash
        version_dir = label_dir / bundle.metadata['version']
        version_dir.mkdir(parents=True, exist_ok=True)

        bundle.results.save(str(version_dir / 'model.pkl'))
        (version_dir / 'metadata.json').write_text(json.dumps(bundle.metadata, indent=2))

        latest_meta = self._load_latest_metadata(label_dir)
        promoted = False
        if latest_meta is None or bundle.metadata['rmse'] <= latest_meta.get('rmse', float('inf')):
            promoted = True
            latest_file = label_dir / 'latest.json'
            latest_file.write_text(json.dumps(bundle.metadata, indent=2))
            cache_key = (bundle.metadata['metric'], label_hash)
            self._cache[cache_key] = bundle
        else:
            logger.info(
                "New model for %s/%s not promoted; rmse %.4f worse than %.4f",
                metric,
                label_hash,
                bundle.metadata['rmse'],
                latest_meta.get('rmse'),
            )
        return promoted

    def _load_latest_metadata(self, label_dir: Path) -> Optional[Dict[str, object]]:
        latest_file = label_dir / 'latest.json'
        if latest_file.exists():
            try:
                return json.loads(latest_file.read_text())
            except json.JSONDecodeError:
                logger.warning("Failed to parse %s", latest_file)
        return None

    def _label_hash(self, labels: Dict[str, str]) -> str:
        filtered = {str(k): str(v) for k, v in labels.items() if k != '__name__'}
        if not filtered:
            return 'default'
        canonical = json.dumps(filtered, sort_keys=True, separators=(',', ':'))
        return hashlib.sha1(canonical.encode('utf-8')).hexdigest()[:16]

    def _lookback_steps(self) -> int:
        return max(1, int((self.lookback_minutes * 60) / self.cadence_seconds))

    def _horizon_steps(self) -> int:
        return max(1, int((self.horizon_minutes * 60) / self.cadence_seconds))

    def _compute_prediction_timestamp(
        self,
        history_index: pd.Index,
        forecast_obj,
        horizon_steps: int,
    ) -> int:
        """
        Determine the timestamp for the current point forecast.

        Prefer the timestamp produced by the forecasting model. If unavailable,
        fall back to extrapolating from the latest history point. Returns
        milliseconds since the Unix epoch.
        """
        timestamp: Optional[pd.Timestamp] = None
        forecast_series = getattr(forecast_obj, 'predicted_mean', None)
        forecast_index = getattr(forecast_series, 'index', None)
        if forecast_index is not None:
            try:
                raw_label = forecast_index[-1]
            except Exception:
                raw_label = None
            if raw_label is not None:
                try:
                    ts_candidate = pd.Timestamp(raw_label)
                    if ts_candidate.tzinfo is None:
                        timestamp = ts_candidate.tz_localize(timezone.utc)
                    else:
                        timestamp = ts_candidate.tz_convert(timezone.utc)
                except (TypeError, ValueError):
                    timestamp = None
        if timestamp is None:
            fallback_ts: Optional[pd.Timestamp] = None
            if len(history_index):
                try:
                    last_history = pd.Timestamp(history_index[-1])
                    if last_history.tzinfo is None:
                        last_history = last_history.tz_localize(timezone.utc)
                    else:
                        last_history = last_history.tz_convert(timezone.utc)
                    fallback_ts = last_history + pd.Timedelta(seconds=self.cadence_seconds * horizon_steps)
                except Exception:
                    fallback_ts = None
            if fallback_ts is None:
                fallback_ts = pd.Timestamp.now(tz=timezone.utc)
            timestamp = fallback_ts
        return int(timestamp.value // 1_000_000)

    def _new_version_identifier(self) -> str:
        now = datetime.now(timezone.utc)
        return now.strftime('v%Y%m%d-%H%M')

    def _prepare_model_dir(self, candidate: Path, allow_fallback: bool) -> Path:
        resolved = self._ensure_writable_dir(candidate)
        if resolved:
            return resolved
        if allow_fallback:
            fallback = Path.home() / '.cache' / 'networking-dashboard' / 'forecaster_models'
            fallback_resolved = self._ensure_writable_dir(fallback)
            if fallback_resolved:
                logger.info("Using fallback model directory at %s", fallback_resolved)
                return fallback_resolved
        raise PermissionError(
            f"Model directory {candidate} is not writable. Set FORECAST_MODEL_DIR to a writable path."
        )

    def _ensure_writable_dir(self, path: Path) -> Optional[Path]:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.warning("Cannot create model directory %s due to insufficient permissions", path)
            return None
        if os.access(path, os.W_OK | os.X_OK):
            return path
        logger.warning("Model directory %s exists but is not writable", path)
        return None
