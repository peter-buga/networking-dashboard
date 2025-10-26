"""Minimal LSTM-based forecaster implementation following the project plan."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import joblib

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
	model: keras.Model
	scaler: MinMaxScaler
	target_scaler: Optional[MinMaxScaler]
	metadata: Dict[str, object]


class LSTMForecaster:
	def __init__(self) -> None:
		self.lookback_minutes = int(os.getenv('FORECAST_LOOKBACK_MINUTES', 180))
		self.horizon_minutes = int(os.getenv('FORECAST_HORIZON_MINUTES', 180))
		self.retrain_days = int(os.getenv('FORECAST_RETRAIN_DAYS', 7))
		self.cadence_seconds = int(os.getenv('FORECAST_CADENCE_SECONDS', 15))
		self.max_parallel_jobs = int(os.getenv('FORECAST_MAX_CONCURRENCY', 10))
		clip_sigma = os.getenv('FORECAST_CLIP_Z')
		self.clip_sigma = float(clip_sigma) if clip_sigma else None
		env_model_dir = os.getenv('FORECAST_MODEL_DIR')
		default_dir = Path(__file__).resolve().parents[2] / 'docker_setup' / 'forecaster_models'
		if env_model_dir:
			# Honour explicit configuration; fail fast if not writable.
			self.model_dir = self._prepare_model_dir(Path(env_model_dir).expanduser(), allow_fallback=False)
		else:
			self.model_dir = self._prepare_model_dir(default_dir, allow_fallback=True)
		self._cache: Dict[Tuple[str, str], ModelBundle] = {}
		logger.info(
			"LSTMForecaster initialised (lookback=%sm horizon=%sm cadence=%ss models=%s)",
			self.lookback_minutes,
			self.horizon_minutes,
			self.cadence_seconds,
			self.model_dir,
		)

	# --- Public helpers -------------------------------------------------
	def train(self, metric: str, labels: Dict[str, str], history: pd.DataFrame) -> Dict[str, object]:
		label_hash = self._label_hash(labels)
		logger.info("Training started for %s/%s (samples=%d)", metric, label_hash, len(history))
		series = self._preprocess(history)
		lookback_steps = self._lookback_steps()
		if len(series) < lookback_steps + self._horizon_steps():
			raise ValueError('Not enough datapoints to train model')

		values = series.values.astype(np.float32).reshape(-1, 1)
		feature_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
		scaled_values = feature_scaler.fit_transform(values).flatten()

		horizon_steps = self._horizon_steps()
		X_windows, target_indices = self._build_windows(scaled_values, lookback_steps, horizon_steps)
		X_scaled = X_windows.reshape(X_windows.shape[0], lookback_steps, 1).astype(np.float32)

		original_values = values.flatten()
		y_original = original_values[target_indices].astype(np.float32)
		target_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
		y_scaled = target_scaler.fit_transform(y_original.reshape(-1, 1)).astype(np.float32).flatten()

		split_ratio = float(os.getenv('FORECAST_TRAIN_SPLIT', '0.67'))
		split_ratio = min(max(split_ratio, 0.1), 0.9)
		split_idx = max(1, int(len(X_scaled) * split_ratio))
		X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
		y_train_orig, y_val_orig = y_original[:split_idx], y_original[split_idx:]
		y_train_scaled = y_scaled[:split_idx]
		y_val_scaled = y_scaled[split_idx:]

		model = self._build_model(lookback_steps)
		epochs = int(os.getenv('FORECAST_TRAIN_EPOCHS', 5))
		batch_size = int(os.getenv('FORECAST_BATCH_SIZE', 32))
		batch_size = max(1, min(batch_size, len(X_train)))
		verbose = 0 if os.getenv('FORECAST_TRAIN_QUIET', '1') == '1' else 1
		callbacks: List[keras.callbacks.Callback] = []
		patience = int(os.getenv('FORECAST_EARLY_STOP_PATIENCE', '0'))
		if patience > 0 and len(X_val) > 0:
			callbacks.append(
				keras.callbacks.EarlyStopping(
					monitor='val_loss',
					patience=patience,
					restore_best_weights=True,
				)
			)
		validation_data = (X_val, y_val_scaled) if len(X_val) > 0 else None
		model.fit(
			X_train,
			y_train_scaled,
			epochs=epochs,
			batch_size=batch_size,
			verbose=verbose,
			validation_data=validation_data,
			callbacks=callbacks,
		)

		if len(X_val) > 0:
			pred_scaled = model.predict(X_val, verbose=0)
			predictions = target_scaler.inverse_transform(pred_scaled)
			residuals = predictions.flatten() - y_val_orig
		else:
			pred_scaled = model.predict(X_train, verbose=0)
			predictions = target_scaler.inverse_transform(pred_scaled)
			residuals = predictions.flatten() - y_train_orig

		rmse = float(np.sqrt(np.mean(np.square(residuals))))
		residual_std = float(np.std(residuals))

		metadata = {
			'metric': metric,
			'labels': labels,
			'label_hash': label_hash,
			'version': self._new_version_identifier(),
			'trained_at': datetime.now(timezone.utc).isoformat(),
			'lookback_minutes': self.lookback_minutes,
			'horizon_minutes': self.horizon_minutes,
			'train_split': split_ratio,
			'rmse': rmse,
			'residual_std': residual_std,
			'value_scaler': 'minmax',
		}

		promoted = self._save_bundle(
			metric,
			metadata['label_hash'],
			ModelBundle(model, feature_scaler, target_scaler, metadata),
		)
		logger.info(
			"Training completed for %s/%s version=%s rmse=%.4f promoted=%s",
			metric,
			label_hash,
			metadata['version'],
			rmse,
			promoted,
		)
		metadata['promoted'] = promoted
		return metadata

	def predict(self, metric: str, labels: Dict[str, str], history: pd.DataFrame) -> Dict[str, object]:
		bundle = self._ensure_bundle(metric, labels)
		if bundle is None:
			raise FileNotFoundError('Model not trained for given metric/labels')

		series = self._preprocess(history).tail(self._lookback_steps())
		if len(series) < self._lookback_steps():
			raise ValueError('Insufficient history for prediction window')

		window = series.values.astype(np.float32).reshape(-1, 1)
		scaled_matrix = bundle.scaler.transform(window).astype(np.float32)
		scaled = scaled_matrix.reshape(1, self._lookback_steps(), 1)
		scaled_forecast = bundle.model.predict(scaled, verbose=0)
		target_scaler = bundle.target_scaler or bundle.scaler
		point_forecast = float(target_scaler.inverse_transform(scaled_forecast).flatten()[0])
		std = float(bundle.metadata.get('residual_std') or 0.0)
		if std > 0:
			delta = 1.96 * std
			confidence_interval = (point_forecast - delta, point_forecast + delta)
		else:
			confidence_interval = None

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
	def _preprocess(self, history: pd.DataFrame) -> pd.Series:
		if history.empty:
			return pd.Series(dtype=float)
		df = history.copy()
		df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
		df = df.sort_values('timestamp').set_index('timestamp')
		target = df['value'].astype(float)
		resampled = target.resample(f'{self.cadence_seconds}s').mean()
		resampled = resampled.ffill(limit=4).bfill(limit=4)
		if self.clip_sigma is not None and len(resampled) > 0:
			clipped = resampled.clip(resampled.mean() - self.clip_sigma * resampled.std(), resampled.mean() + self.clip_sigma * resampled.std())
		else:
			clipped = resampled
		return clipped.ffill().bfill().astype(np.float32)
	
	def _build_windows(self, values: np.ndarray, lookback_steps: int, horizon_steps: int) -> Tuple[np.ndarray, np.ndarray]:
			flat_values = np.asarray(values, dtype=np.float32).flatten()
			X: List[np.ndarray] = []
			target_indices: List[int] = []
			for end_idx in range(lookback_steps, len(flat_values) - horizon_steps + 1):
				start_idx = end_idx - lookback_steps
				target_idx = end_idx + horizon_steps - 1
				X.append(flat_values[start_idx:end_idx])
				target_indices.append(int(target_idx))
			if not X:
				raise ValueError('Unable to build training windows for provided series')
			return np.array(X, dtype=np.float32), np.array(target_indices, dtype=np.int64)

	def _build_model(self, lookback_steps: int) -> keras.Model:
			model = keras.Sequential([
				layers.Input(shape=(lookback_steps, 1)),
				layers.LSTM(50, return_sequences=True),
				layers.LSTM(50),
				layers.Dense(1),
			])
			model.compile(optimizer='adam', loss='mse', metrics=['mae'])
			return model

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
		model_path = version_dir / 'model.keras'
		scaler_path = version_dir / 'scaler.pkl'
		if not model_path.exists() or not scaler_path.exists():
			logger.warning("Incomplete artifacts for %s", version_dir)
			return None
		model = keras.models.load_model(model_path)
		scaler = self._load_scaler(scaler_path)
		try:
			target_scaler = self._load_scaler(version_dir / 'target_scaler.pkl')
		except FileNotFoundError:
			target_scaler = None
		return ModelBundle(model=model, scaler=scaler, target_scaler=target_scaler, metadata=metadata)

	def _save_bundle(self, metric: str, label_hash: str, bundle: ModelBundle) -> bool:
		label_dir = self.model_dir / metric / label_hash
		version_dir = label_dir / bundle.metadata['version']
		version_dir.mkdir(parents=True, exist_ok=True)

		bundle.model.save(version_dir / 'model.keras', include_optimizer=False)
		self._save_scaler(version_dir / 'scaler.pkl', bundle.scaler)
		if bundle.target_scaler is not None:
			self._save_scaler(version_dir / 'target_scaler.pkl', bundle.target_scaler)
		(version_dir / 'metadata.json').write_text(json.dumps(bundle.metadata, indent=2))

		latest_meta = self._load_latest_metadata(label_dir)
		promoted = False
		if latest_meta is None or bundle.metadata['rmse'] <= latest_meta.get('rmse', float('inf')):
			promoted = True
			latest_file = label_dir / 'latest.json'
			latest_file.write_text(json.dumps(bundle.metadata, indent=2))
			cache_key = (metric, label_hash)
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

	def _save_scaler(self, path: Path, scaler: MinMaxScaler) -> None:
		joblib.dump(scaler, path)

	def _load_scaler(self, path: Path) -> MinMaxScaler:
		return joblib.load(path)

	def _label_hash(self, labels: Dict[str, str]) -> str:
		"""Create a deterministic, filesystem-safe hash for a label set."""
		filtered = {str(k): str(v) for k, v in labels.items() if k != '__name__'}
		if not filtered:
			return 'default'
		canonical = json.dumps(filtered, sort_keys=True, separators=(',', ':'))
		return hashlib.sha1(canonical.encode('utf-8')).hexdigest()[:16]

	def _lookback_steps(self) -> int:
		return max(1, int((self.lookback_minutes * 60) / self.cadence_seconds))

	def _horizon_steps(self) -> int:
		return max(1, int((self.horizon_minutes * 60) / self.cadence_seconds))

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
