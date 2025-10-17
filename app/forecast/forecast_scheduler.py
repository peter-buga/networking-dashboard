"""
Scheduler for periodic training and forecasting.
Handles automatic model retraining and prediction generation.
Automatically discovers metrics from Prometheus.
"""

import time
import logging
import threading
import schedule
from typing import Dict, List, Optional
from datetime import datetime
import concurrent.futures

from forecasting_service import MetricsForecastService


logger = logging.getLogger(__name__)


class ForecastScheduler:
    """Manages periodic training and forecasting tasks."""
    
    def __init__(
        self,
        service: MetricsForecastService,
        retrain_interval: str = "weekly",
        forecast_interval: int = 300,  # seconds
        max_workers: int = 4  # Parallel training threads
    ):
        """
        Initialize scheduler.
        
        Args:
            service: MetricsForecastService instance
            retrain_interval: Training frequency ("daily", "weekly", "monthly")
            forecast_interval: Seconds between forecasts (default: 5 minutes)
            max_workers: Number of parallel training threads (default: 4)
        """
        self.service = service
        self.retrain_interval = retrain_interval
        self.forecast_interval = forecast_interval
        self.max_workers = max_workers
        
        self.is_running = False
        self.scheduler_thread = None
        self.lock = threading.Lock()
        
        # Metrics for tracking
        self.last_training_time: Optional[datetime] = None
        self.last_forecast_time: Optional[datetime] = None
        self.training_count = 0
        self.forecast_count = 0
        self.total_training_errors = 0
        self.total_forecast_errors = 0
    
    def schedule_training(self):
        """Schedule periodic model training."""
        # Parse interval
        if self.retrain_interval.lower() == "daily":
            schedule.every().day.at("02:00").do(self._train_all_models)
            logger.info("Scheduled daily model training at 02:00")
        
        elif self.retrain_interval.lower() == "weekly":
            schedule.every().monday.at("02:00").do(self._train_all_models)
            logger.info("Scheduled weekly model training on Monday at 02:00")
        
        elif self.retrain_interval.lower() == "monthly":
            # Schedule for 1st of month (approximated weekly check)
            schedule.every().week.do(self._check_monthly_training)
            logger.info("Scheduled monthly model training check")
        
        else:
            logger.warning(f"Unknown retrain interval: {self.retrain_interval}")
    
    def schedule_forecasting(self):
        """Schedule periodic forecasting."""
        schedule.every(self.forecast_interval).seconds.do(self._forecast_all_metrics)
        logger.info(f"Scheduled forecasting every {self.forecast_interval} seconds")
    
    def _train_all_models(self):
        """
        Train all discovered metrics from Prometheus in parallel.
        
        Only metrics defined in MetricsExporter will be trained.
        """
        logger.info("Starting scheduled model training")
        
        try:
            # Discover available metrics from Prometheus
            available_metrics = self.service.discover_metrics()
            
            if not available_metrics:
                logger.warning("No metrics found in Prometheus or none match MetricsExporter definitions")
                return
            
            total_metrics = len(available_metrics)
            logger.info(f"Discovered {total_metrics} defined metrics from Prometheus (using {self.max_workers} workers)")
            
            # Train metrics in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for idx, metric_name in enumerate(available_metrics, 1):
                    future = executor.submit(
                        self._train_single_metric,
                        metric_name,
                        idx,
                        total_metrics
                    )
                    futures[future] = (metric_name, idx)
                
                # Wait for completion and track results
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    metric_name, idx = futures[future]
                    completed += 1
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"[{completed}/{total_metrics}] Failed to train {metric_name}: {e}")
                        self.total_training_errors += 1
            
            # Save trained models
            self.service.save_models()
            self.last_training_time = datetime.now()
            self.training_count += 1
            
            logger.info(f"Training completed at {self.last_training_time} - Trained {total_metrics} metrics")
        
        except Exception as e:
            logger.error(f"Training cycle failed: {e}")
    
    def _train_single_metric(self, metric_name: str, idx: int, total: int):
        """Train a single metric (called in parallel)."""
        try:
            logger.info(f"[{idx}/{total}] Training model for {metric_name}")
            results = self.service.train_forecaster(
                metric_name=metric_name,
                history_hours=24,
                epochs=50
            )
            
            # Count successes/failures
            for label_key, result in results.items():
                if result.get('status') == 'success':
                    logger.info(f"  ✓ [{idx}/{total}] {label_key}: {result['metrics']['test_r2']:.4f} R²")
                else:
                    logger.error(f"  ✗ [{idx}/{total}] {label_key}: {result.get('error')}")
                    self.total_training_errors += 1
        
        except Exception as e:
            logger.error(f"[{idx}/{total}] Failed to train {metric_name}: {e}")
            self.total_training_errors += 1
    
    
    def _forecast_all_metrics(self):
        """Generate forecasts for all discovered metrics."""
        try:
            # Get all trained models
            trained_model_keys = list(self.service.models.keys())
            
            if not trained_model_keys:
                logger.debug("No trained models available for forecasting")
                return
            
            forecast_count = 0
            for model_key in trained_model_keys:
                try:
                    # Split model_key into metric_name and label_key
                    # Format: "metric_name:label_key"
                    parts = model_key.split(':', 1)
                    if len(parts) != 2:
                        logger.warning(f"Invalid model key format: {model_key}")
                        continue
                    
                    metric_name, label_key = parts
                    
                    forecast = self.service.forecast(
                        metric_name=metric_name,
                        label_key=label_key,
                        steps_ahead=6
                    )
                    
                    if forecast is not None:
                        forecast_count += 1
                        logger.debug(f"Generated forecast for {model_key}")
                
                except Exception as e:
                    logger.error(f"Failed to forecast {model_key}: {e}")
                    self.total_forecast_errors += 1
            
            if forecast_count > 0:
                logger.debug(f"Generated {forecast_count} forecasts in this cycle")
            
            self.last_forecast_time = datetime.now()
            self.forecast_count += 1
        
        except Exception as e:
            logger.error(f"Forecast cycle failed: {e}")
    
    def _check_monthly_training(self):
        """Check if it's time for monthly training."""
        # Simple approximation: train if last training was >28 days ago
        if self.last_training_time is None:
            self._train_all_models()
        else:
            days_since = (datetime.now() - self.last_training_time).days
            if days_since >= 28:
                self._train_all_models()
    
    def start(self):
        """Start the scheduler in a background thread."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        
        # Schedule tasks
        self.schedule_training()
        self.schedule_forecasting()
        
        # Initial training
        logger.info("Performing initial model training...")
        self._train_all_models()
        
        # Start background scheduler thread
        self.scheduler_thread = threading.Thread(
            target=self._run_scheduler,
            daemon=True,
            name="ForecastScheduler"
        )
        self.scheduler_thread.start()
        logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        self.is_running = False
        schedule.clear()
        logger.info("Scheduler stopped")
    
    def _run_scheduler(self):
        """Run the scheduler loop."""
        logger.info("Scheduler loop started")
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(5)
    
    def get_status(self) -> Dict:
        """Get scheduler status."""
        return {
            "is_running": self.is_running,
            "retrain_interval": self.retrain_interval,
            "forecast_interval": self.forecast_interval,
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "last_forecast_time": self.last_forecast_time.isoformat() if self.last_forecast_time else None,
            "training_count": self.training_count,
            "forecast_count": self.forecast_count,
            "training_errors": self.total_training_errors,
            "forecast_errors": self.total_forecast_errors,
            "models_loaded": len(self.service.models),
        }
