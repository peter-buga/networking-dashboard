"""
LSTM forecaster for time series prediction using Keras.
"""
import numpy as np
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import keras
from keras import layers, models, callbacks


class LSTMForecaster:
    """LSTM-based time series forecaster using Keras."""
    
    def __init__(
        self,
        sequence_length: int = 24,
        forecast_horizon: int = 6,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM forecaster.
        
        Args:
            sequence_length: Number of past time steps to use for prediction
            forecast_horizon: Number of steps to forecast into the future
            lstm_units: Number of units in LSTM layers
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
        self.training_history = None
    
    def _create_sequences(
        self,
        data: np.ndarray,
        seq_length: int,
        forecast_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: 1D array of values
            seq_length: Length of input sequences
            forecast_len: Length of forecast target
        
        Returns:
            Tuple of (X, y) where X is input sequences and y is targets
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length - forecast_len + 1):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length:i + seq_length + forecast_len])
        
        return np.array(X), np.array(y)
    
    def _prepare_data(
        self,
        values: np.ndarray,
        test_split: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare and normalize data for training.
        
        Args:
            values: 1D array of time series values
            test_split: Fraction of data to use for testing
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test, original_values)
        """
        # Remove NaN values
        clean_values = values[~np.isnan(values)]
        
        if len(clean_values) < self.sequence_length + self.forecast_horizon:
            raise ValueError(
                f"Not enough data. Need at least {self.sequence_length + self.forecast_horizon} "
                f"points, got {len(clean_values)}"
            )
        
        # Normalize data
        scaled_values = self.scaler.fit_transform(clean_values.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self._create_sequences(
            scaled_values,
            self.sequence_length,
            self.forecast_horizon
        )
        
        # Split into train and test
        split_idx = int(len(X) * (1 - test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Reshape for LSTM [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        return X_train, y_train, X_test, y_test, clean_values
    
    def build_model(self):
        """Build the LSTM neural network model."""
        model = models.Sequential([
            layers.Input(shape=(self.sequence_length, 1)),
            layers.LSTM(
                self.lstm_units,
                activation='relu',
                return_sequences=True
            ),
            layers.Dropout(self.dropout_rate),
            layers.LSTM(self.lstm_units, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.forecast_horizon)
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def fit(
        self,
        values: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping: bool = True,
        verbose: int = 1
    ) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            values: Time series data to train on
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            early_stopping: Whether to use early stopping
            verbose: Verbosity level
        
        Returns:
            Dictionary with training metrics
        """
        if self.model is None:
            self.build_model()
        
        X_train, y_train, X_test, y_test, clean_values = self._prepare_data(
            values,
            test_split=0.2
        )
        
        # Prepare callbacks
        cb_list = []
        if early_stopping:
            cb_list.append(
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=verbose
                )
            )
        
        cb_list.append(
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=verbose
            )
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=cb_list,
            verbose=verbose
        )
        
        self.training_history = history.history
        self.is_fitted = True
        
        # Evaluate on test set
        y_pred_train = self.model.predict(X_train, verbose=0)
        y_pred_test = self.model.predict(X_test, verbose=0)
        
        # Inverse transform predictions for metrics
        y_pred_train_original = self.scaler.inverse_transform(y_pred_train)
        y_pred_test_original = self.scaler.inverse_transform(y_pred_test)
        y_train_original = self.scaler.inverse_transform(y_train)
        y_test_original = self.scaler.inverse_transform(y_test)
        
        metrics = {
            'train_mse': float(np.mean((y_pred_train_original - y_train_original) ** 2)),
            'test_mse': float(np.mean((y_pred_test_original - y_test_original) ** 2)),
            'train_mae': float(np.mean(np.abs(y_pred_train_original - y_train_original))),
            'test_mae': float(np.mean(np.abs(y_pred_test_original - y_test_original))),
            'train_r2': float(r2_score(y_train_original.flatten(), y_pred_train_original.flatten())),
            'test_r2': float(r2_score(y_test_original.flatten(), y_pred_test_original.flatten()))
        }
        
        return metrics
    
    def predict(self, values: np.ndarray, steps_ahead: Optional[int] = None) -> np.ndarray:
        """
        Make predictions for future time steps.
        
        Args:
            values: Historical time series data
            steps_ahead: Number of steps to predict ahead (uses forecast_horizon if None)
        
        Returns:
            Array of predictions
        """
        if self.model is None or not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if steps_ahead is None:
            steps_ahead = self.forecast_horizon
        
        # Clean and normalize the data
        clean_values = values[~np.isnan(values)]
        scaled_values = self.scaler.transform(clean_values.reshape(-1, 1)).flatten()
        
        # Use last sequence_length points for initial prediction
        current_seq = scaled_values[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        predictions = []
        
        for _ in range(steps_ahead):
            # Predict next value
            next_pred = self.model.predict(current_seq, verbose=0)[0]
            predictions.append(next_pred)
            
            # Update sequence for next iteration (use mean of forecast_horizon predictions)
            next_val = next_pred[0] if self.forecast_horizon > 1 else next_pred[0]
            current_seq = np.append(current_seq[0, 1:, 0], next_val)
            current_seq = current_seq.reshape(1, self.sequence_length, 1)
        
        # Convert predictions to array and inverse transform
        predictions = np.array(predictions)
        predictions_original = self.scaler.inverse_transform(predictions)
        
        return predictions_original.flatten()
    
    def save_model(self, filepath: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load model from file."""
        self.model = keras.models.load_model(filepath)
        self.is_fitted = True
