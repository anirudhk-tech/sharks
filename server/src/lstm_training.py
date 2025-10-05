"""
Custom LSTM Training Script for Sea Surface Temperature Prediction
Compatible with Azure AutoML and designed for time series forecasting.
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import joblib
from datetime import datetime

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure ML imports (optional)
try:
    from azureml.core import Run, Dataset
    from azureml.core.model import Model
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False
    logger.warning("Azure ML not available. Running in local mode.")

class SSTLSTMModel:
    """LSTM model for Sea Surface Temperature prediction."""
    
    def __init__(self, 
                 sequence_length: int = 24,
                 n_features: int = 20,
                 lstm_units: int = 128,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of input features
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate
            learning_rate: Learning rate
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self) -> Model:
        """Build the LSTM model architecture."""
        logger.info("Building LSTM model...")
        
        # Input layer
        input_layer = Input(shape=(self.sequence_length, self.n_features), name='input')
        
        # LSTM layers
        lstm1 = LSTM(
            self.lstm_units,
            return_sequences=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
            name='lstm1'
        )(input_layer)
        
        lstm2 = LSTM(
            self.lstm_units // 2,
            return_sequences=False,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
            name='lstm2'
        )(lstm1)
        
        # Dense layers
        dense1 = Dense(
            64,
            activation='relu',
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
            name='dense1'
        )(lstm2)
        
        dropout1 = Dropout(self.dropout_rate, name='dropout1')(dense1)
        
        dense2 = Dense(
            32,
            activation='relu',
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
            name='dense2'
        )(dropout1)
        
        dropout2 = Dropout(self.dropout_rate, name='dropout2')(dense2)
        
        # Output layer
        output = Dense(1, activation='linear', name='output')(dropout2)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=output)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logger.info("Model built successfully")
        return self.model
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray = None,
              y_val: np.ndarray = None,
              epochs: int = 100,
              batch_size: int = 32,
              verbose: int = 1) -> Dict:
        """Train the LSTM model."""
        logger.info("Starting model training...")
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Print model summary
        if verbose:
            self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        logger.info("Training completed")
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


class AzureMLTrainer:
    """Azure ML trainer for LSTM model."""
    
    def __init__(self):
        """Initialize Azure ML trainer."""
        if AZURE_ML_AVAILABLE:
            self.run = Run.get_context()
        else:
            self.run = None
        self.model = None
        
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to Azure ML."""
        if self.run is not None:
            for name, value in metrics.items():
                self.run.log(name, value)
                logger.info(f"Logged {name}: {value}")
        else:
            for name, value in metrics.items():
                logger.info(f"Local mode - {name}: {value}")
    
    def log_plots(self, history: Dict):
        """Log training plots to Azure ML."""
        import matplotlib.pyplot as plt
        
        # Plot training history
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE
        axes[0, 1].plot(history['mae'], label='Training MAE')
        if 'val_mae' in history:
            axes[0, 1].plot(history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Model MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MAPE
        axes[1, 0].plot(history['mape'], label='Training MAPE')
        if 'val_mape' in history:
            axes[1, 0].plot(history['val_mape'], label='Validation MAPE')
        axes[1, 0].set_title('Model MAPE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        if 'lr' in history:
            axes[1, 1].plot(history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        if self.run is not None:
            self.run.log_image("training_history", plot=fig)
        else:
            # Save plot locally
            plt.savefig("training_history.png")
            logger.info("Training history plot saved as training_history.png")
        plt.close()
    
    def train_model(self, 
                   data_path: str,
                   model_params: Dict[str, Any] = None) -> SSTLSTMModel:
        """Train LSTM model with Azure ML integration."""
        logger.info("Starting Azure ML training...")
        
        # Load preprocessed data
        logger.info("Loading preprocessed data...")
        X_train = np.load(os.path.join(data_path, 'train_X.npy'))
        y_train = np.load(os.path.join(data_path, 'train_y.npy'))
        X_val = np.load(os.path.join(data_path, 'val_X.npy'))
        y_val = np.load(os.path.join(data_path, 'val_y.npy'))
        X_test = np.load(os.path.join(data_path, 'test_X.npy'))
        y_test = np.load(os.path.join(data_path, 'test_y.npy'))
        
        logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Default model parameters
        if model_params is None:
            model_params = {
                'sequence_length': X_train.shape[1],
                'n_features': X_train.shape[2],
                'lstm_units': 128,
                'dropout_rate': 0.2,
                'learning_rate': 0.001
            }
        
        # Initialize model
        self.model = SSTLSTMModel(**model_params)
        
        # Train model
        history = self.model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=100,
            batch_size=32
        )
        
        # Log training history
        self.log_plots(history)
        
        # Evaluate model
        train_metrics = self.model.evaluate(X_train, y_train)
        val_metrics = self.model.evaluate(X_val, y_val)
        test_metrics = self.model.evaluate(X_test, y_test)
        
        # Log metrics
        self.log_metrics({f'train_{k}': v for k, v in train_metrics.items()})
        self.log_metrics({f'val_{k}': v for k, v in val_metrics.items()})
        self.log_metrics({f'test_{k}': v for k, v in test_metrics.items()})
        
        # Save model
        model_path = 'outputs/sst_lstm_model.h5'
        os.makedirs('outputs', exist_ok=True)
        self.model.save_model(model_path)
        
        # Register model (if Azure ML available)
        if self.run is not None:
            self.run.upload_file('sst_lstm_model.h5', model_path)
            self.run.register_model(
                model_name='sst-lstm-model',
                model_path='sst_lstm_model.h5',
                description='LSTM model for Sea Surface Temperature prediction'
            )
        else:
            logger.info("Model saved locally (Azure ML not available)")
        
        logger.info("Training completed successfully!")
        return self.model


def main():
    """Main function for training."""
    parser = argparse.ArgumentParser(description='Train LSTM model for SST prediction')
    parser.add_argument('--input_data', type=str, required=True, help='Path to preprocessed data')
    parser.add_argument('--output_model', type=str, default='outputs/model.h5', help='Output model path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lstm_units', type=int, default=128, help='Number of LSTM units')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = AzureMLTrainer()
    
    # Model parameters
    model_params = {
        'lstm_units': args.lstm_units,
        'dropout_rate': args.dropout_rate,
        'learning_rate': args.learning_rate
    }
    
    # Train model
    model = trainer.train_model(args.input_data, model_params)
    
    logger.info("Training pipeline completed!")


if __name__ == "__main__":
    main()

