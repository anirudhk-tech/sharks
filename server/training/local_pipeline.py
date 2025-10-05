#!/usr/bin/env python3
"""
Local SST LSTM Pipeline (No Azure ML Required)
Runs the complete pipeline locally for development and testing.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import Dict, Any

# Load environment variables
try:
    from server.setup.load_env import load_env_file
    load_env_file()
except ImportError:
    print("Warning: Could not load environment file")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules (without Azure ML)
from data_preprocessing import SSTDataProcessor
from lstm_training import SSTLSTMModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('local_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LocalSSTPipeline:
    """Local SST LSTM pipeline without Azure ML."""
    
    def __init__(self):
        """Initialize the local pipeline."""
        self.data_processor = None
        self.model = None
        
    def preprocess_data(self, 
                       input_file: str = "sst_2025_global.csv",
                       output_dir: str = "data/preprocessed",
                       sample_size: int = 100000) -> bool:
        """
        Preprocess the SST data locally.
        
        Args:
            input_file: Path to input CSV file
            output_dir: Directory for preprocessed data
            sample_size: Number of rows to sample for faster processing
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting local data preprocessing...")
            
            # Initialize data processor
            self.data_processor = SSTDataProcessor(
                max_memory_mb=2000,  # Lower memory for local processing
                chunk_size=50000
            )
            
            # Check if data exists
            if not os.path.exists(input_file):
                logger.error(f"Data file not found: {input_file}")
                return False
            
            # Load and sample data
            logger.info(f"Loading data from {input_file}...")
            import pandas as pd
            df = pd.read_csv(input_file, nrows=sample_size)
            logger.info(f"Loaded {len(df)} rows")
            
            # Create features
            logger.info("Creating time series features...")
            df_with_features = self.data_processor.create_time_series_features(df)
            
            # Prepare LSTM data
            logger.info("Preparing LSTM data...")
            lstm_data = self.data_processor.prepare_lstm_data(
                df_with_features,
                sequence_length=24
            )
            
            # Save preprocessed data
            os.makedirs(output_dir, exist_ok=True)
            self.data_processor.save_preprocessed_data(lstm_data, output_dir)
            
            logger.info("Data preprocessing completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            return False
    
    def train_model(self, 
                   preprocessed_data_path: str = "data/preprocessed",
                   model_params: Dict[str, Any] = None) -> bool:
        """
        Train LSTM model locally.
        
        Args:
            preprocessed_data_path: Path to preprocessed data
            model_params: Model parameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting local LSTM training...")
            
            # Check if preprocessed data exists
            if not os.path.exists(os.path.join(preprocessed_data_path, 'train_X.npy')):
                logger.error("Preprocessed data not found. Run preprocessing first.")
                return False
            
            # Load preprocessed data
            import numpy as np
            logger.info("Loading preprocessed data...")
            X_train = np.load(os.path.join(preprocessed_data_path, 'train_X.npy'))
            y_train = np.load(os.path.join(preprocessed_data_path, 'train_y.npy'))
            X_val = np.load(os.path.join(preprocessed_data_path, 'val_X.npy'))
            y_val = np.load(os.path.join(preprocessed_data_path, 'val_y.npy'))
            X_test = np.load(os.path.join(preprocessed_data_path, 'test_X.npy'))
            y_test = np.load(os.path.join(preprocessed_data_path, 'test_y.npy'))
            
            logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            
            # Default model parameters
            if model_params is None:
                model_params = {
                    'sequence_length': X_train.shape[1],
                    'n_features': X_train.shape[2],
                    'lstm_units': 64,  # Smaller for local training
                    'dropout_rate': 0.2,
                    'learning_rate': 0.001
                }
            
            # Initialize model
            logger.info("Building LSTM model...")
            self.model = SSTLSTMModel(**model_params)
            self.model.build_model()
            
            # Train model
            logger.info("Training model...")
            history = self.model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=20,  # Shorter training for demo
                batch_size=32,
                verbose=1
            )
            
            # Evaluate model
            logger.info("Evaluating model...")
            train_metrics = self.model.evaluate(X_train, y_train)
            val_metrics = self.model.evaluate(X_val, y_val)
            test_metrics = self.model.evaluate(X_test, y_test)
            
            # Log results
            logger.info("=== Model Performance ===")
            logger.info(f"Training RMSE: {train_metrics['rmse']:.4f}")
            logger.info(f"Validation RMSE: {val_metrics['rmse']:.4f}")
            logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
            logger.info(f"Training MAE: {train_metrics['mae']:.4f}")
            logger.info(f"Validation MAE: {val_metrics['mae']:.4f}")
            logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
            
            # Save model
            os.makedirs("models", exist_ok=True)
            model_path = "models/local_lstm_model.h5"
            self.model.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save scalers
            import joblib
            scalers_path = "models/local_scalers.pkl"
            joblib.dump(self.data_processor.scalers, scalers_path)
            logger.info(f"Scalers saved to {scalers_path}")
            
            logger.info("Local training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def run_inference(self, 
                     model_path: str = "models/local_lstm_model.h5",
                     scalers_path: str = "models/local_scalers.pkl",
                     preprocessed_data_path: str = "data/preprocessed") -> bool:
        """
        Run inference on the trained model.
        
        Args:
            model_path: Path to trained model
            scalers_path: Path to scalers
            preprocessed_data_path: Path to preprocessed data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Running inference...")
            
            # Check if model exists
            if not os.path.exists(model_path):
                logger.error(f"Model not found: {model_path}")
                return False
            
            # Load model
            model = SSTLSTMModel()
            model.load_model(model_path)
            
            # Load test data
            import numpy as np
            X_test = np.load(os.path.join(preprocessed_data_path, 'test_X.npy'))
            y_test = np.load(os.path.join(preprocessed_data_path, 'test_y.npy'))
            
            # Make predictions
            logger.info("Making predictions...")
            predictions = model.predict(X_test[:10])  # Predict on first 10 samples
            
            # Show results
            logger.info("Sample predictions:")
            for i in range(5):
                logger.info(f"  Actual: {y_test[i]:.4f}, Predicted: {predictions[i][0]:.4f}")
            
            # Calculate metrics
            test_metrics = model.evaluate(X_test, y_test)
            logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
            logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
            
            logger.info("Inference completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return False
    
    def run_full_pipeline(self, 
                         input_file: str = "sst_2025_global.csv",
                         sample_size: int = 100000) -> bool:
        """
        Run the complete local pipeline.
        
        Args:
            input_file: Path to input CSV file
            sample_size: Number of rows to sample
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting full local SST LSTM pipeline...")
        start_time = datetime.now()
        
        try:
            # Step 1: Preprocess data
            if not self.preprocess_data(input_file, sample_size=sample_size):
                logger.error("Data preprocessing failed. Aborting pipeline.")
                return False
            
            # Step 2: Train model
            if not self.train_model():
                logger.error("Model training failed.")
                return False
            
            # Step 3: Run inference
            if not self.run_inference():
                logger.error("Inference failed.")
                return False
            
            # Pipeline completed successfully
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"Pipeline completed successfully in {duration}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False


def main():
    """Main function for local pipeline."""
    parser = argparse.ArgumentParser(description='Local SST LSTM Pipeline')
    parser.add_argument('--input_file', type=str, default='sst_2025_global.csv',
                       help='Input CSV file path')
    parser.add_argument('--sample_size', type=int, default=100000,
                       help='Number of rows to sample for faster processing')
    parser.add_argument('--preprocess_only', action='store_true',
                       help='Only run data preprocessing')
    parser.add_argument('--train_only', action='store_true',
                       help='Only run model training (requires preprocessed data)')
    parser.add_argument('--inference_only', action='store_true',
                       help='Only run inference (requires trained model)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = LocalSSTPipeline()
    
    # Run pipeline based on arguments
    if args.preprocess_only:
        print("Preprocessing data...")
        success = pipeline.preprocess_data(args.input_file, sample_size=args.sample_size)
    elif args.train_only:
        print("Training model...")
        success = pipeline.train_model()
    elif args.inference_only:
        print("Running inference...")
        success = pipeline.run_inference()
    else:
        print("Running full pipeline...")
        success = pipeline.run_full_pipeline(args.input_file, args.sample_size)
    
    if success:
        logger.info("Local pipeline completed successfully!")
        print("\n🎉 Pipeline completed successfully!")
        print("📁 Check the following files:")
        print("   - models/local_lstm_model.h5 (trained model)")
        print("   - models/local_scalers.pkl (data scalers)")
        print("   - data/preprocessed/ (preprocessed data)")
        print("   - local_pipeline.log (detailed logs)")
    else:
        logger.error("Pipeline failed!")
        print("\n❌ Pipeline failed! Check local_pipeline.log for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
