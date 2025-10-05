"""
Main Pipeline Script for Sea Surface Temperature LSTM Prediction
Orchestrates the entire pipeline from data preprocessing to model deployment.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

# Import our modules
from data_preprocessing import SSTDataProcessor
from azure_automl_pipeline import AzureAutoMLPipeline
from lstm_training import AzureMLTrainer
from azure_config import get_config, AzureConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SSTPipeline:
    """Main pipeline for SST LSTM prediction."""
    
    def __init__(self, environment: str = "development"):
        """
        Initialize the SST pipeline.
        
        Args:
            environment: Environment (development/production)
        """
        self.config = get_config(environment)
        self.data_processor = None
        self.azure_pipeline = None
        self.trainer = None
        
        # Validate configuration
        if not self.config.validate_config():
            logger.warning("Azure configuration not fully set. Some features may not work.")
    
    def setup_azure_ml(self) -> bool:
        """Set up Azure ML workspace and compute."""
        try:
            logger.info("Setting up Azure ML workspace...")
            
            self.azure_pipeline = AzureAutoMLPipeline(
                subscription_id=self.config.SUBSCRIPTION_ID,
                resource_group=self.config.RESOURCE_GROUP,
                workspace_name=self.config.WORKSPACE_NAME,
                workspace_region=self.config.WORKSPACE_REGION
            )
            
            # Setup workspace
            workspace = self.azure_pipeline.setup_workspace()
            logger.info(f"Azure ML workspace ready: {workspace.name}")
            
            # Setup compute target
            compute_target = self.azure_pipeline.setup_compute_target(
                compute_name=self.config.COMPUTE_NAME,
                vm_size=self.config.VM_SIZE,
                min_nodes=self.config.MIN_NODES,
                max_nodes=self.config.MAX_NODES
            )
            logger.info(f"Compute target ready: {compute_target.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Azure ML: {e}")
            return False
    
    def preprocess_data(self, 
                       input_file: str = "sst_2025_global.csv",
                       output_dir: str = "data/preprocessed") -> bool:
        """
        Preprocess the SST data.
        
        Args:
            input_file: Path to input CSV file
            output_dir: Directory for preprocessed data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting data preprocessing...")
            
            # Initialize data processor
            self.data_processor = SSTDataProcessor(
                max_memory_mb=self.config.DATA_CONFIG["max_memory_mb"],
                chunk_size=self.config.DATA_CONFIG["chunk_size"]
            )
            
            # Load data
            logger.info(f"Loading data from {input_file}...")
            df = self.data_processor._load_csv_data(input_file)
            
            # Sample data for faster processing (remove for full dataset)
            if self.config.DATA_CONFIG["sample_size"] < len(df):
                logger.info(f"Sampling {self.config.DATA_CONFIG['sample_size']} rows for faster processing...")
                df = df.sample(n=self.config.DATA_CONFIG["sample_size"], random_state=42)
            
            # Create features
            logger.info("Creating time series features...")
            df_with_features = self.data_processor.create_time_series_features(df)
            
            # Prepare LSTM data
            logger.info("Preparing LSTM data...")
            lstm_data = self.data_processor.prepare_lstm_data(
                df_with_features,
                sequence_length=self.config.LSTM_CONFIG["sequence_length"]
            )
            
            # Save preprocessed data
            os.makedirs(output_dir, exist_ok=True)
            self.data_processor.save_preprocessed_data(lstm_data, output_dir)
            
            logger.info("Data preprocessing completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            return False
    
    def train_with_azure_automl(self, 
                               dataset_path: str = "sst_2025_global.csv") -> bool:
        """
        Train model using Azure AutoML.
        
        Args:
            dataset_path: Path to dataset
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.azure_pipeline:
                logger.error("Azure ML not set up. Run setup_azure_ml() first.")
                return False
            
            logger.info("Starting Azure AutoML training...")
            
            # Upload dataset
            dataset = self.azure_pipeline.upload_dataset(dataset_path)
            logger.info(f"Dataset uploaded: {dataset.name}")
            
            # Create AutoML config
            automl_config = self.azure_pipeline.create_lstm_automl_config(dataset)
            
            # Run experiment
            run = self.azure_pipeline.run_experiment(
                automl_config,
                experiment_name=self.config.EXPERIMENT_NAME
            )
            
            # Wait for completion
            logger.info("Waiting for experiment to complete...")
            run.wait_for_completion(show_output=True)
            
            # Deploy model
            model = self.azure_pipeline.deploy_model(run)
            logger.info(f"Model deployed: {model.name}")
            
            logger.info("Azure AutoML training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Azure AutoML training failed: {e}")
            return False
    
    def train_custom_lstm(self, 
                         preprocessed_data_path: str = "data/preprocessed") -> bool:
        """
        Train custom LSTM model.
        
        Args:
            preprocessed_data_path: Path to preprocessed data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting custom LSTM training...")
            
            # Initialize trainer
            self.trainer = AzureMLTrainer()
            
            # Train model
            model = self.trainer.train_model(
                preprocessed_data_path,
                model_params=self.config.get_lstm_config()
            )
            
            logger.info("Custom LSTM training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Custom LSTM training failed: {e}")
            return False
    
    def run_full_pipeline(self, 
                         input_file: str = "sst_2025_global.csv",
                         use_azure_automl: bool = True) -> bool:
        """
        Run the complete pipeline.
        
        Args:
            input_file: Path to input CSV file
            use_azure_automl: Whether to use Azure AutoML or custom LSTM
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting full SST LSTM pipeline...")
        start_time = datetime.now()
        
        try:
            # Step 1: Setup Azure ML (if using Azure)
            if use_azure_automl:
                if not self.setup_azure_ml():
                    logger.error("Failed to setup Azure ML. Switching to local training.")
                    use_azure_automl = False
            
            # Step 2: Preprocess data
            if not self.preprocess_data(input_file):
                logger.error("Data preprocessing failed. Aborting pipeline.")
                return False
            
            # Step 3: Train model
            if use_azure_automl:
                success = self.train_with_azure_automl(input_file)
            else:
                success = self.train_custom_lstm()
            
            if not success:
                logger.error("Model training failed.")
                return False
            
            # Pipeline completed successfully
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"Pipeline completed successfully in {duration}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False
    
    def create_inference_script(self, output_file: str = "inference.py"):
        """Create inference script for the trained model."""
        logger.info("Creating inference script...")
        
        inference_code = '''
"""
Inference Script for SST LSTM Model
Loads trained model and makes predictions on new data.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple
import joblib
import os

class SSTInference:
    """SST LSTM model inference."""
    
    def __init__(self, model_path: str, scalers_path: str):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model
            scalers_path: Path to scalers
        """
        self.model = tf.keras.models.load_model(model_path)
        self.scalers = joblib.load(scalers_path)
        
    def preprocess_input(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess input data for prediction."""
        # Apply same preprocessing as training
        feature_cols = [col for col in data.columns if col not in ['time', 'analysed_sst']]
        
        # Scale features
        data[feature_cols] = self.scalers['features'].transform(data[feature_cols])
        
        return data[feature_cols].values
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        predictions = self.model.predict(X)
        
        # Inverse transform predictions
        predictions = self.scalers['target'].inverse_transform(predictions)
        
        return predictions
    
    def predict_single_location(self, 
                               lat: float, 
                               lon: float, 
                               historical_data: pd.DataFrame) -> float:
        """Predict SST for a single location."""
        # Create sequence from historical data
        location_data = historical_data[
            (historical_data['lat'] == lat) & 
            (historical_data['lon'] == lon)
        ].sort_values('time')
        
        if len(location_data) < 24:
            raise ValueError("Insufficient historical data")
        
        # Get last 24 hours
        sequence = location_data.tail(24)
        X = self.preprocess_input(sequence).reshape(1, 24, -1)
        
        # Make prediction
        prediction = self.predict(X)[0][0]
        
        return prediction

# Example usage
if __name__ == "__main__":
    # Initialize inference
    inference = SSTInference("model.h5", "scalers.pkl")
    
    # Load new data
    new_data = pd.read_csv("new_sst_data.csv")
    
    # Make predictions
    predictions = inference.predict(new_data)
    print(f"Predictions: {predictions}")
'''
        
        with open(output_file, 'w') as f:
            f.write(inference_code)
        
        logger.info(f"Inference script created: {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='SST LSTM Pipeline')
    parser.add_argument('--input_file', type=str, default='sst_2025_global.csv',
                       help='Input CSV file path')
    parser.add_argument('--environment', type=str, default='development',
                       choices=['development', 'production'],
                       help='Environment configuration')
    parser.add_argument('--use_azure_automl', action='store_true',
                       help='Use Azure AutoML instead of custom LSTM')
    parser.add_argument('--preprocess_only', action='store_true',
                       help='Only run data preprocessing')
    parser.add_argument('--train_only', action='store_true',
                       help='Only run model training (requires preprocessed data)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SSTPipeline(environment=args.environment)
    
    # Run pipeline based on arguments
    if args.preprocess_only:
        success = pipeline.preprocess_data(args.input_file)
    elif args.train_only:
        if args.use_azure_automl:
            success = pipeline.train_with_azure_automl(args.input_file)
        else:
            success = pipeline.train_custom_lstm()
    else:
        success = pipeline.run_full_pipeline(
            args.input_file, 
            use_azure_automl=args.use_azure_automl
        )
    
    if success:
        logger.info("Pipeline completed successfully!")
        # Create inference script
        pipeline.create_inference_script()
    else:
        logger.error("Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

