#!/usr/bin/env python3
"""
Example Usage Script for SST LSTM Pipeline
Demonstrates how to use the pipeline for different scenarios.
"""

import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

from data_preprocessing import SSTDataProcessor
from azure_automl_pipeline import AzureAutoMLPipeline
from azure_config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_data_preprocessing():
    """Example: Data preprocessing only."""
    print("🔍 Example 1: Data Preprocessing")
    print("-" * 40)
    
    # Initialize processor
    processor = SSTDataProcessor(
        max_memory_mb=2000,  # Lower memory for demo
        chunk_size=50000
    )
    
    # Check if data exists
    if not os.path.exists("sst_2025_global.csv"):
        print("❌ Data file not found. Please ensure sst_2025_global.csv exists.")
        return False
    
    try:
        # Load and sample data
        print("📊 Loading data...")
        import pandas as pd
        df = pd.read_csv("sst_2025_global.csv", nrows=100000)  # Sample for demo
        
        # Create features
        print("🔧 Creating features...")
        df_with_features = processor.create_time_series_features(df)
        
        # Prepare LSTM data
        print("🎯 Preparing LSTM data...")
        lstm_data = processor.prepare_lstm_data(df_with_features, sequence_length=12)
        
        # Save preprocessed data
        print("💾 Saving preprocessed data...")
        processor.save_preprocessed_data(lstm_data, "data/preprocessed_demo/")
        
        print("✅ Data preprocessing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def example_azure_setup():
    """Example: Azure ML setup (without actual training)."""
    print("\n☁️ Example 2: Azure ML Setup")
    print("-" * 40)
    
    # Get configuration
    config = get_config("development")
    
    # Check if Azure credentials are set
    if not config.validate_config():
        print("⚠️ Azure credentials not configured. Skipping Azure setup.")
        print("To use Azure AutoML:")
        print("1. Set environment variables:")
        print("   export AZURE_SUBSCRIPTION_ID='your-subscription-id'")
        print("   export AZURE_RESOURCE_GROUP='your-resource-group'")
        print("   export AZURE_WORKSPACE_NAME='your-workspace-name'")
        print("2. Run: az login")
        return False
    
    try:
        # Initialize Azure pipeline
        azure_pipeline = AzureAutoMLPipeline(
            subscription_id=config.SUBSCRIPTION_ID,
            resource_group=config.RESOURCE_GROUP,
            workspace_name=config.WORKSPACE_NAME
        )
        
        # Setup workspace (this will create if it doesn't exist)
        print("🏗️ Setting up Azure ML workspace...")
        workspace = azure_pipeline.setup_workspace()
        print(f"✅ Workspace ready: {workspace.name}")
        
        # Setup compute target
        print("💻 Setting up compute target...")
        compute_target = azure_pipeline.setup_compute_target(
            compute_name="demo-compute",
            vm_size="STANDARD_D2_V2",
            min_nodes=0,
            max_nodes=2
        )
        print(f"✅ Compute target ready: {compute_target.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Azure setup failed: {e}")
        print("This is normal if you don't have Azure credentials configured.")
        return False

def example_custom_training():
    """Example: Custom LSTM training (simplified)."""
    print("\n🧠 Example 3: Custom LSTM Training")
    print("-" * 40)
    
    try:
        # Check if preprocessed data exists
        if not os.path.exists("data/preprocessed_demo/train_X.npy"):
            print("❌ Preprocessed data not found. Run example 1 first.")
            return False
        
        # Import training modules
        from lstm_training import SSTLSTMModel
        import numpy as np
        
        # Load preprocessed data
        print("📊 Loading preprocessed data...")
        X_train = np.load("data/preprocessed_demo/train_X.npy")
        y_train = np.load("data/preprocessed_demo/train_y.npy")
        X_val = np.load("data/preprocessed_demo/val_X.npy")
        y_val = np.load("data/preprocessed_demo/val_y.npy")
        
        print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}")
        
        # Initialize model
        print("🏗️ Building LSTM model...")
        model = SSTLSTMModel(
            sequence_length=X_train.shape[1],
            n_features=X_train.shape[2],
            lstm_units=64,  # Smaller for demo
            dropout_rate=0.2,
            learning_rate=0.001
        )
        
        # Build model
        model.build_model()
        
        # Train model (short training for demo)
        print("🚀 Training model...")
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=5,  # Short training for demo
            batch_size=32,
            verbose=1
        )
        
        # Evaluate model
        print("📈 Evaluating model...")
        train_metrics = model.evaluate(X_train, y_train)
        val_metrics = model.evaluate(X_val, y_val)
        
        print(f"Training RMSE: {train_metrics['rmse']:.4f}")
        print(f"Validation RMSE: {val_metrics['rmse']:.4f}")
        
        # Save model
        model.save_model("models/demo_lstm_model.h5")
        print("💾 Model saved to models/demo_lstm_model.h5")
        
        print("✅ Custom training completed!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def example_inference():
    """Example: Model inference."""
    print("\n🔮 Example 4: Model Inference")
    print("-" * 40)
    
    try:
        # Check if model exists
        if not os.path.exists("models/demo_lstm_model.h5"):
            print("❌ Model not found. Run example 3 first.")
            return False
        
        # Import inference modules
        from lstm_training import SSTLSTMModel
        import numpy as np
        
        # Load model
        print("📥 Loading trained model...")
        model = SSTLSTMModel()
        model.load_model("models/demo_lstm_model.h5")
        
        # Load test data
        if os.path.exists("data/preprocessed_demo/test_X.npy"):
            X_test = np.load("data/preprocessed_demo/test_X.npy")
            y_test = np.load("data/preprocessed_demo/test_y.npy")
            
            # Make predictions
            print("🔮 Making predictions...")
            predictions = model.predict(X_test[:10])  # Predict on first 10 samples
            
            # Show results
            print("Sample predictions:")
            for i in range(5):
                print(f"  Actual: {y_test[i]:.4f}, Predicted: {predictions[i][0]:.4f}")
            
            # Calculate metrics
            test_metrics = model.evaluate(X_test, y_test)
            print(f"Test RMSE: {test_metrics['rmse']:.4f}")
            
        else:
            print("⚠️ Test data not found. Creating dummy data for demo...")
            # Create dummy data for demo
            dummy_X = np.random.randn(5, 12, 15)  # 5 samples, 12 timesteps, 15 features
            predictions = model.predict(dummy_X)
            print("Dummy predictions:")
            for i, pred in enumerate(predictions):
                print(f"  Sample {i+1}: {pred[0]:.4f}")
        
        print("✅ Inference completed!")
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

def main():
    """Run all examples."""
    print("🚀 SST LSTM Pipeline Examples")
    print("=" * 50)
    
    examples = [
        ("Data Preprocessing", example_data_preprocessing),
        ("Azure ML Setup", example_azure_setup),
        ("Custom LSTM Training", example_custom_training),
        ("Model Inference", example_inference)
    ]
    
    results = []
    
    for name, example_func in examples:
        try:
            result = example_func()
            results.append((name, result))
        except KeyboardInterrupt:
            print(f"\n⏹️ Interrupted during {name}")
            break
        except Exception as e:
            print(f"❌ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Example Results Summary:")
    for name, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {name}: {status}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n🎯 Completed {successful}/{total} examples successfully")
    
    if successful == total:
        print("🎉 All examples completed successfully!")
        print("\n📚 Next steps:")
        print("1. Configure Azure credentials for full Azure AutoML pipeline")
        print("2. Use the full dataset for production training")
        print("3. Deploy your model for real-time predictions")
    else:
        print("⚠️ Some examples failed. Check the output above for details.")

if __name__ == "__main__":
    main()

