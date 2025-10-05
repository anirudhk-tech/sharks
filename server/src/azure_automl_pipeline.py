"""
Azure AutoML Pipeline for Sea Surface Temperature LSTM Prediction
Handles Azure ML workspace setup, data upload, and AutoML configuration.
"""

import os
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Azure ML imports
from azureml.core import Workspace, Dataset, Experiment, Run
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.train.automl import AutoMLConfig
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.data.dataset_factory import TabularDatasetFactory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureAutoMLPipeline:
    """Azure AutoML pipeline for SST LSTM prediction."""
    
    def __init__(self, 
                 subscription_id: str,
                 resource_group: str,
                 workspace_name: str,
                 workspace_region: str = "eastus"):
        """
        Initialize Azure AutoML pipeline.
        
        Args:
            subscription_id: Azure subscription ID
            resource_group: Azure resource group name
            workspace_name: Azure ML workspace name
            workspace_region: Azure region for workspace
        """
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.workspace_region = workspace_region
        self.workspace = None
        self.compute_target = None
        
    def setup_workspace(self) -> Workspace:
        """Set up Azure ML workspace."""
        logger.info("Setting up Azure ML workspace...")
        
        try:
            # Try to get existing workspace
            self.workspace = Workspace.get(
                name=self.workspace_name,
                subscription_id=self.subscription_id,
                resource_group=self.resource_group
            )
            logger.info(f"Found existing workspace: {self.workspace_name}")
            
        except Exception:
            # Create new workspace
            logger.info(f"Creating new workspace: {self.workspace_name}")
            self.workspace = Workspace.create(
                name=self.workspace_name,
                subscription_id=self.subscription_id,
                resource_group=self.resource_group,
                location=self.workspace_region,
                create_resource_group=True,
                exist_ok=True
            )
            logger.info(f"Created workspace: {self.workspace_name}")
        
        return self.workspace
    
    def setup_compute_target(self, 
                           compute_name: str = "sst-compute",
                           vm_size: str = "STANDARD_D2_V2",
                           min_nodes: int = 0,
                           max_nodes: int = 4) -> ComputeTarget:
        """Set up compute target for training."""
        logger.info("Setting up compute target...")
        
        try:
            # Try to get existing compute target
            self.compute_target = ComputeTarget(
                workspace=self.workspace,
                name=compute_name
            )
            logger.info(f"Found existing compute target: {compute_name}")
            
        except ComputeTargetException:
            # Create new compute target
            logger.info(f"Creating new compute target: {compute_name}")
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=vm_size,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                idle_seconds_before_scaledown=300
            )
            
            self.compute_target = ComputeTarget.create(
                self.workspace,
                compute_name,
                compute_config
            )
            self.compute_target.wait_for_completion(show_output=True)
            logger.info(f"Created compute target: {compute_name}")
        
        return self.compute_target
    
    def create_environment(self) -> Environment:
        """Create Azure ML environment with required packages."""
        logger.info("Creating Azure ML environment...")
        
        # Create conda dependencies
        conda_deps = CondaDependencies()
        conda_deps.add_conda_package("python=3.8")
        conda_deps.add_conda_package("pip")
        
        # Add pip packages
        pip_packages = [
            "azureml-sdk",
            "azureml-train-automl",
            "pandas",
            "numpy",
            "scikit-learn",
            "tensorflow",
            "keras",
            "matplotlib",
            "seaborn",
            "xarray",
            "netcdf4",
            "psutil"
        ]
        
        for package in pip_packages:
            conda_deps.add_pip_package(package)
        
        # Create environment
        env = Environment("sst-lstm-env")
        env.python.conda_dependencies = conda_deps
        
        return env
    
    def upload_dataset(self, 
                      local_data_path: str,
                      dataset_name: str = "sst-dataset") -> Dataset:
        """Upload local dataset to Azure ML."""
        logger.info(f"Uploading dataset from {local_data_path}...")
        
        # Create dataset from local files
        dataset = Dataset.Tabular.from_delimited_files(
            path=[(self.workspace.get_default_datastore(), local_data_path)]
        )
        
        # Register dataset
        dataset = dataset.register(
            workspace=self.workspace,
            name=dataset_name,
            description="Sea Surface Temperature dataset for LSTM prediction",
            create_new_version=True
        )
        
        logger.info(f"Dataset uploaded and registered: {dataset_name}")
        return dataset
    
    def create_automl_config(self, 
                           dataset: Dataset,
                           target_column: str = "analysed_sst",
                           task: str = "regression",
                           primary_metric: str = "normalized_root_mean_squared_error",
                           max_trials: int = 50,
                           max_concurrent_trials: int = 4,
                           timeout_minutes: int = 120) -> AutoMLConfig:
        """Create AutoML configuration for time series forecasting."""
        logger.info("Creating AutoML configuration...")
        
        # Define AutoML settings
        automl_settings = {
            "task": task,
            "primary_metric": primary_metric,
            "training_data": dataset,
            "label_column_name": target_column,
            "n_cross_validations": 3,
            "max_trials": max_trials,
            "max_concurrent_trials": max_concurrent_trials,
            "experiment_timeout_minutes": timeout_minutes,
            "enable_early_stopping": True,
            "featurization": "auto",
            "verbosity": logging.INFO,
            "compute_target": self.compute_target,
            "enable_onnx_compatible_models": True
        }
        
        # Add time series specific settings
        if task == "forecasting":
            automl_settings.update({
                "forecast_horizon": 24,  # Predict next 24 hours
                "target_lags": [1, 2, 3, 6, 12, 24],  # Lag features
                "target_rolling_window_size": 24,  # Rolling window
                "time_column_name": "time",
                "grain_column_names": ["lat", "lon"]  # Group by location
            })
        
        # Create AutoML config
        automl_config = AutoMLConfig(**automl_settings)
        
        return automl_config
    
    def create_lstm_automl_config(self, 
                                dataset: Dataset,
                                target_column: str = "analysed_sst") -> AutoMLConfig:
        """Create specialized AutoML config for LSTM time series prediction."""
        logger.info("Creating LSTM-specific AutoML configuration...")
        
        # Custom featurization for time series
        featurization_config = {
            "time_column_name": "time",
            "grain_column_names": ["lat", "lon"],
            "drop_columns": ["time"],  # Drop time column as it's handled separately
            "categorical_columns": ["is_tropical", "is_polar"],
            "numerical_columns": [
                "analysed_sst", "lat", "lon", "hour", "day_of_year", 
                "day_of_week", "month", "hour_sin", "hour_cos", 
                "day_sin", "day_cos", "distance_from_equator",
                "lon_sin", "lon_cos"
            ]
        }
        
        # LSTM-specific settings
        automl_settings = {
            "task": "forecasting",
            "primary_metric": "normalized_root_mean_squared_error",
            "training_data": dataset,
            "label_column_name": target_column,
            "n_cross_validations": 3,
            "max_trials": 30,  # Fewer trials for LSTM
            "max_concurrent_trials": 2,  # LSTM is memory intensive
            "experiment_timeout_minutes": 180,  # Longer timeout for LSTM
            "enable_early_stopping": True,
            "featurization": featurization_config,
            "verbosity": logging.INFO,
            "compute_target": self.compute_target,
            "enable_onnx_compatible_models": False,  # LSTM models not ONNX compatible
            "forecast_horizon": 24,
            "target_lags": [1, 2, 3, 6, 12, 24],
            "target_rolling_window_size": 24,
            "time_column_name": "time",
            "grain_column_names": ["lat", "lon"],
            "allowed_models": ["LSTM", "GRU", "RNN"],  # Restrict to RNN models
            "blocked_models": ["LinearRegression", "RandomForest", "XGBoost"]  # Block non-RNN models
        }
        
        automl_config = AutoMLConfig(**automl_settings)
        return automl_config
    
    def run_experiment(self, 
                      automl_config: AutoMLConfig,
                      experiment_name: str = "sst-lstm-prediction") -> Run:
        """Run AutoML experiment."""
        logger.info(f"Starting experiment: {experiment_name}")
        
        # Create experiment
        experiment = Experiment(workspace=self.workspace, name=experiment_name)
        
        # Submit experiment
        run = experiment.submit(automl_config, show_output=True)
        
        logger.info(f"Experiment submitted. Run ID: {run.id}")
        return run
    
    def deploy_model(self, 
                    run: Run,
                    model_name: str = "sst-lstm-model",
                    deployment_name: str = "sst-lstm-deployment") -> Any:
        """Deploy the best model from the experiment."""
        logger.info("Deploying best model...")
        
        # Get the best run
        best_run = run.get_best_child()
        
        # Register model
        model = best_run.register_model(
            model_name=model_name,
            model_path="outputs",
            description="LSTM model for Sea Surface Temperature prediction"
        )
        
        # Deploy model (this would require additional setup for ACI/AKS)
        logger.info(f"Model registered: {model_name}")
        logger.info("Model deployment requires additional ACI/AKS setup")
        
        return model
    
    def create_training_pipeline(self, 
                               dataset: Dataset,
                               pipeline_name: str = "sst-training-pipeline") -> Pipeline:
        """Create a complete training pipeline."""
        logger.info("Creating training pipeline...")
        
        # Create pipeline data
        preprocessed_data = PipelineData(
            name="preprocessed_data",
            datastore=self.workspace.get_default_datastore()
        )
        
        # Preprocessing step
        preprocessing_step = PythonScriptStep(
            name="preprocessing",
            script_name="data_preprocessing.py",
            source_directory="../src",
            arguments=[
                "--input_data", dataset.as_named_input("input_data"),
                "--output_data", preprocessed_data
            ],
            compute_target=self.compute_target,
            runconfig=self.create_run_config(),
            allow_reuse=True
        )
        
        # Training step
        training_step = PythonScriptStep(
            name="training",
            script_name="lstm_training.py",
            source_directory="../src",
            arguments=[
                "--input_data", preprocessed_data,
                "--output_model", PipelineData(name="model_output")
            ],
            compute_target=self.compute_target,
            runconfig=self.create_run_config(),
            allow_reuse=True
        )
        
        # Create pipeline
        pipeline = Pipeline(
            workspace=self.workspace,
            steps=[preprocessing_step, training_step]
        )
        
        return pipeline
    
    def create_run_config(self) -> RunConfiguration:
        """Create run configuration for pipeline steps."""
        run_config = RunConfiguration()
        run_config.environment = self.create_environment()
        return run_config


def main():
    """Main function to run the Azure AutoML pipeline."""
    # Configuration - Replace with your Azure credentials
    SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "your-subscription-id")
    RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "your-resource-group")
    WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME", "sst-ml-workspace")
    
    # Initialize pipeline
    pipeline = AzureAutoMLPipeline(
        subscription_id=SUBSCRIPTION_ID,
        resource_group=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )
    
    # Setup workspace and compute
    workspace = pipeline.setup_workspace()
    compute_target = pipeline.setup_compute_target()
    
    # Upload dataset
    dataset = pipeline.upload_dataset("../sst_2025_global.csv")
    
    # Create AutoML config
    automl_config = pipeline.create_lstm_automl_config(dataset)
    
    # Run experiment
    run = pipeline.run_experiment(automl_config)
    
    # Wait for completion
    run.wait_for_completion(show_output=True)
    
    # Deploy model
    model = pipeline.deploy_model(run)
    
    logger.info("Azure AutoML pipeline completed successfully!")


if __name__ == "__main__":
    main()

