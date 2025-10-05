#!/usr/bin/env python3
"""
Setup script for SST LSTM Pipeline
Handles initial configuration and environment setup.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_directories():
    """Create necessary directories."""
    print("📁 Creating project directories...")
    directories = [
        "data/preprocessed",
        "logs",
        "models",
        "outputs",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies."""
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")

def setup_azure_cli():
    """Setup Azure CLI if not already installed."""
    print("☁️ Checking Azure CLI...")
    try:
        result = subprocess.run("az --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Azure CLI is already installed")
            return True
    except:
        pass
    
    print("❌ Azure CLI not found. Please install it manually:")
    print("   - Windows: https://aka.ms/installazurecliwindows")
    print("   - macOS: brew install azure-cli")
    print("   - Linux: curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash")
    return False

def create_env_file():
    """Create .env file template."""
    print("📝 Creating environment file template...")
    
    env_content = """# Azure ML Configuration
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=sst-ml-resource-group
AZURE_WORKSPACE_NAME=sst-ml-workspace
AZURE_WORKSPACE_REGION=eastus

# Optional: Override default settings
# MAX_MEMORY_MB=4000
# SAMPLE_SIZE=1000000
# VM_SIZE=STANDARD_D2_V2
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Created .env file template")
    print("📝 Please edit .env file with your Azure credentials")

def validate_data_files():
    """Check if data files exist."""
    print("📊 Checking data files...")
    
    data_files = [
        "sst_2025_global.csv",
        "data/*.nc"
    ]
    
    found_files = []
    for pattern in data_files:
        if "*" in pattern:
            import glob
            files = glob.glob(pattern)
            found_files.extend(files)
        else:
            if os.path.exists(pattern):
                found_files.append(pattern)
    
    if found_files:
        print(f"✅ Found {len(found_files)} data files")
        for file in found_files[:5]:  # Show first 5 files
            print(f"   - {file}")
        if len(found_files) > 5:
            print(f"   ... and {len(found_files) - 5} more files")
    else:
        print("⚠️ No data files found. Please ensure you have:")
        print("   - sst_2025_global.csv in the root directory")
        print("   - NetCDF files in the data/ directory")

def run_quick_test():
    """Run a quick test to verify setup."""
    print("🧪 Running quick test...")
    
    try:
        # Test imports
        import pandas as pd
        import numpy as np
        import tensorflow as tf
        print("✅ Core dependencies imported successfully")
        
        # Test data processing
        from src.data_preprocessing import SSTDataProcessor
        processor = SSTDataProcessor()
        print("✅ Data processor initialized successfully")
        
        # Test Azure config
        from config.azure_config import AzureConfig
        config = AzureConfig()
        print("✅ Azure configuration loaded successfully")
        
        print("✅ All tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup SST LSTM Pipeline")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-azure", action="store_true", help="Skip Azure CLI check")
    parser.add_argument("--test-only", action="store_true", help="Only run tests")
    
    args = parser.parse_args()
    
    print("🚀 Setting up SST LSTM Pipeline...")
    print("=" * 50)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            success = False
    
    # Setup Azure CLI
    if not args.skip_azure:
        if not setup_azure_cli():
            print("⚠️ Azure CLI setup failed, but you can continue with local training")
    
    # Create environment file
    create_env_file()
    
    # Validate data files
    validate_data_files()
    
    # Run tests
    if not run_quick_test():
        success = False
    
    print("=" * 50)
    if success:
        print("🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Edit .env file with your Azure credentials")
        print("2. Run: python main_pipeline.py --preprocess_only")
        print("3. Run: python main_pipeline.py --use_azure_automl")
        print("\n📚 For more information, see README.md")
    else:
        print("❌ Setup completed with errors. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

