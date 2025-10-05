#!/usr/bin/env python3
"""
Load environment variables from azure.env file
"""

import os
from pathlib import Path

def load_env_file(env_file="azure.env"):
    """Load environment variables from a file."""
    env_path = Path(env_file)
    
    if not env_path.exists():
        print(f"Environment file {env_file} not found.")
        return False
    
    print(f"Loading environment variables from {env_file}...")
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse key=value pairs
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                # Set environment variable
                os.environ[key] = value
                print(f"  {key} = {value}")
    
    print("Environment variables loaded successfully!")
    return True

if __name__ == "__main__":
    load_env_file()


