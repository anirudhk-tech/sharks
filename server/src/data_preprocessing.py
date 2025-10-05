"""
Data Preprocessing Pipeline for Sea Surface Temperature LSTM Model
Handles large datasets efficiently with chunked processing and memory management.
"""

import pandas as pd
import numpy as np
import xarray as xr
import glob
import os
import gc
import psutil
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SSTDataProcessor:
    """Handles preprocessing of large SST datasets for LSTM modeling."""
    
    def __init__(self, 
                 max_memory_mb: int = 4000,
                 chunk_size: int = 100000,
                 target_columns: List[str] = None):
        """
        Initialize the SST data processor.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            chunk_size: Size of chunks for processing
            target_columns: Columns to use as features
        """
        self.max_memory_mb = max_memory_mb
        self.chunk_size = chunk_size
        self.target_columns = target_columns or ['analysed_sst', 'lat', 'lon']
        self.scalers = {}
        self.feature_columns = []
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def process_netcdf_files(self, data_dir: str, output_file: str = None) -> str:
        """
        Process NetCDF files and create a consolidated CSV.
        
        Args:
            data_dir: Directory containing NetCDF files
            output_file: Output CSV file path
            
        Returns:
            Path to the output CSV file
        """
        logger.info("Starting NetCDF file processing...")
        
        # Find all NetCDF files
        netcdf_files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
        logger.info(f"Found {len(netcdf_files)} NetCDF files")
        
        if not netcdf_files:
            raise ValueError(f"No NetCDF files found in {data_dir}")
        
        # Process files in chunks
        all_dataframes = []
        
        for i, file_path in enumerate(netcdf_files):
            logger.info(f"Processing file {i+1}/{len(netcdf_files)}: {os.path.basename(file_path)}")
            
            if self.get_memory_usage() > self.max_memory_mb:
                logger.warning(f"Memory usage ({self.get_memory_usage():.1f} MB) exceeds limit")
                break
                
            df = self._process_single_netcdf(file_path)
            if df is not None:
                all_dataframes.append(df)
                
            # Clean up memory
            gc.collect()
        
        # Combine and save
        if all_dataframes:
            logger.info("Combining all dataframes...")
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            if output_file is None:
                output_file = "sst_processed_data.csv"
                
            combined_df.to_csv(output_file, index=False)
            logger.info(f"Saved processed data to {output_file}")
            
            return output_file
        else:
            raise ValueError("No data was successfully processed")
    
    def _process_single_netcdf(self, file_path: str) -> Optional[pd.DataFrame]:
        """Process a single NetCDF file."""
        try:
            ds = xr.open_dataset(file_path)
            raw = ds['analysed_sst']
            
            # Get scaling parameters
            scale = raw.attrs.get('scale_factor', 1.0)
            offset = raw.attrs.get('add_offset', 0.0)
            fill_value = raw.attrs.get('_FillValue', None)
            
            # Apply scaling and conversion to Celsius
            sst_c = raw.where(raw != fill_value) * scale + offset - 273.15
            
            # Convert to DataFrame
            df = sst_c.to_dataframe().reset_index()
            df = df.dropna(subset=["analysed_sst"])
            
            # Clean up
            ds.close()
            del ds, raw, sst_c
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time series features for LSTM modeling.
        
        Args:
            df: Input DataFrame with time, lat, lon, analysed_sst columns
            
        Returns:
            DataFrame with additional time series features
        """
        logger.info("Creating time series features...")
        
        # Ensure time column is datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Sort by time and location
        df = df.sort_values(['lat', 'lon', 'time']).reset_index(drop=True)
        
        # Create time-based features
        df['hour'] = df['time'].dt.hour
        df['day_of_year'] = df['time'].dt.dayofyear
        df['day_of_week'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        
        # Create cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Create lag features for each location
        logger.info("Creating lag features...")
        df = self._create_lag_features(df)
        
        # Create rolling statistics
        logger.info("Creating rolling statistics...")
        df = self._create_rolling_features(df)
        
        # Create spatial features
        logger.info("Creating spatial features...")
        df = self._create_spatial_features(df)
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """Create lag features for each location."""
        for lag in lags:
            df[f'sst_lag_{lag}'] = df.groupby(['lat', 'lon'])['analysed_sst'].shift(lag)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame, windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        """Create rolling statistics for each location."""
        for window in windows:
            # Create rolling mean
            rolling_mean = df.groupby(['lat', 'lon'])['analysed_sst'].rolling(
                window=window, min_periods=1
            ).mean()
            df[f'sst_rolling_mean_{window}'] = rolling_mean.values
            
            # Create rolling std
            rolling_std = df.groupby(['lat', 'lon'])['analysed_sst'].rolling(
                window=window, min_periods=1
            ).std()
            df[f'sst_rolling_std_{window}'] = rolling_std.values
        
        return df
    
    def _create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spatial features."""
        # Distance from equator
        df['distance_from_equator'] = np.abs(df['lat'])
        
        # Ocean basin indicators (simplified)
        df['is_tropical'] = (df['lat'].abs() <= 23.5).astype(int)
        df['is_polar'] = (df['lat'].abs() >= 66.5).astype(int)
        
        # Longitude wrapping for circular nature
        df['lon_sin'] = np.sin(2 * np.pi * df['lon'] / 360)
        df['lon_cos'] = np.cos(2 * np.pi * df['lon'] / 360)
        
        return df
    
    def load_csv_in_chunks(self, 
                          file_path: str, 
                          chunk_size: int = None,
                          max_rows: int = None) -> pd.DataFrame:
        """
        Load large CSV file in chunks to manage memory usage.
        
        Args:
            file_path: Path to the CSV file
            chunk_size: Size of each chunk (defaults to self.chunk_size)
            max_rows: Maximum number of rows to read (None for all)
            
        Returns:
            Combined DataFrame from all chunks
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        logger.info(f"Loading CSV file in chunks of {chunk_size} rows...")
        
        chunks = []
        total_rows = 0
        
        try:
            # First, get the total number of rows for progress tracking
            total_file_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
            logger.info(f"Total rows in file: {total_file_rows:,}")
            
            # Read file in chunks
            for chunk_num, chunk_df in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                logger.info(f"Processing chunk {chunk_num + 1} ({len(chunk_df):,} rows)")
                
                # Check memory usage
                current_memory = self.get_memory_usage()
                if current_memory > self.max_memory_mb:
                    logger.warning(f"Memory usage ({current_memory:.1f} MB) exceeds limit, stopping chunk processing")
                    break
                
                chunks.append(chunk_df)
                total_rows += len(chunk_df)
                
                # Check if we've reached the max_rows limit
                if max_rows and total_rows >= max_rows:
                    logger.info(f"Reached maximum rows limit ({max_rows:,})")
                    break
                
                # Progress update
                if chunk_num % 10 == 0:
                    progress = (total_rows / total_file_rows) * 100
                    logger.info(f"Progress: {progress:.1f}% ({total_rows:,}/{total_file_rows:,} rows)")
            
            # Combine all chunks
            if chunks:
                logger.info("Combining chunks...")
                combined_df = pd.concat(chunks, ignore_index=True)
                
                # Clean up chunks to free memory
                del chunks
                gc.collect()
                
                logger.info(f"Successfully loaded {len(combined_df):,} rows")
                return combined_df
            else:
                raise ValueError("No data was loaded from the CSV file")
                
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise

    def process_csv_chunks_streaming(self, 
                                   file_path: str, 
                                   output_dir: str,
                                   chunk_size: int = None,
                                   max_chunks: int = None) -> str:
        """
        Process CSV file in streaming mode - process each chunk individually
        without loading the entire file into memory.
        
        Args:
            file_path: Path to the CSV file
            output_dir: Directory to save processed chunks
            chunk_size: Size of each chunk
            max_chunks: Maximum number of chunks to process
            
        Returns:
            Path to the output directory
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        logger.info(f"Processing CSV file in streaming mode with chunks of {chunk_size} rows...")
        
        os.makedirs(output_dir, exist_ok=True)
        processed_chunks = []
        
        try:
            chunk_num = 0
            for chunk_df in pd.read_csv(file_path, chunksize=chunk_size):
                if max_chunks and chunk_num >= max_chunks:
                    logger.info(f"Reached maximum chunks limit ({max_chunks})")
                    break
                    
                logger.info(f"Processing chunk {chunk_num + 1} ({len(chunk_df):,} rows)")
                
                # Check memory usage
                current_memory = self.get_memory_usage()
                if current_memory > self.max_memory_mb:
                    logger.warning(f"Memory usage ({current_memory:.1f} MB) exceeds limit, stopping")
                    break
                
                # Process this chunk
                try:
                    # Create features for this chunk
                    chunk_with_features = self.create_time_series_features(chunk_df)
                    
                    # Save processed chunk
                    chunk_output_file = os.path.join(output_dir, f'processed_chunk_{chunk_num:04d}.csv')
                    chunk_with_features.to_csv(chunk_output_file, index=False)
                    processed_chunks.append(chunk_output_file)
                    
                    logger.info(f"Saved processed chunk to {chunk_output_file}")
                    
                    # Clean up memory
                    del chunk_df, chunk_with_features
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_num}: {e}")
                    continue
                
                chunk_num += 1
            
            logger.info(f"Successfully processed {len(processed_chunks)} chunks")
            return output_dir
            
        except Exception as e:
            logger.error(f"Error in streaming processing: {e}")
            raise

    def prepare_lstm_data(self, 
                         df: pd.DataFrame, 
                         sequence_length: int = 24,
                         target_column: str = 'analysed_sst',
                         test_size: float = 0.2,
                         val_size: float = 0.1) -> Dict:
        """
        Prepare data for LSTM training.
        
        Args:
            df: Input DataFrame with features
            sequence_length: Length of input sequences
            target_column: Name of target column
            test_size: Fraction of data for testing
            val_size: Fraction of data for validation
            
        Returns:
            Dictionary containing train/val/test data and scalers
        """
        logger.info("Preparing LSTM data...")
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in ['time', target_column]]
        self.feature_columns = feature_cols
        
        # Remove rows with NaN values
        df_clean = df.dropna().reset_index(drop=True)
        logger.info(f"Data shape after cleaning: {df_clean.shape}")
        
        # Split data by location to avoid data leakage
        locations = df_clean[['lat', 'lon']].drop_duplicates()
        train_locs, test_locs = train_test_split(locations, test_size=test_size, random_state=42)
        train_locs, val_locs = train_test_split(train_locs, test_size=val_size/(1-test_size), random_state=42)
        
        # Create splits
        train_data = df_clean.merge(train_locs, on=['lat', 'lon'])
        val_data = df_clean.merge(val_locs, on=['lat', 'lon'])
        test_data = df_clean.merge(test_locs, on=['lat', 'lon'])
        
        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Scale features
        train_data, val_data, test_data = self._scale_data(train_data, val_data, test_data, feature_cols)
        
        # Create sequences
        train_sequences = self._create_sequences(train_data, feature_cols, target_column, sequence_length)
        val_sequences = self._create_sequences(val_data, feature_cols, target_column, sequence_length)
        test_sequences = self._create_sequences(test_data, feature_cols, target_column, sequence_length)
        
        return {
            'train': train_sequences,
            'val': val_sequences,
            'test': test_sequences,
            'feature_columns': feature_cols,
            'scalers': self.scalers
        }
    
    def _scale_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                   test_data: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Scale the data using StandardScaler."""
        logger.info("Scaling data...")
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        train_data[feature_cols] = self.scalers['features'].fit_transform(train_data[feature_cols])
        val_data[feature_cols] = self.scalers['features'].transform(val_data[feature_cols])
        test_data[feature_cols] = self.scalers['features'].transform(test_data[feature_cols])
        
        # Scale target
        self.scalers['target'] = StandardScaler()
        train_data['analysed_sst'] = self.scalers['target'].fit_transform(train_data[['analysed_sst']])
        val_data['analysed_sst'] = self.scalers['target'].transform(val_data[['analysed_sst']])
        test_data['analysed_sst'] = self.scalers['target'].transform(test_data[['analysed_sst']])
        
        return train_data, val_data, test_data
    
    def _create_sequences(self, df: pd.DataFrame, feature_cols: List[str], 
                         target_col: str, sequence_length: int) -> Dict:
        """Create sequences for LSTM training."""
        logger.info(f"Creating sequences of length {sequence_length}...")
        
        X, y = [], []
        
        # Group by location
        for (lat, lon), group in df.groupby(['lat', 'lon']):
            group = group.sort_values('time').reset_index(drop=True)
            
            if len(group) < sequence_length + 1:
                continue
                
            # Create sequences
            for i in range(len(group) - sequence_length):
                X.append(group[feature_cols].iloc[i:i+sequence_length].values)
                y.append(group[target_col].iloc[i+sequence_length])
        
        return {
            'X': np.array(X),
            'y': np.array(y)
        }
    
    def save_preprocessed_data(self, data: Dict, output_dir: str):
        """Save preprocessed data and scalers."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data
        np.save(os.path.join(output_dir, 'train_X.npy'), data['train']['X'])
        np.save(os.path.join(output_dir, 'train_y.npy'), data['train']['y'])
        np.save(os.path.join(output_dir, 'val_X.npy'), data['val']['X'])
        np.save(os.path.join(output_dir, 'val_y.npy'), data['val']['y'])
        np.save(os.path.join(output_dir, 'test_X.npy'), data['test']['X'])
        np.save(os.path.join(output_dir, 'test_y.npy'), data['test']['y'])
        
        # Save scalers
        joblib.dump(data['scalers'], os.path.join(output_dir, 'scalers.pkl'))
        
        # Save feature columns
        with open(os.path.join(output_dir, 'feature_columns.txt'), 'w') as f:
            f.write('\n'.join(data['feature_columns']))
        
        logger.info(f"Preprocessed data saved to {output_dir}")


def main():
    """Main function to run the preprocessing pipeline."""
    # Initialize processor
    processor = SSTDataProcessor(max_memory_mb=4000, chunk_size=10000)
    
    # Process NetCDF files (if needed)
    # processor.process_netcdf_files('../data/', 'sst_processed_data.csv')
    
    # Choose processing method based on file size and memory constraints
    processing_mode = "chunked_loading"  # Options: "chunked_loading" or "streaming"
    
    if processing_mode == "chunked_loading":
        # Method 1: Load CSV in chunks and combine (good for moderate file sizes)
        logger.info("Using chunked loading method...")
        df = processor.load_csv_in_chunks('../sst_2025_global.csv', 
                                         chunk_size=50000,  # Smaller chunks for better memory management
                                         max_rows=1000000)  # Limit for faster processing
        
        # Create features
        df_with_features = processor.create_time_series_features(df)
        
        # Prepare LSTM data
        lstm_data = processor.prepare_lstm_data(df_with_features, sequence_length=24)
        
        # Save preprocessed data
        processor.save_preprocessed_data(lstm_data, '../data/preprocessed/')
        
    elif processing_mode == "streaming":
        # Method 2: Process chunks individually without loading everything (best for very large files)
        logger.info("Using streaming processing method...")
        output_dir = processor.process_csv_chunks_streaming('../sst_2025_global.csv',
                                                           '../data/processed_chunks/',
                                                           chunk_size=25000,  # Even smaller chunks for streaming
                                                           max_chunks=40)     # Limit for faster processing
        
        logger.info(f"Processed chunks saved to: {output_dir}")
        logger.info("Note: For LSTM training, you'll need to combine and process these chunks further")
    
    logger.info("Preprocessing completed successfully!")


if __name__ == "__main__":
    main()

