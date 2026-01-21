"""
Data loading utilities for historical solar data.
Handles loading yield, GHI (Global Horizontal Irradiance), and temperature data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class SolarDataLoader:
    """Load and preprocess solar time-series data."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load solar data from CSV file.
        Expected columns: timestamp, yield, ghi, temp (minimum required)
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        return df
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data integrity and fill minor gaps.
        """
        required_cols = ['yield', 'ghi', 'temp']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Remove rows with NaN
        df = df.dropna()
        
        # Remove negative values (physical impossibility)
        df = df[(df['yield'] >= 0) & (df['ghi'] >= 0)]
        
        return df
    
    def get_daily_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sub-daily data to daily summaries.
        """
        daily = df.resample('D').agg({
            'yield': 'sum',
            'ghi': 'sum',
            'temp': 'mean'
        })
        
        # Remove days with no data
        daily = daily[daily['yield'] > 0]
        
        return daily
    
    def split_train_test(
        self, 
        df: pd.DataFrame, 
        test_fraction: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Time-based train/test split.
        Recent data goes to test set.
        """
        split_idx = int(len(df) * (1 - test_fraction))
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        return train, test
