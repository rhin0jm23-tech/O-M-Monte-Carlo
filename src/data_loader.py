"""
Data loading utilities for solar daily data from vcom or similar systems.
Simple format: Date, Energy (kWh), POA Irradiance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class SolarDataLoader:
    """Load and preprocess daily solar data."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        # Don't create the directory - just use it if it exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir.resolve()}")
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load daily solar data from CSV file.
        
        Filename format: Energy_generation_YYYY_MM_DD
        Where DD is the last day of the month (determines month length).
        
        CSV should have columns:
        - Date (1, 2, 3, ..., DD - day of month)
        - Energy [kW] (daily generation)
        - POA irradiance (daily irradiance)
        
        Args:
            filename: CSV file in data/ directory (e.g., 'Energy_generation_2024_01_31.csv')
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Parse filename to extract year, month, last_day
        # Format: Energy_generation_YYYY_MM_DD
        try:
            parts = filepath.stem.split('_')  # Remove .csv extension
            year = int(parts[-3])
            month = int(parts[-2])
            last_day = int(parts[-1])
        except (IndexError, ValueError):
            raise ValueError(
                f"Filename must follow format: Energy_generation_YYYY_MM_DD.csv\n"
                f"Got: {filepath.name}"
            )
        
        # Try multiple encodings and separators
        df = None
        skip_rows = 0
        
        for encoding in ['utf-8', 'utf-16', 'utf-16-le', 'latin1', 'iso-8859-1']:
            for sep in ['\t', ',', ';']:
                for skip in [0, 1, 2, 3]:
                    try:
                        df_temp = pd.read_csv(
                            filepath, 
                            encoding=encoding, 
                            sep=sep,
                            skiprows=skip,
                            quotechar='"'
                        )
                        # Check if we have expected columns
                        cols_lower = [c.lower().strip() for c in df_temp.columns]
                        if any('date' in c for c in cols_lower) and len(df_temp.columns) >= 2:
                            df = df_temp
                            skip_rows = skip
                            print(f"  Loaded {filename} with {encoding}/{sep}/skip={skip}")
                            break
                    except Exception:
                        continue
                if df is not None:
                    break
            if df is not None:
                break
        
        if df is None:
            raise ValueError(f"Could not read {filename} with any known encoding/separator combination")
        
        # Normalize column names (case-insensitive)
        df.columns = df.columns.str.lower().str.strip()
        print(f"    Columns found: {list(df.columns)}")
        
        # Convert numeric columns to float
        for col in df.columns:
            if col != 'date':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Build timestamps from Date column (1, 2, ..., last_day)
        if 'date' not in df.columns:
            raise ValueError(f"CSV must have 'Date' column with day numbers (1-31)\nColumns found: {list(df.columns)}")
        
        df['timestamp'] = pd.to_datetime({
            'year': year,
            'month': month,
            'day': df['date'].astype(int)
        })
        
        # Energy column - try various names
        energy_col = None
        for col in df.columns:
            if 'energy' in col.lower():
                energy_col = col
                break
        if energy_col:
            df = df.rename(columns={energy_col: 'yield'})
        elif 'yield' not in df.columns:
            raise ValueError("No 'Energy' column found")
        
        # POA irradiance column
        poa_col = None
        for col in df.columns:
            if 'poa' in col.lower() or 'irrad' in col.lower():
                poa_col = col
                break
        if poa_col:
            df = df.rename(columns={poa_col: 'ghi'})
        elif 'ghi' not in df.columns:
            raise ValueError("No 'POA irradiance' column found")
        
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        return df
    
    def load_all_csv(self, pattern: str = "Energy_generation_*.csv") -> pd.DataFrame:
        """
        Load and concatenate all CSV files matching pattern.
        
        Args:
            pattern: Glob pattern for files (default: all Energy_generation files)
        
        Returns:
            Combined DataFrame with data from all months
        """
        import glob
        files = sorted(self.data_dir.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No files matching {pattern} in {self.data_dir}")
        
        dfs = [self.load_csv(f.name) for f in files]
        combined = pd.concat(dfs).sort_index()
        combined = combined[~combined.index.duplicated(keep='first')]
        
        return combined
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data integrity - requires yield and ghi.
        """
        required_cols = ['yield', 'ghi']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Got: {list(df.columns)}")
        
        # Remove rows with NaN
        df = df.dropna()
        
        # Remove negative values (physical impossibility)
        df = df[(df['yield'] >= 0) & (df['ghi'] >= 0)]
        
        return df
    
    def get_daily_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Data is already daily aggregates - just clean it.
        """
        daily = df[['yield', 'ghi']].copy()
        
        # Convert to numeric if needed
        daily['yield'] = pd.to_numeric(daily['yield'], errors='coerce')
        daily['ghi'] = pd.to_numeric(daily['ghi'], errors='coerce')
        
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
    
    def efficiency_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute efficiency ratio: yield / ghi
        """
        df = df.copy()
        df['efficiency'] = df['yield'] / (df['ghi'] + 1e-6)  # avoid division by zero
        return df
    
    def fit_seasonal_baseline(self, df: pd.DataFrame, percentile: float = 0.75, smooth_window: int = 15):
        """
        Fit a seasonal baseline to solar performance using top-quartile efficiency per DOY.
        
        Key improvement: Instead of fitting a sinusoid to all top-quartile days globally,
        compute the top percentile *per day-of-year*, then smooth with moving average.
        This better captures true seasonal patterns without overfitting to bad days.
        
        Args:
            df: Daily data with 'yield' and 'ghi' columns (indexed by timestamp)
            percentile: Top percentile per DOY to use (default 0.75)
            smooth_window: Window for rolling smoothing (default 15 days)
        
        Returns:
            Dictionary with:
            - 'seasonal_baseline': Series of seasonal baseline efficiency per day
            - 'popt': Fitted sinusoid parameters (if used) or None
            - 'efficiency_by_doy': Series of efficiency ratio per day
            - 'method': 'moving_quantile' or 'sinusoid'
            - 'doy_quantiles': Per-DOY quantile values (for diagnostics)
        """
        # Compute efficiency ratio
        eff = df['yield'] / (df['ghi'] + 1e-6)
        doy = df.index.dayofyear.values
        
        # Step 1: Compute top percentile per day-of-year
        doy_quantiles = df.groupby(doy)['yield'].apply(
            lambda x: (x / (df.loc[x.index, 'ghi'] + 1e-6)).quantile(percentile)
        )
        
        print(f"Per-DOY quantile range: {doy_quantiles.min():.6f} - {doy_quantiles.max():.6f}")
        print(f"Seasonal variation: {(doy_quantiles.max() - doy_quantiles.min()):.6f}")
        
        # Step 2: Apply rolling smoothing to reduce noise
        doy_quantiles_smooth = doy_quantiles.rolling(
            window=smooth_window, 
            center=True, 
            min_periods=1
        ).mean()
        
        # Step 3: Interpolate to all days
        seasonal_baseline = pd.Series(index=df.index, dtype=float)
        for idx, d in enumerate(doy):
            seasonal_baseline.iloc[idx] = doy_quantiles_smooth.get(d, doy_quantiles_smooth.mean())
        
        # Step 4: Try to fit sinusoid for reference (but use moving average as primary)
        try:
            from scipy.optimize import curve_fit
            
            def seasonal_curve(x, a, b, c):
                return a + b * np.sin((2 * np.pi / 365) * x + c)
            
            x_fit = doy_quantiles_smooth.index.values
            y_fit = doy_quantiles_smooth.values
            
            popt, _ = curve_fit(
                seasonal_curve,
                x_fit,
                y_fit,
                p0=[eff.mean(), 0.001, 0],
                maxfev=5000
            )
            
            # Make amplitude absolute for clarity
            popt[1] = abs(popt[1])
            
            print(f"\nSinusoid fit (reference only):")
            print(f"  Baseline (a): {popt[0]:.6f}")
            print(f"  Amplitude (|b|): {popt[1]:.6f}")
            print(f"  Phase (c): {popt[2]:.4f}")
            
        except Exception as e:
            print(f"Warning: Could not fit sinusoid: {e}")
            popt = None
        
        # Identify healthy days based on seasonal baseline
        # A day is "healthy" if actual efficiency >= seasonal baseline - tolerance
        tolerance = 0.005  # ±0.5%
        healthy_mask = eff >= (seasonal_baseline - tolerance)
        
        return {
            'seasonal_baseline': seasonal_baseline,
            'popt': popt,
            'efficiency_by_doy': eff,
            'doy_quantiles': doy_quantiles,
            'doy_quantiles_smooth': doy_quantiles_smooth,
            'healthy_mask': healthy_mask,
            'method': 'moving_quantile',
            'tolerance': tolerance
        }
