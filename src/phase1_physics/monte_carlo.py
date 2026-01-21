"""
Monte Carlo simulator for solar irradiance and power generation.
Calibrates to historical healthy days to establish performance envelope.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy import stats


class MonteCarloSimulator:
    """
    Simulates expected power generation under various conditions.
    Calibrated using healthy days as the baseline.
    """
    
    def __init__(self, n_simulations: int = 1000, random_seed: int = 42):
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Calibration parameters (to be set during calibration)
        self.efficiency_mean = None
        self.efficiency_std = None
        self.ghi_noise_std = None
    
    def calibrate_from_healthy_days(
        self, 
        df: pd.DataFrame, 
        healthy_mask: np.ndarray
    ) -> None:
        """
        Calibrate Monte Carlo parameters using known healthy days.
        
        Args:
            df: DataFrame with 'yield' and 'ghi' columns
            healthy_mask: Boolean array indicating healthy days
        """
        healthy_data = df[healthy_mask].copy()
        
        # Calculate efficiency = yield / ghi (normalized power output)
        efficiency = healthy_data['yield'] / (healthy_data['ghi'] + 1e-6)
        
        self.efficiency_mean = efficiency.mean()
        self.efficiency_std = efficiency.std()
        
        # Estimate noise in GHI measurements
        self.ghi_noise_std = healthy_data['ghi'].std() * 0.05  # 5% measurement noise
        
        print(f"Calibration complete:")
        print(f"  Mean efficiency: {self.efficiency_mean:.4f}")
        print(f"  Efficiency std: {self.efficiency_std:.4f}")
        print(f"  GHI noise std: {self.ghi_noise_std:.4f}")
    
    def simulate_day(
        self, 
        ghi: np.ndarray,
        efficiency_override: Optional[float] = None
    ) -> np.ndarray:
        """
        Run Monte Carlo simulations for a single day.
        
        Args:
            ghi: Array of GHI values for the day
            efficiency_override: Optional fixed efficiency (for debugging)
        
        Returns:
            Array of shape (n_simulations, n_timepoints) with simulated yields
        """
        if self.efficiency_mean is None:
            raise ValueError("Must calibrate first with calibrate_from_healthy_days()")
        
        n_timepoints = len(ghi)
        simulations = np.zeros((self.n_simulations, n_timepoints))
        
        for i in range(self.n_simulations):
            # Sample efficiency from calibration distribution
            if efficiency_override is not None:
                eff = efficiency_override
            else:
                eff = np.random.normal(
                    self.efficiency_mean, 
                    self.efficiency_std
                )
                eff = np.clip(eff, 0, 1)  # Physically valid range
            
            # Add measurement noise to GHI
            ghi_noisy = ghi + np.random.normal(0, self.ghi_noise_std, size=n_timepoints)
            ghi_noisy = np.maximum(ghi_noisy, 0)  # GHI can't be negative
            
            # Calculate expected yield
            simulations[i, :] = eff * ghi_noisy
        
        return simulations
    
    def get_percentiles(
        self, 
        simulations: np.ndarray,
        percentiles: list = [10, 50, 90]
    ) -> Dict[str, np.ndarray]:
        """
        Extract percentile envelopes from simulations.
        
        Returns:
            Dictionary with keys like 'p10', 'p50', 'p90'
        """
        result = {}
        for p in percentiles:
            result[f'p{p}'] = np.percentile(simulations, p, axis=0)
        return result
    
    def calculate_day_statistics(
        self, 
        actual_yield: np.ndarray,
        simulations: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate how the actual day compares to MC simulations.
        
        Returns:
            Dictionary with percentile rank, z-score, etc.
        """
        daily_actual = actual_yield.sum()
        daily_sims = simulations.sum(axis=1)
        
        # What percentile is the actual day?
        percentile_rank = stats.percentileofscore(daily_sims, daily_actual)
        
        # Z-score of actual vs simulations
        z_score = (daily_actual - daily_sims.mean()) / (daily_sims.std() + 1e-6)
        
        return {
            'percentile_rank': percentile_rank,
            'z_score': z_score,
            'mc_mean': daily_sims.mean(),
            'mc_std': daily_sims.std(),
            'actual': daily_actual,
            'actual_vs_mean_pct': (daily_actual / daily_sims.mean() - 1) * 100
        }
