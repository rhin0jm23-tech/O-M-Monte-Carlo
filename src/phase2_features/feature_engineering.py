"""
Feature engineering for anomaly detection.
Computes shape-based and Monte Carlo-derived features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from scipy import signal


class FeatureEngineer:
    """Compute features for anomaly detection."""
    
    @staticmethod
    def yield_to_ghi_ratio(yield_series: np.ndarray, ghi_series: np.ndarray) -> float:
        """
        Efficiency metric: total yield / total GHI.
        Lower values indicate underperformance (clouds, soiling, etc.)
        """
        ghi_sum = ghi_series.sum()
        if ghi_sum == 0:
            return 0
        return yield_series.sum() / ghi_sum
    
    @staticmethod
    def solar_noon_ratio(yield_series: np.ndarray, ghi_series: np.ndarray) -> float:
        """
        Peak performance ratio.
        Calculate yield around solar noon (peak 3-4 hours) vs full day.
        Cloudy days have flatter profiles.
        """
        n = len(yield_series)
        peak_window = int(n * 0.25)  # Middle 25% of day (roughly 3 hours)
        start_idx = max(0, n // 2 - peak_window // 2)
        end_idx = min(n, n // 2 + peak_window // 2)
        
        peak_yield = yield_series[start_idx:end_idx].sum()
        total_yield = yield_series.sum()
        
        if total_yield == 0:
            return 0
        
        return peak_yield / total_yield
    
    @staticmethod
    def morning_afternoon_asymmetry(yield_series: np.ndarray) -> float:
        """
        Asymmetry in morning vs afternoon generation.
        Clear days: symmetric, near 1.0
        Soiling or clouds: skewed toward morning or afternoon
        """
        n = len(yield_series)
        mid = n // 2
        
        morning = yield_series[:mid].sum()
        afternoon = yield_series[mid:].sum()
        
        total = morning + afternoon
        if total == 0:
            return 0
        
        # Ratio will be near 1.0 for symmetric days
        return morning / (afternoon + 1e-6)
    
    @staticmethod
    def yield_variability(yield_series: np.ndarray) -> float:
        """
        Coefficient of variation in yield across the day.
        Cloudy days: high variability (rapid changes)
        Clear days: smoother progression
        """
        if yield_series.mean() == 0:
            return 0
        return yield_series.std() / (yield_series.mean() + 1e-6)
    
    @staticmethod
    def early_morning_ramp(yield_series: np.ndarray) -> float:
        """
        Rate of change in early morning (first 2 hours).
        Helps detect if day starts normally or with problems.
        """
        early_hours = int(len(yield_series) * 0.15)  # First ~2 hours
        if early_hours < 2:
            return 0
        
        early_segment = yield_series[:early_hours]
        if early_segment.max() == 0:
            return 0
        
        # Return average slope
        slope = np.gradient(early_segment).mean()
        return slope / (early_segment.max() + 1e-6)
    
    @staticmethod
    def mc_percentile_feature(actual_daily: float, mc_sims: np.ndarray) -> float:
        """
        What percentile of the MC distribution is the actual day?
        0-50: Below normal (suspect)
        50-100: Normal
        """
        daily_sims = mc_sims.sum(axis=1) if len(mc_sims.shape) > 1 else mc_sims
        percentile = (daily_sims < actual_daily).mean() * 100
        return percentile
    
    @staticmethod
    def mc_z_score_feature(actual_daily: float, mc_sims: np.ndarray) -> float:
        """
        Z-score: how many standard deviations from MC mean.
        Positive: outperforming expectations
        Negative: underperforming expectations
        """
        daily_sims = mc_sims.sum(axis=1) if len(mc_sims.shape) > 1 else mc_sims
        mean = daily_sims.mean()
        std = daily_sims.std()
        
        if std == 0:
            return 0
        
        return (actual_daily - mean) / std
    
    @staticmethod
    def mc_uncertainty_width(mc_sims: np.ndarray) -> float:
        """
        Width of MC envelope: P90 - P10.
        Larger = more uncertainty in day's outcome.
        """
        daily_sims = mc_sims.sum(axis=1) if len(mc_sims.shape) > 1 else mc_sims
        p90 = np.percentile(daily_sims, 90)
        p10 = np.percentile(daily_sims, 10)
        return p90 - p10
    
    def compute_day_features(
        self,
        yield_day: np.ndarray,
        ghi_day: np.ndarray,
        mc_simulations: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all features for a single day.
        
        Args:
            yield_day: Hourly/sub-hourly yield for the day
            ghi_day: Corresponding GHI values
            mc_simulations: Monte Carlo simulation array (n_sims, n_timepoints)
        
        Returns:
            Dictionary of feature names to values
        """
        actual_daily = yield_day.sum()
        
        features = {
            # Shape-based features
            'yield_to_ghi_ratio': self.yield_to_ghi_ratio(yield_day, ghi_day),
            'solar_noon_ratio': self.solar_noon_ratio(yield_day, ghi_day),
            'morning_afternoon_asymmetry': self.morning_afternoon_asymmetry(yield_day),
            'yield_variability': self.yield_variability(yield_day),
            'early_morning_ramp': self.early_morning_ramp(yield_day),
            
            # MC-derived features
            'mc_percentile': self.mc_percentile_feature(actual_daily, mc_simulations),
            'mc_z_score': self.mc_z_score_feature(actual_daily, mc_simulations),
            'mc_uncertainty_width': self.mc_uncertainty_width(mc_simulations),
        }
        
        return features
    
    def compute_batch_features(
        self,
        df_daily: pd.DataFrame,
        mc_simulations_list: List[np.ndarray]
    ) -> pd.DataFrame:
        """
        Compute features for multiple days.
        
        Args:
            df_daily: Daily data with 'yield' and 'ghi' columns
            mc_simulations_list: List of MC simulation arrays, one per day
        
        Returns:
            DataFrame with one row per day and feature columns
        """
        all_features = []
        
        for idx, (date, row) in enumerate(df_daily.iterrows()):
            # Reconstruct daily arrays (in real usage, you'd have sub-daily data)
            yield_day = np.array([row['yield']])
            ghi_day = np.array([row['ghi']])
            
            if idx < len(mc_simulations_list):
                mc_sims = mc_simulations_list[idx]
            else:
                mc_sims = np.zeros((1000, 1))  # Fallback
            
            features = self.compute_day_features(yield_day, ghi_day, mc_sims)
            features['date'] = date
            all_features.append(features)
        
        return pd.DataFrame(all_features).set_index('date')
