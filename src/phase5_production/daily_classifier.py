"""
Daily classification system for production use.
Classifies yesterday's data each morning, logs results, tracks issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
import json


class DailyClassificationLogger:
    """
    Logs daily classifications for monitoring and analysis.
    """
    
    def __init__(self, log_dir: str = "data/daily_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "classifications.jsonl"
    
    def log_classification(
        self,
        date: pd.Timestamp,
        prediction: str,
        confidence: float,
        mc_percentile: float,
        features: Dict[str, float],
        notes: str = ""
    ) -> None:
        """
        Log a daily classification result.
        
        Args:
            date: Date of classification
            prediction: Predicted class
            confidence: Confidence score (0-1)
            mc_percentile: MC percentile rank
            features: Dictionary of computed features
            notes: Optional notes
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'date_classified': date.isoformat(),
            'prediction': prediction,
            'confidence': float(confidence),
            'mc_percentile': float(mc_percentile),
            'features': {k: float(v) for k, v in features.items()},
            'notes': notes
        }
        
        # Append to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
    
    def load_recent_logs(self, days: int = 30) -> pd.DataFrame:
        """Load classification logs from last N days."""
        if not self.log_file.exists():
            return pd.DataFrame()
        
        logs = []
        with open(self.log_file, 'r') as f:
            for line in f:
                logs.append(json.loads(line))
        
        df = pd.DataFrame(logs)
        df['date_classified'] = pd.to_datetime(df['date_classified'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter to recent
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df['timestamp'] > cutoff]
        
        return df.sort_values('date_classified')
    
    def get_alert_summary(self, days: int = 7) -> Dict:
        """
        Summarize alerts from the last N days.
        """
        logs = self.load_recent_logs(days=days)
        
        if len(logs) == 0:
            return {'total_days': 0, 'alerts': []}
        
        # Find investigate/cloudy/soiling days
        alert_classes = ['Investigate', 'Cloudy', 'Soiling', 'Curtailment']
        alerts = logs[logs['prediction'].isin(alert_classes)].copy()
        
        return {
            'total_days_classified': len(logs),
            'alert_count': len(alerts),
            'alert_rate': len(alerts) / max(len(logs), 1),
            'alerts_by_class': alerts['prediction'].value_counts().to_dict(),
            'low_confidence_alerts': len(alerts[alerts['confidence'] < 0.7]),
            'recent_alerts': alerts.tail(5).to_dict('records')
        }


class DailyProductionPipeline:
    """
    Complete daily classification pipeline for production.
    """
    
    def __init__(self, model_path: str, data_loader, feature_engineer, mc_simulator):
        """
        Initialize production pipeline.
        
        Args:
            model_path: Path to trained RF model
            data_loader: SolarDataLoader instance
            feature_engineer: FeatureEngineer instance
            mc_simulator: MonteCarloSimulator instance
        """
        from phase4_model.rf_classifier import SolarAnomalyRF
        
        self.model = SolarAnomalyRF.load_model(model_path)
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.mc_simulator = mc_simulator
        self.logger = DailyClassificationLogger()
    
    def classify_yesterday(self) -> Dict:
        """
        Classify yesterday's performance.
        Runs each morning.
        
        Returns:
            Dictionary with classification results
        """
        yesterday = pd.Timestamp.now().normalize() - timedelta(days=1)
        
        # In production, you'd load yesterday's actual data
        # For now, this is a template
        
        result = {
            'date': yesterday,
            'status': 'ready_for_implementation',
            'note': 'Integrate with real data source'
        }
        
        return result
    
    def run_weekly_manual_check(self, n_samples: int = 10) -> Dict:
        """
        Weekly task: manually review 5-10 recent 'Investigate' classifications.
        
        Args:
            n_samples: Number of recent investigate days to surface
        
        Returns:
            List of recent investigate days for manual review
        """
        logs = self.logger.load_recent_logs(days=7)
        investigate = logs[logs['prediction'] == 'Investigate'].tail(n_samples)
        
        return {
            'review_date': datetime.now().isoformat(),
            'days_to_review': len(investigate),
            'investigate_cases': investigate.to_dict('records'),
            'action_items': [
                "Check each day visually for actual issues",
                "Validate MC calibration if systematic misses",
                "Update training labels if new patterns found"
            ]
        }
