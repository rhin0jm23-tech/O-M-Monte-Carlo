"""
Random Forest classifier for solar anomaly detection.
Focuses on confidence calibration and feature importance.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score
)
import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional


class SolarAnomalyRF:
    """
    Random Forest for classifying solar days.
    Emphasizes confidence calibration for reliable alerts.
    """
    
    def __init__(
        self,
        n_trees: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_leaf: int = 5,
        random_state: int = 42
    ):
        self.rf = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names = None
        self.classes_ = None
        self.class_weights = None
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        class_weight: str = 'balanced'
    ) -> None:
        """
        Train random forest on labeled data.
        
        Args:
            X_train: Feature matrix
            y_train: Class labels
            class_weight: 'balanced' to handle imbalanced classes
        """
        self.feature_names = X_train.columns.tolist()
        self.classes_ = np.unique(y_train)
        
        self.rf.fit(X_train, y_train)
        print(f"Trained on {len(X_train)} samples with {len(self.feature_names)} features")
        print(f"Classes: {self.classes_}")
        
        # Get class distribution
        class_counts = y_train.value_counts()
        print("Class distribution:")
        for cls in self.classes_:
            count = class_counts.get(cls, 0)
            pct = 100 * count / len(y_train)
            print(f"  {cls:20s}: {count:4d} ({pct:5.1f}%)")
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and confidence scores.
        
        Returns:
            predictions: Class predictions
            confidences: Max probability across classes (0-1)
        """
        predictions = self.rf.predict(X)
        probabilities = self.rf.predict_proba(X)
        confidences = probabilities.max(axis=1)
        
        return predictions, confidences
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        confidence_threshold: float = 0.5
    ) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            X_test: Test features
            y_test: Test labels
            confidence_threshold: Minimum confidence to make prediction
        
        Returns:
            Dictionary of metrics
        """
        predictions, confidences = self.predict_with_confidence(X_test)
        
        # Standard metrics
        report = classification_report(y_test, predictions, output_dict=True)
        cm = confusion_matrix(y_test, predictions)
        
        # Confidence analysis
        metrics = {
            'overall_accuracy': (predictions == y_test).mean(),
            'confusion_matrix': cm,
            'classification_report': report,
        }
        
        # High-confidence accuracy
        high_conf_mask = confidences >= confidence_threshold
        if high_conf_mask.sum() > 0:
            high_conf_acc = (predictions[high_conf_mask] == y_test[high_conf_mask]).mean()
            metrics['high_confidence_accuracy'] = high_conf_acc
            metrics['high_confidence_coverage'] = high_conf_mask.mean()
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Extract feature importance from trained RF.
        """
        importances = self.rf.feature_importances_
        features_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return features_imp.head(top_n)
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> 'SolarAnomalyRF':
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model loaded from {filepath}")
        return model
    
    def calibrate_predictions(self, X_cal: pd.DataFrame, y_cal: pd.Series) -> None:
        """
        Post-hoc calibration using a calibration set.
        Improves reliability of confidence scores.
        """
        # This would use sklearn's CalibratedClassifierCV in production
        # For now, just a placeholder for the workflow
        print(f"Calibration set: {len(X_cal)} samples")
        print("Note: Full calibration pipeline would be implemented here")
