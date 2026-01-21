"""
Interactive labeling utilities for creating gold standard dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict


class SolarDayLabeler:
    """
    Tool for manually labeling solar days.
    Labels: Healthy, Cloudy, Curtailment, Soiling, Investigate
    """
    
    VALID_LABELS = [
        'Healthy',
        'Cloudy', 
        'Curtailment',
        'Soiling',
        'Investigate'
    ]
    
    def __init__(self, output_file: str = "data/labeled_days.csv"):
        self.output_file = Path(output_file)
        self.labels = []
        self.load_existing_labels()
    
    def load_existing_labels(self) -> None:
        """Load any previously saved labels."""
        if self.output_file.exists():
            self.labels = pd.read_csv(self.output_file, parse_dates=['date'])
            print(f"Loaded {len(self.labels)} existing labels from {self.output_file}")
        else:
            self.labels = pd.DataFrame(columns=['date', 'label', 'notes'])
    
    def save_labels(self) -> None:
        """Save labels to CSV."""
        if len(self.labels) > 0:
            self.labels.to_csv(self.output_file, index=False)
            print(f"Saved {len(self.labels)} labels to {self.output_file}")
    
    def add_label(
        self,
        date: pd.Timestamp,
        label: str,
        notes: str = ""
    ) -> None:
        """
        Add or update a label for a specific day.
        
        Args:
            date: Date of the day
            label: One of VALID_LABELS
            notes: Optional free-text notes
        """
        if label not in self.VALID_LABELS:
            raise ValueError(f"Invalid label: {label}. Must be one of {self.VALID_LABELS}")
        
        # Check if already labeled
        existing = self.labels[self.labels['date'] == date]
        if len(existing) > 0:
            self.labels.loc[existing.index[0], 'label'] = label
            self.labels.loc[existing.index[0], 'notes'] = notes
        else:
            new_row = pd.DataFrame({
                'date': [date],
                'label': [label],
                'notes': [notes]
            })
            self.labels = pd.concat([self.labels, new_row], ignore_index=True)
    
    def get_label_stats(self) -> Dict[str, int]:
        """Return count of each label type."""
        if len(self.labels) == 0:
            return {}
        return self.labels['label'].value_counts().to_dict()
    
    def get_unlabeled_days(
        self,
        candidate_dates: pd.DatetimeIndex,
        limit: Optional[int] = None
    ) -> pd.DatetimeIndex:
        """
        Get days from candidate set that haven't been labeled yet.
        
        Args:
            candidate_dates: Dates to check
            limit: Maximum number to return
        
        Returns:
            DatetimeIndex of unlabeled dates
        """
        labeled_dates = set(self.labels['date'].dt.date)
        unlabeled = [d for d in candidate_dates if d.date() not in labeled_dates]
        
        if limit:
            unlabeled = unlabeled[:limit]
        
        return pd.DatetimeIndex(unlabeled)
    
    def interactive_label_session(
        self,
        df_features: pd.DataFrame,
        limit: int = 50
    ) -> None:
        """
        Interactive labeling session.
        Displays features for each day and prompts for label.
        
        Args:
            df_features: DataFrame with features and visualization data
            limit: Number of days to label in this session
        """
        candidate_dates = df_features.index
        unlabeled = self.get_unlabeled_days(candidate_dates, limit=limit)
        
        print(f"\nLabeling session: {len(unlabeled)} days to label")
        print(f"Current stats: {self.get_label_stats()}")
        print("\nValid labels:")
        for i, label in enumerate(self.VALID_LABELS, 1):
            print(f"  {i}. {label}")
        
        for i, date in enumerate(unlabeled):
            print(f"\n--- Day {i+1}/{len(unlabeled)}: {date.date()} ---")
            
            # Display relevant features
            if date in df_features.index:
                day_features = df_features.loc[date]
                print("Features:")
                for feat, val in day_features.items():
                    if pd.notna(val):
                        print(f"  {feat:30s}: {val:8.3f}")
            
            # Prompt for label
            while True:
                user_input = input("\nEnter label (1-5 or name), or 'skip': ").strip()
                
                if user_input.lower() == 'skip':
                    break
                
                # Parse numeric input
                try:
                    idx = int(user_input) - 1
                    if 0 <= idx < len(self.VALID_LABELS):
                        label = self.VALID_LABELS[idx]
                    else:
                        print("Invalid number, try again")
                        continue
                except ValueError:
                    # Try string input
                    if user_input in self.VALID_LABELS:
                        label = user_input
                    else:
                        print("Invalid label, try again")
                        continue
                
                # Get optional notes
                notes = input("Notes (optional, press enter to skip): ").strip()
                
                self.add_label(date, label, notes)
                print(f"Labeled as: {label}")
                break
        
        self.save_labels()
        print(f"\nSession complete. Total labeled: {len(self.labels)}")
