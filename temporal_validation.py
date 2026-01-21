#!/usr/bin/env python
"""
Temporal Validation Analysis
Validates Random Forest model generalization using time-based and seasonal holdout strategies.
"""

import sys
sys.path.insert(0, '/workspaces/O-M-Monte-Carlo')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
df_features = pd.read_csv('data/engineered_features.csv', index_col=0)
df_labels = pd.read_csv('data/labeled_days.csv', index_col=0)

# Simplify labels
label_map = {
    'Cloudy': 'Cloudy',
    'Healthy': 'Healthy',
    'Soiling': 'Investigate',
    'Investigate': 'Investigate'
}
df_labels['simplified_label'] = df_labels['manual_label'].map(label_map)

# Merge
df_train = df_features.merge(df_labels[['simplified_label']], left_index=True, right_index=True, how='inner')
feature_cols = [col for col in df_train.columns if col not in ['simplified_label', 'healthy', 'cluster']]

X = df_train[feature_cols]
y = df_train['simplified_label']

print("="*80)
print("TIME-BASED HOLDOUT VALIDATION")
print("="*80)

# Strategy 1: 60/40 Time-based split
print("\n1. TEMPORAL SPLIT (First 60% train, Last 40% test)")
print("-" * 80)

split_idx = int(len(df_train) * 0.6)
X_train_temporal = X.iloc[:split_idx]
X_test_temporal = X.iloc[split_idx:]
y_train_temporal = y.iloc[:split_idx]
y_test_temporal = y.iloc[split_idx:]

print(f"Train dates: {X_train_temporal.index[0]} to {X_train_temporal.index[-1]} ({len(X_train_temporal)} days)")
print(f"Test dates:  {X_test_temporal.index[0]} to {X_test_temporal.index[-1]} ({len(X_test_temporal)} days)")

rf_temporal = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temporal.fit(X_train_temporal, y_train_temporal)

y_pred_temporal = rf_temporal.predict(X_test_temporal)
acc_temporal = accuracy_score(y_test_temporal, y_pred_temporal)

train_acc = accuracy_score(y_train_temporal, rf_temporal.predict(X_train_temporal))

print(f"\nTraining Accuracy: {train_acc:.3f}")
print(f"Test Accuracy:     {acc_temporal:.3f}")
print(f"Generalization Gap: {train_acc - acc_temporal:.3f}")
print(f"\nTest Set Classification Report:")
print(classification_report(y_test_temporal, y_pred_temporal))

# Strategy 2: Leave-one-season-out
print("\n" + "="*80)
print("2. LEAVE-ONE-SEASON-OUT VALIDATION")
print("-" * 80)

# Convert index to datetime and extract seasons
dates = pd.to_datetime(X.index)
X_with_dates = X.copy()
X_with_dates['month'] = dates.month

# Define seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

seasons = X_with_dates['month'].apply(get_season)

results_by_season = {}

for test_season in ['Winter', 'Spring', 'Summer', 'Fall']:
    train_mask = (seasons != test_season)
    test_mask = (seasons == test_season)
    
    if test_mask.sum() == 0:
        continue
    
    X_train_season = X[train_mask]
    X_test_season = X[test_mask]
    y_train_season = y[train_mask]
    y_test_season = y[test_mask]
    
    print(f"\nLOO-{test_season.upper()}:")
    print(f"  Train: {len(X_train_season)} days (all seasons except {test_season})")
    print(f"  Test:  {len(X_test_season)} days ({test_season} only)")
    
    rf_season = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_season.fit(X_train_season, y_train_season)
    
    y_pred_season = rf_season.predict(X_test_season)
    acc_season = accuracy_score(y_test_season, y_pred_season)
    
    train_acc_season = accuracy_score(y_train_season, rf_season.predict(X_train_season))
    
    results_by_season[test_season] = {
        'train_acc': train_acc_season,
        'test_acc': acc_season,
        'n_test': len(X_test_season)
    }
    
    print(f"  Train Accuracy: {train_acc_season:.3f}")
    print(f"  Test Accuracy:  {acc_season:.3f}")
    print(f"  Gap:            {train_acc_season - acc_season:.3f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nTemporal Split (60/40):")
print(f"  Train Acc: {train_acc:.3f}")
print(f"  Test Acc:  {acc_temporal:.3f}")
print(f"  Gap:       {train_acc - acc_temporal:.3f}")

print(f"\nLeave-One-Season-Out Results:")
for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    if season in results_by_season:
        results = results_by_season[season]
        gap = results['train_acc'] - results['test_acc']
        status = "⚠️  COLLAPSE" if gap > 0.2 else "✓ STABLE" if gap < 0.1 else "◐ MODERATE"
        print(f"  {season:8s}: Train={results['train_acc']:.3f}, Test={results['test_acc']:.3f}, Gap={gap:.3f} {status}")
