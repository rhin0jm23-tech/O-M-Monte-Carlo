# Solar Anomaly Detection - Monte Carlo Based Single-Site Prototype

A 5-phase workflow for detecting solar generation anomalies (clouds, soiling, curtailment) using physics-based Monte Carlo simulation and machine learning.

## Project Overview

This system classifies daily solar performance into categories:
- **Healthy**: Normal generation matching expectations
- **Cloudy**: Reduced generation due to cloud cover
- **Soiling**: Gradual performance degradation  
- **Curtailment**: Intentional power limitation
- **Investigate**: Anomalous patterns requiring manual review

## Architecture

```
src/
├── data_loader.py              # Load and preprocess solar data
├── phase1_physics/             # Monte Carlo calibration
│   └── monte_carlo.py
├── phase2_features/            # Feature engineering
│   └── feature_engineering.py
├── phase3_labeling/            # Manual annotation tools
│   └── labeler.py
├── phase4_model/               # Random Forest classifier
│   └── rf_classifier.py
└── phase5_production/          # Daily classification & logging
    └── daily_classifier.py

notebooks/
├── 01_phase1_physics_baseline.ipynb       # MC calibration & visualization
├── 02_phase2_feature_engineering.ipynb    # Feature computation & analysis
├── 03_phase3_manual_labeling.ipynb        # Interactive labeling workflow
├── 04_phase4_rf_training.ipynb            # Model training & evaluation
└── 05_phase5_production_testing.ipynb     # Daily pipeline simulation

data/
├── solar_data.csv              # Input: Raw time-series data
├── daily_data.csv              # Daily aggregates
├── mc_results.csv              # MC simulation outputs
├── engineered_features.csv     # Computed features
├── labeled_days.csv            # Manual labels (Phase 3)
└── daily_logs/                 # Production classification logs
```

## 5-Phase Workflow

### Phase 1: Physics Baseline (Monte Carlo)
- **Input**: 1-2 years historical data (yield, GHI, temperature)
- **Process**: 
  - Identify "healthy" days (high efficiency, smooth generation)
  - Calibrate Monte Carlo to healthy days
  - Generate MC envelope (P10, P50, P90) for all days
- **Output**: MC percentile and z-score for each day
- **Notebook**: `01_phase1_physics_baseline.ipynb`

**Sanity Checks**:
- ✓ Cloudy days fall in lower tail of MC distribution?
- ✓ Healthy days cluster near P50 (50th percentile)?

### Phase 2: Feature Engineering
- **Input**: Daily data + MC simulations
- **Process**: Compute 8 features per day:
  - **Shape-based**: efficiency ratio, solar noon ratio, asymmetry, variability
  - **MC-derived**: percentile rank, z-score, uncertainty width
- **Output**: Feature matrix for all days
- **Notebook**: `02_phase2_feature_engineering.ipynb`

**Sanity Checks**:
- ✓ Features visually separate cloudy vs clear days?
- ✓ Feature importance via Cohen's d effect sizes?

### Phase 3: Manual Labeling
- **Input**: Feature matrix + visualization tools
- **Process**: 
  - Select 50-100 representative days via clustering
  - Interactively label each day
  - Build gold standard validation set
- **Output**: `data/labeled_days.csv` with 5 labels per day
- **Notebook**: `03_phase3_manual_labeling.ipynb`

**Labels**:
1. `Healthy` - Normal operation, no issues
2. `Cloudy` - Cloud cover reducing generation
3. `Curtailment` - Intentional power limitation
4. `Soiling` - Performance loss from dust/debris
5. `Investigate` - Uncertain or mixed signals

### Phase 4: Train Random Forest
- **Input**: Labeled data + features
- **Process**:
  - Time-based train/test split
  - Train RF with 2-3 classes (start simple)
  - Evaluate confidence calibration
- **Output**: Trained model saved to `models/anomaly_rf.pkl`
- **Notebook**: `04_phase4_rf_training.ipynb`

**Metrics**:
- Overall accuracy
- Per-class precision/recall
- High-confidence accuracy (predictions with >70% confidence)
- Confusion matrix

### Phase 5: Daily Production Testing
- **Input**: Yesterday's solar data
- **Process**: 
  - Each morning: classify yesterday
  - Log prediction, confidence, MC percentile, features
  - Weekly: manually review 5-10 "Investigate" days
- **Output**: `data/daily_logs/classifications.jsonl`
- **Notebook**: `05_phase5_production_testing.ipynb`

**Key Question**: Are you catching real issues or just flagging noise?

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data
Create a CSV file `data/solar_data.csv` with columns:
```
timestamp,yield,ghi,temp
2023-01-01 08:00:00,12.5,150.3,18.2
2023-01-01 09:00:00,45.2,480.1,19.1
...
```

### 2. Run Phase 1 (Physics Baseline)
```bash
jupyter notebook notebooks/01_phase1_physics_baseline.ipynb
```
- Loads data, identifies healthy days, calibrates MC

### 3. Run Phase 2 (Feature Engineering)
```bash
jupyter notebook notebooks/02_phase2_feature_engineering.ipynb
```
- Computes 8 features, visualizes separation

### 4. Run Phase 3 (Manual Labeling)
```bash
jupyter notebook notebooks/03_phase3_manual_labeling.ipynb
```
- Select and label 50-100 representative days

### 5. Run Phase 4 (Model Training)
```bash
jupyter notebook notebooks/04_phase4_rf_training.ipynb
```
- Trains Random Forest, evaluates performance

### 6. Run Phase 5 (Production)
```bash
jupyter notebook notebooks/05_phase5_production_testing.ipynb
```
- Daily classification and logging

## Data Format

### Input CSV (hourly or sub-hourly)
```csv
timestamp,yield,ghi,temp
2023-01-01 08:00:00,5.2,100.5,15.3
2023-01-01 09:00:00,25.8,480.2,16.1
```

Fields:
- `timestamp`: ISO 8601 datetime
- `yield`: AC power generation (kW or normalized 0-1)
- `ghi`: Global Horizontal Irradiance (W/m²)
- `temp`: Ambient temperature (°C)

### Output Logs (JSON Lines)
Each line is a JSON record:
```json
{
  "timestamp": "2024-01-21T08:15:00",
  "date_classified": "2024-01-20",
  "prediction": "Healthy",
  "confidence": 0.92,
  "mc_percentile": 58.3,
  "features": {
    "yield_to_ghi_ratio": 0.148,
    "mc_z_score": 0.42
  },
  "notes": ""
}
```

## Key Classes

### `SolarDataLoader`
Load and preprocess raw solar data.
```python
from src.data_loader import SolarDataLoader

loader = SolarDataLoader('data')
df = loader.load_csv('solar_data.csv')
df = loader.validate_data(df)
df_daily = loader.get_daily_aggregates(df)
```

### `MonteCarloSimulator`
Generate MC envelope of expected generation.
```python
from src.phase1_physics.monte_carlo import MonteCarloSimulator

mc = MonteCarloSimulator(n_simulations=1000)
mc.calibrate_from_healthy_days(df_daily, healthy_mask)
sims = mc.simulate_day(ghi_array)
stats = mc.calculate_day_statistics(actual_yield, sims)
```

### `FeatureEngineer`
Compute shape and MC-derived features.
```python
from src.phase2_features.feature_engineering import FeatureEngineer

fe = FeatureEngineer()
features = fe.compute_day_features(yield_day, ghi_day, mc_sims)
```

### `SolarDayLabeler`
Interactive tool for manual day labeling.
```python
from src.phase3_labeling.labeler import SolarDayLabeler

labeler = SolarDayLabeler()
labeler.add_label(pd.Timestamp('2023-06-15'), 'Cloudy', 'Heavy cloud cover all day')
labeler.save_labels()
```

### `SolarAnomalyRF`
Random Forest classifier for anomaly detection.
```python
from src.phase4_model.rf_classifier import SolarAnomalyRF

model = SolarAnomalyRF(n_trees=100, max_depth=10)
model.fit(X_train, y_train)
predictions, confidences = model.predict_with_confidence(X_test)
model.save_model('models/anomaly_rf.pkl')
```

### `DailyClassificationLogger`
Log and monitor daily classifications.
```python
from src.phase5_production.daily_classifier import DailyClassificationLogger

logger = DailyClassificationLogger()
logger.log_classification(date, prediction, confidence, mc_percentile, features)
summary = logger.get_alert_summary(days=7)
```

## Validation Criteria

### Phase 1: Physics Baseline
- [ ] MC calibrated successfully on healthy days
- [ ] MC P50 closely matches healthy day median generation
- [ ] Cloudy days statistically in lower MC percentiles
- [ ] Healthy days concentrated near P50

### Phase 2: Feature Engineering
- [ ] 8 features computed for all days
- [ ] Visual separation between cloudy and clear days
- [ ] Cohen's d > 0.5 for top features
- [ ] No strongly correlated features (< 0.8 Pearson)

### Phase 3: Manual Labeling
- [ ] 50-100 days labeled across 5 classes
- [ ] Reasonably balanced classes (no single class > 70%)
- [ ] Label agreement on review (if multiple raters)
- [ ] Labels saved in `data/labeled_days.csv`

### Phase 4: Model Training
- [ ] Train/test split by time (no data leakage)
- [ ] Overall accuracy > 70%
- [ ] High-confidence predictions > 80% accuracy
- [ ] Precision > 0.7 for "Investigate" class

### Phase 5: Production
- [ ] Daily classification runs without errors
- [ ] Logs accumulate in `data/daily_logs/`
- [ ] Weekly manual review identifies real issues
- [ ] Alert rate < 20% (to avoid alert fatigue)

## Common Issues & Troubleshooting

### MC Envelope Not Reasonable
- **Issue**: P50 too high/low, wide range
- **Check**: Is healthy day identification correct? Try top 50% of efficiency instead of 75%
- **Fix**: Review scatter plot of efficiency vs time, manually select healthy period

### Poor Feature Separation
- **Issue**: Features don't separate cloudy from clear days
- **Check**: Are your "healthy" vs "unhealthy" days actually different?
- **Fix**: Visualize with `df_features.plot()`, check for missing data patterns

### Model Overfits
- **Issue**: High train accuracy, low test accuracy
- **Check**: How many labeled days do you have? 
- **Fix**: Reduce `max_depth` to 8, increase `min_samples_leaf` to 10

### High False Positive Rate
- **Issue**: Too many "Investigate" predictions
- **Check**: Are "Investigate" examples clear in training data?
- **Fix**: Make "Investigate" class examples more extreme, or combine with "Healthy"

## Next Steps

1. **Collect Site-Specific Data**: Replace synthetic data with real solar telemetry
2. **Refine Feature Set**: Add time-of-year features, inverter-specific metrics
3. **Expand Classes**: Add "Inverter fault", "Transmission loss" once core system works
4. **Integrate Alerts**: Email/SMS alerts for high-confidence "Investigate" days
5. **Continuous Learning**: Retrain model weekly with new manual labels
6. **A/B Testing**: Compare MC-based approach to simple threshold-based rules

## References

- Monte Carlo simulation for solar: [PV Wavelet paper]
- Feature engineering for anomaly detection: [Solar nowcasting literature]
- Random Forest calibration: [Guo et al. 2017 on calibration curves]

## License

[Specify your license here]

## Contact

[Your contact info]
