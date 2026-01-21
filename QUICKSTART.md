# O-M-Monte-Carlo: Solar Anomaly Detection Prototype

## Quick Reference

### Phase 1: Physics Baseline
- Load historical data (1-2 years)
- Identify healthy days (top quartile efficiency)
- Calibrate Monte Carlo to healthy days
- Generate P10, P50, P90 envelopes
- **Verify**: Are unhealthy days in lower tail? Do healthy days cluster at P50?

### Phase 2: Feature Engineering  
- Compute 8 features per day:
  - Efficiency ratio (yield/GHI)
  - Solar noon concentration ratio
  - Morning/afternoon asymmetry
  - Yield variability (hourly changes)
  - MC percentile rank
  - MC z-score
  - MC uncertainty width (P90-P10)
- **Verify**: Visual separation between cloudy and clear?

### Phase 3: Manual Labeling
- Select 50-100 representative days via k-means clustering
- Label each day: Healthy / Cloudy / Curtailment / Soiling / Investigate
- Build gold standard validation set
- **Verify**: ~20 labels per class minimum

### Phase 4: Model Training
- Use labeled data + engineered features
- Train Random Forest (100 trees, max_depth=10)
- Time-based train/test split
- Start with 2-3 classes, expand later
- **Verify**: High-confidence predictions > 80% accurate

### Phase 5: Daily Production
- Each morning: classify yesterday
- Log prediction + confidence + MC percentile + features
- Weekly: manually review 5-10 "Investigate" cases
- **Key Q**: Catching real issues or just noise?

---

## File Structure

```
data/
  solar_data.csv              # Your raw time-series data
  daily_data.csv              # Daily aggregates (Phase 1 output)
  mc_results.csv              # MC percentiles (Phase 1 output)
  engineered_features.csv     # All 8 features (Phase 2 output)
  labeled_days.csv            # Manual labels (Phase 3 output)
  daily_logs/
    classifications.jsonl     # Daily predictions (Phase 5 output)

src/
  data_loader.py              # SolarDataLoader class
  phase1_physics/
    monte_carlo.py            # MonteCarloSimulator class
  phase2_features/
    feature_engineering.py    # FeatureEngineer class
  phase3_labeling/
    labeler.py                # SolarDayLabeler class
  phase4_model/
    rf_classifier.py          # SolarAnomalyRF class
  phase5_production/
    daily_classifier.py       # DailyClassificationLogger, DailyProductionPipeline

notebooks/
  01_phase1_physics_baseline.ipynb       # MC calibration
  02_phase2_feature_engineering.ipynb    # Feature computation
  03_phase3_manual_labeling.ipynb        # Interactive labeling
  04_phase4_rf_training.ipynb            # Model training
  05_phase5_production_testing.ipynb     # Daily pipeline

models/
  anomaly_rf.pkl              # Trained Random Forest (Phase 4 output)
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your data: data/solar_data.csv
#    Columns: timestamp, yield, ghi, temp

# 3. Run each phase sequentially
jupyter notebook notebooks/01_phase1_physics_baseline.ipynb
jupyter notebook notebooks/02_phase2_feature_engineering.ipynb
jupyter notebook notebooks/03_phase3_manual_labeling.ipynb
jupyter notebook notebooks/04_phase4_rf_training.ipynb
jupyter notebook notebooks/05_phase5_production_testing.ipynb
```

---

## Example: Quick Integration

```python
from src.data_loader import SolarDataLoader
from src.phase1_physics.monte_carlo import MonteCarloSimulator
from src.phase2_features.feature_engineering import FeatureEngineer
from src.phase4_model.rf_classifier import SolarAnomalyRF

# Load data
loader = SolarDataLoader('data')
df = loader.load_csv('solar_data.csv')
df_daily = loader.get_daily_aggregates(df)

# MC calibration
mc = MonteCarloSimulator()
healthy_mask = (df_daily['yield'] / df_daily['ghi']) > 0.14
mc.calibrate_from_healthy_days(df_daily, healthy_mask)

# Features
fe = FeatureEngineer()
features = fe.compute_batch_features(df_daily, [mc.simulate_day(...) for ...])

# Predict
model = SolarAnomalyRF.load_model('models/anomaly_rf.pkl')
pred, conf = model.predict_with_confidence(features)
```

---

## Debugging Tips

### Features look weird?
- Check for NaN values: `df.isnull().sum()`
- Plot time series: `df.plot()`
- Compare healthy vs unhealthy: `df[healthy_mask].mean()`

### Model underperforms?
- Check training set size: need ~50 labeled days minimum
- Review confusion matrix: which classes are confused?
- Check feature correlation: high correlation = redundancy

### Too many false alarms?
- Increase confidence threshold from 0.5 to 0.7
- Check alert rate: should be < 20%
- Review recent "Investigate" cases manually

---

## Key Concepts

### Monte Carlo Envelope
The range of expected generation under normal conditions. We use this to detect when actual performance deviates significantly.
- **P10**: 10th percentile (lower bound of normal)
- **P50**: Median expectation (typical good day)
- **P90**: 90th percentile (upper bound)

If actual generation falls below P10 → suspect cloud/soiling/curtailment

### Feature Separation
We want features that clearly distinguish:
- **Healthy**: High efficiency, smooth profile, MC percentile ~50
- **Cloudy**: Lower efficiency, variable profile, MC percentile < 30
- **Soiling**: Gradual decline, morning > afternoon, efficiency decreases over time

### Confidence Calibration
A prediction is "reliable" if high confidence correlates with high accuracy.
- **Well-calibrated**: 90% confidence → 90% actual accuracy
- **Overconfident**: 90% confidence → 70% actual accuracy
- We measure this in Phase 4 and can improve with post-hoc calibration

---

## Expected Results

### Phase 1
- MC matches healthy days: correlation > 0.9
- Unhealthy days in lower percentiles: median percentile < 40

### Phase 2
- Top 3 features separated by Cohen's d > 0.8
- Feature correlations < 0.7 (no strong redundancy)

### Phase 3
- 50-100 labeled days across 5 classes
- No class representing > 50% of data

### Phase 4
- Accuracy: > 70%
- High-confidence accuracy: > 80%
- Precision for "Investigate": > 0.6

### Phase 5
- Daily classifications: < 5 min compute time
- Alert rate (Investigate + anomalies): < 20%
- Manual review finds real issues: > 50% true positive rate

