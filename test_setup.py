"""
Verification script: Test all imports and basic functionality
"""

import sys
sys.path.insert(0, '/workspaces/O-M-Monte-Carlo')

def test_imports():
    print("Testing imports...")
    
    # Core modules
    from src.data_loader import SolarDataLoader
    print("✓ SolarDataLoader")
    
    from src.phase1_physics.monte_carlo import MonteCarloSimulator
    print("✓ MonteCarloSimulator")
    
    from src.phase2_features.feature_engineering import FeatureEngineer
    print("✓ FeatureEngineer")
    
    from src.phase3_labeling.labeler import SolarDayLabeler
    print("✓ SolarDayLabeler")
    
    from src.phase4_model.rf_classifier import SolarAnomalyRF
    print("✓ SolarAnomalyRF")
    
    from src.phase5_production.daily_classifier import (
        DailyClassificationLogger, 
        DailyProductionPipeline
    )
    print("✓ DailyClassificationLogger")
    print("✓ DailyProductionPipeline")
    
    return True

def test_basic_functionality():
    print("\nTesting basic functionality...")
    
    import pandas as pd
    import numpy as np
    from src.phase1_physics.monte_carlo import MonteCarloSimulator
    from src.phase2_features.feature_engineering import FeatureEngineer
    
    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    df = pd.DataFrame({
        'yield': np.random.uniform(50, 200, 100),
        'ghi': np.random.uniform(1000, 5000, 100),
        'temp': np.random.uniform(10, 30, 100)
    }, index=dates)
    
    print(f"✓ Created test data: {df.shape}")
    
    # Test MC simulator
    mc = MonteCarloSimulator(n_simulations=100)
    healthy_mask = (df['yield'] / df['ghi']) > 0.03
    mc.calibrate_from_healthy_days(df, healthy_mask.values)
    print(f"✓ MC calibrated: eff={mc.efficiency_mean:.4f}, std={mc.efficiency_std:.4f}")
    
    # Test feature engineer
    fe = FeatureEngineer()
    sims = mc.simulate_day(np.array([df.iloc[0]['ghi']]))
    features = fe.compute_day_features(
        np.array([df.iloc[0]['yield']]), 
        np.array([df.iloc[0]['ghi']]), 
        sims
    )
    print(f"✓ Computed {len(features)} features")
    
    return True

if __name__ == '__main__':
    try:
        test_imports()
        test_basic_functionality()
        print("\n" + "="*50)
        print("✓ ALL TESTS PASSED")
        print("="*50)
        print("\nProject is ready to use!")
        print("\nNext steps:")
        print("1. Prepare your solar data in: data/solar_data.csv")
        print("2. Run Phase 1: jupyter notebook notebooks/01_phase1_physics_baseline.ipynb")
        print("3. Follow through all 5 phases in sequence")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
