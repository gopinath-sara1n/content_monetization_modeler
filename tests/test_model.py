import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

def test_model_files_exist():
    assert (BASE_DIR / "model.pkl").exists(), "model.pkl must exist in repository root"
    assert (BASE_DIR / "columns.pkl").exists(), "columns.pkl must exist in repository root"

def test_model_loads_and_predicts():
    with open(BASE_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "columns.pkl", "rb") as f:
        columns = pickle.load(f)
    
    sample_df = pd.DataFrame([{
        'views': 10000,
        'likes': 1000,
        'comments': 200,
        'watch_time_minutes': 30000,
        'video_length_minutes': 15,
        'retention_rate': 0.2,
        'log_subscribers': 13.1,
        'engagement_rate': 0.12
    }])[columns]
    
    pred = model.predict(sample_df)[0]
    # Expected value around $214.16
    assert pred > 200 and pred < 230, f"Unexpected prediction value: {pred}"

def test_non_negativity_and_zero_inputs():
    with open(BASE_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "columns.pkl", "rb") as f:
        columns = pickle.load(f)
    
    zero_df = pd.DataFrame([{
        'views': 0,
        'likes': 0,
        'comments': 0,
        'watch_time_minutes': 0,
        'video_length_minutes': 15,
        'retention_rate': 0.0,
        'log_subscribers': 0.0,
        'engagement_rate': 0.0
    }])[columns]
    
    pred = model.predict(zero_df)[0]
    assert pred >= 0.0, f"Model predicted negative revenue: {pred}"
    assert np.isclose(pred, 0.0, atol=1e-5), f"Zero inputs must predict 0 revenue, got {pred}"

def test_large_scale_revenue():
    with open(BASE_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "columns.pkl", "rb") as f:
        columns = pickle.load(f)
    
    large_df = pd.DataFrame([{
        'views': 1_000_000,
        'likes': 100_000,
        'comments': 20_000,
        'watch_time_minutes': 3_000_000,
        'video_length_minutes': 15,
        'retention_rate': 0.2,
        'log_subscribers': 15.0,
        'engagement_rate': 0.12
    }])[columns]
    
    pred = model.predict(large_df)[0]
    # Expected: 0.00521039*1M + 0.01718735*120k + 0.00471454*3M = 5210.39 + 2062.48 + 14143.62 = 21416.49
    assert np.isclose(pred, 21416.49, rtol=1e-3), f"Prediction deviated on large scale: {pred}"
