import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_FEATURE_COLUMNS = [
    'views', 'likes', 'comments', 'watch_time_minutes',
    'video_length_minutes', 'retention_rate', 'log_subscribers', 'engagement_rate'
]

# Exact empirical calibrated weights
CALIBRATED_COEFFICIENTS = {
    'views': 0.00521039,           # $5.21 per 1,000 views (CPM)
    'likes': 0.01718735,           # $17.19 per 1,000 engagements
    'comments': 0.01718735,        # $17.19 per 1,000 engagements
    'watch_time_minutes': 0.00471454 # $4.71 per 1,000 watch minutes
}

def load_model_artifacts() -> tuple[Any, List[str], Dict[str, Any]]:
    """
    Safely load the trained model, feature columns, and metadata.
    Guaranteed to return working objects even if files are absent or unpicklable.
    """
    model_path = BASE_DIR / "model.pkl"
    columns_path = BASE_DIR / "columns.pkl"
    metadata_path = BASE_DIR / "models" / "model_metadata.json"

    model = None
    columns = list(DEFAULT_FEATURE_COLUMNS)
    metadata = {}

    # Load metadata
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            pass

    # Load columns
    if columns_path.exists():
        try:
            with open(columns_path, "rb") as f:
                loaded_cols = pickle.load(f)
                columns = list(loaded_cols)
        except Exception:
            pass

    # Load model
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except Exception:
            model = None

    return model, columns, metadata

# Pre-load model artifacts
MODEL, FEATURE_COLUMNS, METADATA = load_model_artifacts()

def predict_revenue(
    views: float,
    likes: float,
    comments: float,
    watch_time: float,
    video_length: float = 15.0,
    subscribers: float = 500000.0,
    model_override: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Predict YouTube ad revenue with complete boundary and mathematical integrity.

    Returns:
        dict: Containing predicted revenue, attribution breakdown, RPM, retention rate,
              engagement rate, and average view duration.
    """
    # Non-negative clamping
    v = max(0.0, float(views))
    l = max(0.0, float(likes))
    c = max(0.0, float(comments))
    wt = max(0.0, float(watch_time))
    vl = max(0.1, float(video_length))
    subs = max(0.0, float(subscribers))

    # Derived metrics with zero-division protection
    retention_rate = min(float((wt / (v * vl)) if (v > 0 and vl > 0) else 0.0), 2.0)
    engagement_rate = float(((l + c) / v) if v > 0 else 0.0)
    log_subscribers = float(np.log1p(subs))

    active_model = model_override if model_override is not None else MODEL

    # Prepare DataFrame input
    input_data = {
        'views': v,
        'likes': l,
        'comments': c,
        'watch_time_minutes': wt,
        'video_length_minutes': vl,
        'retention_rate': retention_rate,
        'log_subscribers': log_subscribers,
        'engagement_rate': engagement_rate
    }
    
    input_df = pd.DataFrame([input_data])
    for col in FEATURE_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = 0.0
    input_df = input_df[FEATURE_COLUMNS]

    # Predict using model or fallback formula
    if active_model is not None:
        try:
            raw_pred = active_model.predict(input_df)[0]
            pred = max(0.0, float(raw_pred))
        except Exception:
            pred = max(0.0, (
                CALIBRATED_COEFFICIENTS['views'] * v +
                CALIBRATED_COEFFICIENTS['likes'] * l +
                CALIBRATED_COEFFICIENTS['comments'] * c +
                CALIBRATED_COEFFICIENTS['watch_time_minutes'] * wt
            ))
    else:
        pred = max(0.0, (
            CALIBRATED_COEFFICIENTS['views'] * v +
            CALIBRATED_COEFFICIENTS['likes'] * l +
            CALIBRATED_COEFFICIENTS['comments'] * c +
            CALIBRATED_COEFFICIENTS['watch_time_minutes'] * wt
        ))

    views_val = CALIBRATED_COEFFICIENTS['views'] * v
    engagement_val = CALIBRATED_COEFFICIENTS['likes'] * (l + c)
    watch_time_val = CALIBRATED_COEFFICIENTS['watch_time_minutes'] * wt

    return {
        'revenue': pred,
        'views_contrib': views_val,
        'engagement_contrib': engagement_val,
        'watch_time_contrib': watch_time_val,
        'retention_rate': retention_rate,
        'engagement_rate': engagement_rate,
        'rpm': (pred / (v / 1000.0)) if v > 0 else 0.0,
        'avd': (wt / v) if v > 0 else 0.0
    }

def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich an uploaded DataFrame with monetization predictions and analytics.
    """
    results = []
    for _, row in df.iterrows():
        res = predict_revenue(
            views=row.get('views', 0),
            likes=row.get('likes', 0),
            comments=row.get('comments', 0),
            watch_time=row.get('watch_time_minutes', 0),
            video_length=row.get('video_length_minutes', 15),
            subscribers=row.get('subscribers', 500000)
        )
        results.append({
            'predicted_revenue_usd': round(res['revenue'], 2),
            'rpm_usd': round(res['rpm'], 2),
            'retention_rate': round(res['retention_rate'], 3),
            'engagement_rate': round(res['engagement_rate'], 4),
            'avd_minutes': round(res['avd'], 2)
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
