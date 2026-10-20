import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to import app helpers
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from model_engine import predict_revenue, batch_predict

def test_predict_revenue_standard():
    res = predict_revenue(views=10000, likes=1000, comments=200, watch_time=30000, video_length=15, subscribers=500000)
    assert 'revenue' in res
    assert 'rpm' in res
    assert 'retention_rate' in res
    assert 'engagement_rate' in res
    assert 'views_contrib' in res
    assert 'engagement_contrib' in res
    assert 'watch_time_contrib' in res
    
    assert res['revenue'] > 200 and res['revenue'] < 230
    assert np.isclose(res['rpm'], (res['revenue'] / 10.0), atol=1e-2)

def test_predict_revenue_zero_views():
    # Must not raise ZeroDivisionError
    res = predict_revenue(views=0, likes=0, comments=0, watch_time=0, video_length=15)
    assert res['revenue'] == 0.0
    assert res['rpm'] == 0.0
    assert res['retention_rate'] == 0.0
    assert res['engagement_rate'] == 0.0

def test_predict_revenue_zero_video_length():
    # Must not raise ZeroDivisionError
    res = predict_revenue(views=1000, likes=100, comments=20, watch_time=3000, video_length=0)
    assert res['revenue'] > 0.0
    assert not np.isnan(res['retention_rate'])

def test_revenue_component_sum():
    res = predict_revenue(views=25000, likes=2000, comments=500, watch_time=60000)
    expected_sum = res['views_contrib'] + res['engagement_contrib'] + res['watch_time_contrib']
    assert np.isclose(res['revenue'], expected_sum, atol=1e-3)
