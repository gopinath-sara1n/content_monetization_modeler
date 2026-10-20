import sys
from pathlib import Path

# Run tests
from test_model import (
    test_model_files_exist,
    test_model_loads_and_predicts,
    test_non_negativity_and_zero_inputs,
    test_large_scale_revenue
)
from test_app import (
    test_predict_revenue_standard,
    test_predict_revenue_zero_views,
    test_predict_revenue_zero_video_length,
    test_revenue_component_sum
)

def run_all():
    tests = [
        ("test_model_files_exist", test_model_files_exist),
        ("test_model_loads_and_predicts", test_model_loads_and_predicts),
        ("test_non_negativity_and_zero_inputs", test_non_negativity_and_zero_inputs),
        ("test_large_scale_revenue", test_large_scale_revenue),
        ("test_predict_revenue_standard", test_predict_revenue_standard),
        ("test_predict_revenue_zero_views", test_predict_revenue_zero_views),
        ("test_predict_revenue_zero_video_length", test_predict_revenue_zero_video_length),
        ("test_revenue_component_sum", test_revenue_component_sum),
    ]

    passed = 0
    failed = 0
    print("=" * 60)
    print("Running Test Suite for Content Monetization Modeler")
    print("=" * 60)
    for name, func in tests:
        try:
            func()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Summary: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_all()
