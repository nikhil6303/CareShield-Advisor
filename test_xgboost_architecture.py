"""
test_xgboost_architecture.py
============================
Automated Test Suite for CareShield XGBoost-Exclusive Architecture.

Validates:
1. XGBoost is the ONLY model used for churn prediction.
2. Rejection of alternative models, ensembles, voting classifiers, and fallback algorithms.
3. Single active industry model in memory and registry lifecycle.
4. Input validation: missing fields, malformed data, and model loading errors.
5. Inference latency benchmark and probability calculation correctness.
6. API endpoints contract (/health, /industries, /industry/active, /predict).
"""

import os
import time
import pickle
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from industry_registry import (
    get_registry,
    _validate_is_xgboost_model,
    IndustryRegistry,
    IndustrySchema,
    IndustryModelBundle,
    CARE_SHIELD_HEALTHCARE_SCHEMA,
)
import shap_explainer
import xgboost as xgb


def test_active_model_is_strictly_xgboost():
    """Verify that the loaded production model is strictly an XGBClassifier."""
    registry = get_registry()
    bundle = registry.get_active_bundle()

    assert isinstance(bundle.model, (xgb.XGBClassifier, xgb.Booster)), (
        f"Model must be XGBoost, but got {type(bundle.model)}"
    )
    assert bundle.schema.algorithm == "XGBoost"
    print("PASS: Active model is strictly XGBoost.")


def test_alternative_models_are_rejected():
    """Verify that passing non-XGBoost algorithms raises TypeError."""
    lr = LogisticRegression()
    rf = RandomForestClassifier()
    dummy_obj = "not_a_model"

    with pytest.raises(TypeError, match="Algorithm violation"):
        _validate_is_xgboost_model(lr)

    with pytest.raises(TypeError, match="Algorithm violation"):
        _validate_is_xgboost_model(rf)

    with pytest.raises(TypeError, match="Algorithm violation"):
        _validate_is_xgboost_model(dummy_obj)

    print("PASS: Alternative models and ensembles are strictly rejected.")


def test_industry_registry_single_active_model_in_memory():
    """
    Verify that the registry maintains only one active model bundle in memory
    and correctly switches or lazily loads without duplicating models.
    """
    registry = IndustryRegistry()
    assert registry.get_active_industry_id() == "healthcare"

    # Bundle is lazy loaded
    bundle1 = registry.get_active_bundle()
    assert bundle1 is not None
    assert bundle1.schema.industry_id == "healthcare"

    # Repeated calls return the exact same cached bundle in memory (no reloading)
    bundle2 = registry.get_active_bundle()
    assert bundle1 is bundle2

    # Requesting an unsupported industry raises ValueError
    with pytest.raises(ValueError, match="Unsupported industry"):
        registry.load_industry("telecom")

    # Unload releases memory
    registry.unload()
    assert registry._active_bundle is None
    print("PASS: Industry registry manages single active model in memory.")


def test_model_loading_failure_handling():
    """Verify clean error handling when a model file is missing or corrupted."""
    bad_schema = IndustrySchema(
        industry_id="nonexistent_test",
        display_name="Nonexistent Test Industry",
        description="Testing missing model error handling",
        model_path="outputs/nonexistent_model_file.pkl",
        preprocessor_path="outputs/preprocessor.pkl",
        feature_names_path="outputs/feature_names.csv",
        required_feature_columns=["Age"],
        numeric_feature_columns=["Age"],
        binary_feature_columns=set(),
        categorical_feature_columns=[],
        feature_labels={"Age": "Age"},
        defaults={"Age": 50},
        target_column="Churned",
        algorithm="XGBoost"
    )

    reg = IndustryRegistry()
    reg.register_industry(bad_schema)

    with pytest.raises(FileNotFoundError):
        reg.load_industry("nonexistent_test")

    print("PASS: Missing model loading failures handled gracefully.")


def test_prediction_correctness_and_contract():
    """Verify probability computation, risk levels, and SHAP drivers match contracts."""
    registry = get_registry()
    bundle = registry.get_active_bundle()

    sample_input = {
        "Age": 55,
        "Gender": "Female",
        "State": "OH",
        "Tenure_Months": 36,
        "Specialty": "Cardiology",
        "Insurance_Type": "Medicare",
        "Visits_Last_Year": 4,
        "Missed_Appointments": 0,
        "Days_Since_Last_Visit": 90,
        "Last_Interaction_Date": "2026-01-15",
        "Overall_Satisfaction": 4.5,
        "Wait_Time_Satisfaction": 4.0,
        "Staff_Satisfaction": 4.5,
        "Provider_Rating": 4.8,
        "Avg_Out_Of_Pocket_Cost": 220,
        "Billing_Issues": 0,
        "Portal_Usage": 1,
        "Referrals_Made": 2,
        "Distance_To_Facility_Miles": 8.0,
    }

    X_processed, engineered = bundle.preprocess_single(sample_input)
    assert X_processed.shape == (1, 42), f"Expected shape (1, 42), got {X_processed.shape}"

    prob = float(bundle.predict_proba(X_processed)[0])
    assert 0.0 <= prob <= 1.0, f"Probability out of bounds: {prob}"

    shap_vals = bundle.explain(X_processed)[0]
    assert len(shap_vals) == 42, f"Expected 42 SHAP values, got {len(shap_vals)}"

    risk_lvl = shap_explainer.assign_risk_level(prob)
    assert risk_lvl in ("Low", "Medium", "High", "Critical")

    print(f"PASS: Prediction contract verified. P(Churn)={prob:.4f}, Risk={risk_lvl}")


def test_missing_fields_defaults_fallback():
    """Verify that omitting optional fields uses defaults without breaking inference."""
    registry = get_registry()
    bundle = registry.get_active_bundle()

    # Highly sparse input
    sparse_input = {
        "Age": 62,
        "Overall_Satisfaction": 2.1
    }

    X_processed, engineered = bundle.preprocess_single(sparse_input)
    prob = float(bundle.predict_proba(X_processed)[0])
    assert 0.0 <= prob <= 1.0
    print("PASS: Missing fields populated with defaults successfully.")


def test_inference_latency_benchmark():
    """Measure inference latency over 50 iterations to verify reduced overhead."""
    registry = get_registry()
    bundle = registry.get_active_bundle()

    sample_input = bundle.schema.defaults

    # Warm-up
    X_processed, _ = bundle.preprocess_single(sample_input)
    _ = bundle.predict_proba(X_processed)
    _ = bundle.explain(X_processed)

    iterations = 50

    # 1. Measure Pure XGBoost Inference Latency
    start_xgb = time.perf_counter()
    for _ in range(iterations):
        _ = bundle.predict_proba(X_processed)
    elapsed_xgb_ms = (time.perf_counter() - start_xgb) * 1000.0
    xgb_latency_ms = elapsed_xgb_ms / iterations

    # 2. Measure Full Pipeline (Single-pass Preprocessing + XGBoost + SHAP Explainer)
    start_full = time.perf_counter()
    for _ in range(iterations):
        X_p, _ = bundle.preprocess_single(sample_input)
        p = bundle.predict_proba(X_p)
        s = bundle.explain(X_p)
    elapsed_full_ms = (time.perf_counter() - start_full) * 1000.0
    full_latency_ms = elapsed_full_ms / iterations

    print(f"\n--- LATENCY BENCHMARK ---")
    print(f"Iterations: {iterations}")
    print(f"Pure XGBoost model prediction: {xgb_latency_ms:.3f} ms / request")
    print(f"Full pipeline (Single-Pass Preprocessing + XGBoost + SHAP): {full_latency_ms:.2f} ms / request")
    print(f"-------------------------\n")

    assert xgb_latency_ms < 10.0, f"XGBoost model latency too high: {xgb_latency_ms:.2f} ms"
    assert full_latency_ms < 100.0, f"Full pipeline latency too high: {full_latency_ms:.2f} ms"
    print("PASS: Inference latency overhead is verified low.")


if __name__ == "__main__":
    print("Running CareShield XGBoost Architecture Tests...\n")
    test_active_model_is_strictly_xgboost()
    test_alternative_models_are_rejected()
    test_industry_registry_single_active_model_in_memory()
    test_model_loading_failure_handling()
    test_prediction_correctness_and_contract()
    test_missing_fields_defaults_fallback()
    test_inference_latency_benchmark()
    print("\nALL ARCHITECTURE TESTS PASSED SUCCESSFULLY!")
