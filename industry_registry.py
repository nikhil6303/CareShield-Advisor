"""
industry_registry.py
====================
Lightweight Industry Registry and Model Management for CareShield Churn Prediction.

Architecture & Principles:
--------------------------
1. XGBoost Exclusive:
   Strictly enforces XGBoost as the ONLY churn prediction algorithm. Rejects
   any non-XGBoost models (linear models, random forests, voting classifiers,
   stacking, or fallback classifiers).

2. Single Active Industry in Memory:
   Ensures that only ONE industry-specific model bundle (model, preprocessor,
   SHAP explainer) is held in memory at a time, eliminating RAM bloat and overhead.

3. Low-Overhead Caching & Lazy Loading:
   Loads and validates the XGBoost model once at startup or on demand. Avoids
   reloading from disk or re-initializing the SHAP TreeExplainer for each request.

4. Extensible Schema:
   Allows future industry expansion by defining schema, paths, and defaults per
   industry while keeping Healthcare / CareShield as the working default.
   New industries can be registered only when a validated XGBoost model is trained.

Author  : CTS NPN Project / CareShield Advisor
Created : 2026-10-10
"""

import gc
import os
import pickle
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Base output directory — absolute path so the app loads from any working directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(_THIS_DIR, "outputs")
SNAPSHOT_DATE = pd.Timestamp("2026-08-13")


@dataclass
class IndustrySchema:
    """Defines the feature specification, paths, and metadata for an industry."""
    industry_id: str
    display_name: str
    description: str
    model_path: str
    preprocessor_path: str
    feature_names_path: str
    required_feature_columns: List[str]
    numeric_feature_columns: List[str]
    binary_feature_columns: Set[str]
    categorical_feature_columns: List[str]
    feature_labels: Dict[str, str]
    defaults: Dict[str, Any]
    target_column: str = "Churned"
    algorithm: str = "XGBoost"
    metadata: Dict[str, Any] = field(default_factory=dict)


# Healthcare / CareShield Default Schema
CARE_SHIELD_HEALTHCARE_SCHEMA = IndustrySchema(
    industry_id="healthcare",
    display_name="Healthcare (CareShield Health Plan)",
    description="Member churn prediction and retention advisory using tuned XGBoost gradient boosting.",
    model_path=os.path.join(OUTPUTS_DIR, "churn_model.pkl"),
    preprocessor_path=os.path.join(OUTPUTS_DIR, "preprocessor.pkl"),
    feature_names_path=os.path.join(OUTPUTS_DIR, "feature_names.csv"),
    required_feature_columns=[
        "Age", "Gender", "State", "Tenure_Months", "Specialty", "Insurance_Type",
        "Visits_Last_Year", "Missed_Appointments", "Days_Since_Last_Visit",
        "Last_Interaction_Date", "Overall_Satisfaction", "Wait_Time_Satisfaction",
        "Staff_Satisfaction", "Provider_Rating", "Avg_Out_Of_Pocket_Cost",
        "Billing_Issues", "Portal_Usage", "Referrals_Made", "Distance_To_Facility_Miles"
    ],
    numeric_feature_columns=[
        "Age", "Tenure_Months", "Visits_Last_Year", "Missed_Appointments",
        "Days_Since_Last_Visit", "Overall_Satisfaction", "Wait_Time_Satisfaction",
        "Staff_Satisfaction", "Provider_Rating", "Avg_Out_Of_Pocket_Cost",
        "Billing_Issues", "Portal_Usage", "Referrals_Made", "Distance_To_Facility_Miles"
    ],
    binary_feature_columns={"Billing_Issues", "Portal_Usage"},
    categorical_feature_columns=["Gender", "State", "Specialty", "Insurance_Type"],
    feature_labels={
        "Age": "Age (years)",
        "Tenure_Months": "Membership tenure (months)",
        "Visits_Last_Year": "Visits in last year",
        "Missed_Appointments": "Missed appointments (count)",
        "Days_Since_Last_Visit": "Days since last clinical visit",
        "Overall_Satisfaction": "Overall satisfaction score",
        "Wait_Time_Satisfaction": "Wait-time satisfaction score",
        "Staff_Satisfaction": "Staff satisfaction score",
        "Provider_Rating": "Provider rating",
        "Avg_Out_Of_Pocket_Cost": "Average out-of-pocket cost (USD)",
        "Billing_Issues": "Billing issue recorded (0/1)",
        "Portal_Usage": "Patient portal usage (0/1)",
        "Referrals_Made": "Referrals made (count)",
        "Distance_To_Facility_Miles": "Distance to facility (miles)",
        "days_since_interaction": "Days since last recorded interaction",
        "interaction_month": "Month of last interaction (1-12)",
        "missed_appt_rate": "Missed appointment rate",
        "composite_satisfaction": "Composite satisfaction score",
        "engagement_score": "Engagement score (weighted)",
        "cost_per_tenure_month": "Out-of-pocket cost per tenure month (USD)",
        "is_low_satisfaction": "Low satisfaction flag (score < 2.5)",
        "is_distant": "Distance barrier flag (> 40 miles)",
        "monthly_visit_rate": "Monthly visit rate",
    },
    defaults={
        "Age": 54, "Gender": "Male", "State": "MI", "Tenure_Months": 60,
        "Specialty": "Family Medicine", "Insurance_Type": "Private",
        "Visits_Last_Year": 8, "Missed_Appointments": 2,
        "Days_Since_Last_Visit": 363, "Last_Interaction_Date": "2025-06-03",
        "Overall_Satisfaction": 3.2, "Wait_Time_Satisfaction": 3.3,
        "Staff_Satisfaction": 3.5, "Provider_Rating": 3.8,
        "Avg_Out_Of_Pocket_Cost": 716, "Billing_Issues": 0,
        "Portal_Usage": 0, "Referrals_Made": 1, "Distance_To_Facility_Miles": 25.0
    },
    target_column="Churned",
    algorithm="XGBoost",
    metadata={
        "best_hyperparameters": {
            "n_estimators": 150,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.8
        },
        "cv_roc_auc_mean": 0.6217,
        "test_roc_auc": 0.6416,
        "test_f1": 0.7091,
        "evaluation_metric": "roc_auc"
    }
)


def _validate_is_xgboost_model(model_obj: Any) -> None:
    """
    Strict validation that the given model object is an XGBoost model.
    Raises TypeError if any other model algorithm or ensemble is detected.
    """
    model_cls_name = getattr(model_obj, "__class__", type(model_obj)).__name__
    is_xgb = (
        isinstance(model_obj, (XGBClassifier, xgb.Booster))
        or "XGB" in model_cls_name
        or model_cls_name in ("XGBClassifier", "Booster")
    )
    if not is_xgb:
        raise TypeError(
            f"Algorithm violation: CareShield requires XGBoost exclusively for churn prediction. "
            f"Received model of type '{model_cls_name}', which is not an XGBoost model."
        )


class IndustryModelBundle:
    """
    Encapsulates loaded in-memory assets for a single industry:
    - XGBoost model
    - Fitted ColumnTransformer preprocessor
    - Post-OHE feature names list
    - Cached SHAP TreeExplainer
    """
    def __init__(
        self,
        schema: IndustrySchema,
        model: Any,
        preprocessor: Any,
        feature_names: List[str],
        explainer: shap.TreeExplainer,
    ):
        _validate_is_xgboost_model(model)
        self.schema = schema
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.explainer = explainer

    def predict_proba(self, X_processed: np.ndarray) -> np.ndarray:
        """Return churn probability vector [P(Churn=1)] for preprocessed feature array."""
        return self.model.predict_proba(X_processed)[:, 1]

    def explain(self, X_processed: np.ndarray) -> np.ndarray:
        """Compute SHAP values using the cached TreeExplainer."""
        return self.explainer.shap_values(X_processed)

    def apply_feature_engineering(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Apply deterministic feature engineering consistent with training."""
        df = df_raw.copy()
        if "Last_Interaction_Date" in df.columns:
            df["Last_Interaction_Date"] = pd.to_datetime(df["Last_Interaction_Date"], errors="coerce")
            df["days_since_interaction"] = (SNAPSHOT_DATE - df["Last_Interaction_Date"]).dt.days
            df["interaction_month"] = df["Last_Interaction_Date"].dt.month
            df.drop(columns=["Last_Interaction_Date"], inplace=True)

        sat_cols = ["Overall_Satisfaction", "Wait_Time_Satisfaction", "Staff_Satisfaction", "Provider_Rating"]
        available_sat = [c for c in sat_cols if c in df.columns]
        if available_sat:
            df["composite_satisfaction"] = df[available_sat].mean(axis=1)

        if "Missed_Appointments" in df.columns and "Visits_Last_Year" in df.columns:
            df["missed_appt_rate"] = df["Missed_Appointments"] / df["Visits_Last_Year"].clip(lower=1)

        if "Visits_Last_Year" in df.columns:
            portal = df["Portal_Usage"] if "Portal_Usage" in df.columns else 0
            referrals = df["Referrals_Made"] if "Referrals_Made" in df.columns else 0
            df["engagement_score"] = df["Visits_Last_Year"] + portal * 2 + referrals
            df["monthly_visit_rate"] = df["Visits_Last_Year"] / 12

        if "Avg_Out_Of_Pocket_Cost" in df.columns and "Tenure_Months" in df.columns:
            df["cost_per_tenure_month"] = df["Avg_Out_Of_Pocket_Cost"] / df["Tenure_Months"].clip(lower=1)

        if "Overall_Satisfaction" in df.columns:
            df["is_low_satisfaction"] = (df["Overall_Satisfaction"] < 2.5).astype(int)

        if "Distance_To_Facility_Miles" in df.columns:
            df["is_distant"] = (df["Distance_To_Facility_Miles"] > 40).astype(int)

        return df

    def preprocess_single(self, raw_input_dict: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Single-pass feature engineering and transformation for one member.
        Avoids redundant dataframe creation and duplicate preprocessing.
        Returns (X_processed_1x42, engineered_feature_dict).
        """
        # Ensure default values for any missing fields
        inputs = dict(self.schema.defaults)
        inputs.update(raw_input_dict)

        cols_to_drop = ["PatientID", "Churned"]
        single_df = pd.DataFrame([inputs]).drop(columns=[c for c in cols_to_drop if c in inputs], errors="ignore")
        engineered_df = self.apply_feature_engineering(single_df)
        X_processed = self.preprocessor.transform(engineered_df)
        return X_processed, engineered_df.iloc[0].to_dict()


class IndustryRegistry:
    """
    Lightweight registry mapping industry configurations to their XGBoost models.
    Enforces that only ONE industry model bundle is active in memory at a time.
    """
    def __init__(self):
        self._schemas: Dict[str, IndustrySchema] = {}
        self._active_industry_id: Optional[str] = None
        self._active_bundle: Optional[IndustryModelBundle] = None

        # Register default CareShield healthcare configuration
        self.register_industry(CARE_SHIELD_HEALTHCARE_SCHEMA)

    def register_industry(self, schema: IndustrySchema) -> None:
        """Register an industry schema."""
        if not isinstance(schema, IndustrySchema):
            raise TypeError("Schema must be an instance of IndustrySchema.")
        self._schemas[schema.industry_id] = schema
        logger.info(f"Registered industry schema: '{schema.industry_id}' ({schema.display_name}).")

    def get_supported_industries(self) -> List[Dict[str, Any]]:
        """Return list of supported industries and their metadata."""
        items = []
        for ind_id, sch in self._schemas.items():
            is_active = (ind_id == self._active_industry_id)
            items.append({
                "industry_id": ind_id,
                "display_name": sch.display_name,
                "description": sch.description,
                "algorithm": sch.algorithm,
                "is_active": is_active,
                "is_default": (ind_id == "healthcare"),
                "features_count": len(sch.required_feature_columns),
                "metadata": sch.metadata
            })
        return items

    def get_active_industry_id(self) -> str:
        """Return current active industry identifier (defaults to 'healthcare')."""
        return self._active_industry_id or "healthcare"

    def get_active_bundle(self) -> IndustryModelBundle:
        """Return active model bundle; lazy-loads default if none loaded."""
        if self._active_bundle is None:
            return self.load_industry(self.get_active_industry_id())
        return self._active_bundle

    def load_industry(self, industry_id: str) -> IndustryModelBundle:
        """
        Load the requested industry into memory.
        Enforces single-active-industry memory constraint by freeing any previous model.
        """
        if industry_id not in self._schemas:
            supported = list(self._schemas.keys())
            raise ValueError(
                f"Unsupported industry '{industry_id}'. "
                f"Currently supported industries with validated XGBoost models: {supported}."
            )

        # Return cached bundle if already loaded
        if self._active_bundle is not None and self._active_industry_id == industry_id:
            return self._active_bundle

        schema = self._schemas[industry_id]

        # 1. Unload existing bundle to ensure only ONE industry model in memory
        if self._active_bundle is not None:
            logger.info(f"Unloading previous industry model '{self._active_industry_id}' from memory...")
            self._active_bundle = None
            self._active_industry_id = None
            gc.collect()

        logger.info(f"Loading XGBoost model and assets for industry '{industry_id}'...")

        # 2. Check model file existence
        model_path = schema.model_path
        if not os.path.exists(model_path):
            if industry_id == "healthcare":
                alt_path = os.path.join(OUTPUTS_DIR, "best_model.pkl")
                if os.path.exists(alt_path):
                    model_path = alt_path
                else:
                    raise FileNotFoundError(
                        f"Model file for industry '{industry_id}' not found at '{schema.model_path}'."
                    )
            else:
                raise FileNotFoundError(
                    f"Model file for industry '{industry_id}' not found at '{schema.model_path}'."
                )

        # 3. Load and strictly validate XGBoost model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        _validate_is_xgboost_model(model)
        logger.info(f"Successfully loaded and verified XGBoost model for '{industry_id}'.")

        # 4. Load Preprocessor
        if not os.path.exists(schema.preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor file for industry '{industry_id}' not found at '{schema.preprocessor_path}'."
            )
        with open(schema.preprocessor_path, "rb") as f:
            bundle = pickle.load(f)
            preprocessor = bundle["preprocessor"] if isinstance(bundle, dict) and "preprocessor" in bundle else bundle

        # 5. Load Feature Names
        if not os.path.exists(schema.feature_names_path):
            raise FileNotFoundError(
                f"Feature names file for industry '{industry_id}' not found at '{schema.feature_names_path}'."
            )
        feature_names = pd.read_csv(schema.feature_names_path)["feature"].tolist()

        # 6. Initialize SHAP TreeExplainer for XGBoost once
        logger.info(f"Initializing SHAP TreeExplainer (tree_path_dependent mode) for XGBoost...")
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )

        # 7. Store active bundle
        self._active_bundle = IndustryModelBundle(
            schema=schema,
            model=model,
            preprocessor=preprocessor,
            feature_names=feature_names,
            explainer=explainer,
        )
        self._active_industry_id = industry_id
        logger.info(f"Industry '{industry_id}' is now active in memory.")
        return self._active_bundle

    def unload(self) -> None:
        """Explicitly release in-memory model assets."""
        self._active_bundle = None
        self._active_industry_id = None
        gc.collect()


# Global Singleton Instance
REGISTRY = IndustryRegistry()


def get_registry() -> IndustryRegistry:
    """Return the global IndustryRegistry singleton instance."""
    return REGISTRY
