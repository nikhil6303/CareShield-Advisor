# CareShield Advisor: XGBoost Patient Churn Prediction & Retention Advisor

An enterprise-grade, clinical & health-plan member churn prediction platform built with **XGBoost exclusively**, paired with **SHAP TreeExplainer** interpretability and a rule-based **Retention Advisor** recommendation engine.

---

## 1. System Architecture Overview

CareShield Advisor uses **XGBoost as its sole churn prediction algorithm** across all inference, evaluation, and training pipelines.

```
                           +---------------------------+
                           |  React Frontend (Port 3000)
                           +-------------+-------------+
                                         | REST / JSON
                                         v
                           +---------------------------+
                           |   Flask API (Port 5000)   |
                           +-------------+-------------+
                                         |
                       +-----------------+-----------------+
                       |                                   |
                       v                                   v
             [Industry Registry]                 [Single-Pass Preprocessing]
           (industry_registry.py)                 (StandardScaler + OHE)
          - Single Active Industry                         |
          - XGBoost Type Validation                        v
          - Memory Lifecycle                     [XGBoost Classifier]
                       |                        (outputs/churn_model.pkl)
                       |                                   |
                       |                 +-----------------+-----------------+
                       |                 |                                   |
                       v                 v                                   v
               [Active Industry]   [Predicted P(Churn)]            [SHAP TreeExplainer]
            (Healthcare / CareShield)    |                       (Feature Risk Drivers)
                                         |                                   |
                                         +-----------------+-----------------+
                                                           |
                                                           v
                                              [Retention Advisor Engine]
                                                (retention_advisor.py)
                                                           |
                                                           v
                                            [Structured API Response]
```

### Key Architectural Tenets:
1. **Exclusive XGBoost Model**: All alternative prediction models (Logistic Regression, Random Forests, Voting Classifiers, Stacking, or fallback heuristics) have been removed from the active prediction pipeline.
2. **Strict Model Enforcement**: The system strictly validates loaded models using `_validate_is_xgboost_model()`. Any attempt to load non-XGBoost objects triggers a `TypeError`.
3. **Low-Overhead Inference**: The XGBoost model is loaded once at application startup or via controlled lazy loading. Redundant data conversions and duplicate feature engineering passes have been replaced with single-pass preprocessing (`preprocess_single`), delivering sub-6ms model inference and ~51ms full explainable predictions.
4. **Single-Active-Model Memory Footprint**: An extensible `IndustryRegistry` guarantees that only ONE industry-specific model bundle resides in memory at any given time.
5. **Explainability & Recommendations**: Predictions are explained using **SHAP TreeExplainer** (`tree_path_dependent` mode), feeding into deterministic, evidence-based retention rules.

---

## 2. Model Performance & Benchmarks

The CareShield healthcare churn model was trained and evaluated on 2,000 health-plan members using 5-fold Stratified Cross-Validation on training data and evaluated on held-out test data (400 records, 68.2% baseline churn rate):

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Algorithm** | **XGBoost (XGBClassifier)** | Tuned: `n_estimators=150`, `max_depth=3`, `learning_rate=0.10`, `subsample=0.8` |
| **ROC-AUC** | **0.6416** | Primary evaluation metric (threshold-independent discriminative power) |
| **F1-Score** | **0.7091** | Secondary metric balancing precision and recall |
| **Precision** | **0.7480** | Positive predictive value for churn identification |
| **Recall (Sensitivity)** | **0.6740** | True positive rate (capturing actual churners) |
| **Accuracy** | **62.25%** | Evaluated at standard fixed threshold = 0.50 |
| **5-Fold CV AUC** | **0.6217 ± 0.0164** | Measured leakage-free on training splits only |

---

## 3. Lightweight Industry Registry (`industry_registry.py`)

The system includes a lightweight industry registry designed for future expansion across other domains (e.g., life insurance, clinical networks, pharmacy benefit managers) while keeping CareShield Healthcare as the working default.

### Key Rules:
- **No Speculative Support**: Support is only claimed for industries that have a trained and validated XGBoost model and preprocessor.
- **Single Active Industry in Memory**: When a new industry is loaded, the existing model and explainer are unloaded and garbage-collected before instantiating the new bundle.
- **Default Industry**: The default configuration is `healthcare` (`Healthcare (CareShield Health Plan)`).

### How to Add & Train a New Industry:
1. Prepare the industry dataset with raw features and binary target `Churned` (0 = retained, 1 = churned).
2. Create feature engineering and preprocessing pipeline saving `outputs/<industry>_preprocessor.pkl` and `feature_names.csv`.
3. Train an **XGBClassifier** using cross-validation and save to `outputs/<industry>_churn_model.pkl`.
4. Register the schema in `industry_registry.py`:
   ```python
   from industry_registry import IndustrySchema, get_registry

   NEW_INDUSTRY_SCHEMA = IndustrySchema(
       industry_id="pharmacy",
       display_name="Pharmacy Benefit Plan",
       description="Member churn prediction for pharmacy benefit management using XGBoost.",
       model_path="outputs/pharmacy_churn_model.pkl",
       preprocessor_path="outputs/pharmacy_preprocessor.pkl",
       feature_names_path="outputs/pharmacy_feature_names.csv",
       required_feature_columns=["MemberID", "Refill_Rate", "Copay_Amount", ...],
       numeric_feature_columns=["Refill_Rate", "Copay_Amount", ...],
       binary_feature_columns={"Prior_Auth_Denied"},
       categorical_feature_columns=["Drug_Tier", "Plan_Type"],
       feature_labels={"Refill_Rate": "Prescription Refill Adherence Rate (%)"},
       defaults={"Refill_Rate": 85.0, "Copay_Amount": 25.0},
       target_column="Churned",
       algorithm="XGBoost"
   )

   get_registry().register_industry(NEW_INDUSTRY_SCHEMA)
   ```

---

## 4. API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check endpoint (`{"status": "ok"}`). |
| `GET` | `/industries` | Returns all registered industries, active status, and confirmation of XGBoost exclusivity. |
| `GET` | `/industry/active` | Returns active industry schema, required features, and model metadata. |
| `POST` | `/industry/select` | Safely switches active industry in memory (`{"industry_id": "healthcare"}`). |
| `GET` | `/members` | Retrieves precomputed predictions and risk tiers for all dataset members. |
| `GET` | `/member/<id>` | Retrieves comprehensive profile, top SHAP drivers, and retention actions for a member. |
| `GET` | `/analytics` | Global risk distribution, tenure trend cohorts, and mean absolute SHAP importances. |
| `POST` | `/predict` | Dynamic inference for a single member JSON payload using active XGBoost model. |
| `POST` | `/predict-file` | Batch analysis for uploaded `.csv`, `.xlsx`, or `.xls` member spreadsheets. |
| `GET` | `/load-demo` | Automatically runs batch prediction on bundled demo dataset without file picker. |

### Sample `POST /predict` Request:
```json
{
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
  "Distance_To_Facility_Miles": 8.0
}
```

### Sample `POST /predict` Response:
```json
{
  "churn_probability": 0.2796,
  "risk_level": "Low",
  "drivers": [
    {
      "feature": "Overall_Satisfaction",
      "label": "Overall satisfaction score",
      "value": "4.5",
      "shap_value": -0.3214,
      "direction": "decreases_risk",
      "impact": "Overall satisfaction score is associated with decreased predicted churn risk (SHAP contribution: -0.3214)."
    }
  ],
  "recommendations": [
    {
      "action": "General Member Engagement -- Routine Member Monitoring",
      "category": "General Member Engagement",
      "priority": "Low",
      "reason": "Standard routine monitoring applies to all members regardless of current risk signal.",
      "rule_id": "GE-005",
      "suggested_next_step": "Include member in standard periodic engagement communications and routine care reminder workflows."
    }
  ]
}
```

---

## 5. Running & Testing the Project

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Node.js 20+

### Installation
```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### Step 1: Run Preprocessing & Training
```bash
# 1. Generate preprocessed train/test datasets and ColumnTransformer
python preprocessing.py

# 2. Train and evaluate XGBoost exclusively (Stratified 5-Fold CV + Grid Tuning)
python train_model.py

# 3. Generate comprehensive evaluation report
python evaluate_model.py
```

### Step 2: Run Architecture & Workflow Tests
```bash
# Verify XGBoost exclusivity, latency benchmarks, and memory lifecycle
python test_xgboost_architecture.py

# Verify REST API uploads, spreadsheet parsers, and error handling
python test_workflow.py
```

### Step 3: Run the Application
```bash
# Terminal 1: Start Flask REST API Backend
python app.py

# Terminal 2: Start React SPA Dashboard
cd frontend
npm run dev
```
- Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

## 6. Production Deployment (Render & Vercel)

The repository is pre-configured for instant deployment on **Render.com** (Flask backend) and **Vercel** (React frontend). See [DEPLOYMENT.md](file:///c:/Users/mehta/OneDrive/Desktop/CTS_NPN/DEPLOYMENT.md) for full step-by-step instructions.

- **Procfile** and `render.yaml` are included for automated Render service creation.
- Model artifacts are loaded via robust absolute paths derived from `__file__`.
- All production dependencies are pinned in `requirements.txt`.

