"""
Task 1 — Experiment Tracking & Model Comparison (6 marks)
==========================================================
- Trains SVR and GradientBoosting on training_data.csv
- Logs hyperparams, MAE / RMSE / R² / MAPE to MLflow
- Tags every run with experiment_type = "baseline_comparison"
- Experiment name: "edutrack-completion-days"
- Picks best model by MAE (lower is better)
- Saves: results/step1_s1.json  |  models/best_model.pkl
"""

import os
import json
import math
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Resolve absolute paths so the script works from any cwd ──────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "training_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

EXPERIMENT_NAME = "edutrack-completion-days"
FEATURES        = ["course_hours", "quizzes_count", "difficulty_level", "learner_experience"]
TARGET          = "completion_days"


def compute_metrics(y_true, y_pred):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) /
                                np.where(y_true == 0, 1e-9, y_true))) * 100)
    return {
        "mae":  round(mae,  4),
        "rmse": round(rmse, 4),
        "r2":   round(r2,   4),
        "mape": round(mape, 4),
    }


def main():
    # ── Load data ─────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURES]
    y  = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── MLflow setup ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLRUNS_DIR)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Define models ─────────────────────────────────────────────────────
    models_config = {
        "SVR": SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ),
    }

    all_results = []
    trained_models = {}

    for name, model in models_config.items():
        with mlflow.start_run(run_name=name):
            # Tag as required
            mlflow.set_tag("experiment_type", "baseline_comparison")

            # Log all hyperparameters
            for k, v in model.get_params().items():
                mlflow.log_param(k, v)

            # Train
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            metrics = compute_metrics(y_test.values, y_pred)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model artifact
            mlflow.sklearn.log_model(model, artifact_path=name)

        trained_models[name] = model
        all_results.append({"name": name, **metrics})
        print(f"[{name}]  MAE={metrics['mae']}  RMSE={metrics['rmse']}"
              f"  R²={metrics['r2']}  MAPE={metrics['mape']}")

    # ── Select best by MAE ────────────────────────────────────────────────
    best = min(all_results, key=lambda r: r["mae"])
    best_model = trained_models[best["name"]]

    # ── Persist best model for Tasks 2, 3, 4 ─────────────────────────────
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    with open(os.path.join(MODEL_DIR, "best_model_name.txt"), "w") as f:
        f.write(best["name"])

    # ── Write results JSON ────────────────────────────────────────────────
    output = {
        "experiment_name":    EXPERIMENT_NAME,
        "models":             all_results,
        "best_model":         best["name"],
        "best_metric_name":   "mae",
        "best_metric_value":  best["mae"],
    }

    out_path = os.path.join(RESULT_DIR, "step1_s1.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅  Task 1 complete.")
    print(f"    Best model : {best['name']}  (MAE = {best['mae']})")
    print(f"    Output     : {out_path}")


if __name__ == "__main__":
    main()
