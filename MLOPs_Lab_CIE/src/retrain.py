"""
Task 4 — Retraining Pipeline (8 marks)
========================================
- Combines training_data.csv + new_data.csv (25 + 20 = 45 rows)
- Retrains the same model type that won Task 1
- Compares retrained vs champion on the SAME original test split
- Promotes only if RMSE improves by >= 1.0
- Saves: results/step4_s8.json
"""
import mlflow

mlflow.set_tracking_uri("file:///C:/Users/Admin/Desktop/MLOPs_Lab_CIE/mlruns")
import os
import json
import math
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH      = os.path.join(BASE_DIR, "data", "training_data.csv")
NEW_DATA_PATH   = os.path.join(BASE_DIR, "data", "new_data.csv")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
RESULT_DIR      = os.path.join(BASE_DIR, "results")
MLRUNS_DIR      = os.path.join(BASE_DIR, "mlruns")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

EXPERIMENT_NAME     = "edutrack-completion-days"
FEATURES            = ["course_hours", "quizzes_count", "difficulty_level", "learner_experience"]
TARGET              = "completion_days"
MIN_IMPROVEMENT     = 1.0          # RMSE must improve by at least this much


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def build_fresh_model(name):
    """Return a fresh (unfitted) instance of the champion model type."""
    if name == "GradientBoosting":
        return GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
    # fallback SVR
    return SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)


def main():
    # ── Identify champion model type from Task 1 ──────────────────────────
    name_file = os.path.join(MODEL_DIR, "best_model_name.txt")
    champion_name = "GradientBoosting"          # safe default
    if os.path.exists(name_file):
        with open(name_file) as f:
            champion_name = f.read().strip()
    print(f"Champion model type: {champion_name}")

    # ── Load datasets ─────────────────────────────────────────────────────
    original_df = pd.read_csv(TRAIN_PATH)
    new_df      = pd.read_csv(NEW_DATA_PATH)
    combined_df = pd.concat([original_df, new_df], ignore_index=True)

    original_rows = len(original_df)
    new_rows      = len(new_df)
    combined_rows = len(combined_df)

    print(f"Original rows : {original_rows}")
    print(f"New rows      : {new_rows}")
    print(f"Combined rows : {combined_rows}")

    # ── Fixed test split (same random_state=42 as Tasks 1 & 2) ───────────
    X_orig = original_df[FEATURES]
    y_orig = original_df[TARGET]
    X_orig_train, X_test, y_orig_train, y_test = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=42
    )

    # ── Champion: train on original training split ────────────────────────
    champion_model = build_fresh_model(champion_name)
    champion_model.fit(X_orig_train, y_orig_train)
    champion_rmse = round(rmse(y_test, champion_model.predict(X_test)), 4)
    print(f"\nChampion RMSE  : {champion_rmse}")

    # ── Retrained: use combined data MINUS the held-out test indices ──────
    test_indices      = X_test.index.tolist()
    combined_train_df = combined_df.drop(
        index=[i for i in test_indices if i in combined_df.index], errors="ignore"
    )
    X_combined_train = combined_train_df[FEATURES]
    y_combined_train = combined_train_df[TARGET]

    retrained_model = build_fresh_model(champion_name)

    # ── MLflow tracking ───────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLRUNS_DIR)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="retraining-pipeline"):
        mlflow.log_param("champion_name",       champion_name)
        mlflow.log_param("original_data_rows",  original_rows)
        mlflow.log_param("new_data_rows",       new_rows)
        mlflow.log_param("combined_data_rows",  combined_rows)
        mlflow.log_param("min_improvement",     MIN_IMPROVEMENT)

        retrained_model.fit(X_combined_train, y_combined_train)
        retrained_rmse = round(rmse(y_test, retrained_model.predict(X_test)), 4)

        improvement = round(champion_rmse - retrained_rmse, 4)
        action = "promoted" if improvement >= MIN_IMPROVEMENT else "kept_champion"

        mlflow.log_metric("champion_rmse",  champion_rmse)
        mlflow.log_metric("retrained_rmse", retrained_rmse)
        mlflow.log_metric("improvement",    improvement)
        mlflow.log_param("action",          action)
        mlflow.sklearn.log_model(retrained_model, artifact_path="retrained_model")

    # ── Promote if warranted ──────────────────────────────────────────────
    if action == "promoted":
        joblib.dump(retrained_model, os.path.join(MODEL_DIR, "tuned_model.pkl"))
        print("🚀  Retrained model PROMOTED → models/tuned_model.pkl")
    else:
        print(f"🏆  Champion kept (improvement = {improvement}, threshold = {MIN_IMPROVEMENT})")

    # ── Write results JSON ────────────────────────────────────────────────
    output = {
        "original_data_rows":       original_rows,
        "new_data_rows":            new_rows,
        "combined_data_rows":       combined_rows,
        "champion_rmse":            champion_rmse,
        "retrained_rmse":           retrained_rmse,
        "improvement":              improvement,
        "min_improvement_threshold": MIN_IMPROVEMENT,
        "action":                   action,
        "comparison_metric":        "rmse",
    }

    out_path = os.path.join(RESULT_DIR, "step4_s8.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅  Task 4 complete.")
    print(f"    Action      : {action}")
    print(f"    Improvement : {improvement}")
    print(f"    Output      : {out_path}")


if __name__ == "__main__":
    main()
