"""
Task 2 — Hyperparameter Tuning (8 marks)
=========================================
- Tunes the Task-1 winner (GradientBoosting) via grid search
- Parameter grid:
    n_estimators  : [100, 200, 300]
    learning_rate : [0.01, 0.05, 0.1]
    max_depth     : [3, 5, 7]
- 3-fold cross-validation per trial (27 trials total)
- Each trial logged as a NESTED MLflow run under parent "tuning-edutrack"
- Best config selected by CV-MAE
- Saves: results/step2_s2.json  |  models/tuned_model.pkl
"""

import os
import json
import itertools
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error

# ── Paths ─────────────────────────────────────────────────────────────────────
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
PARENT_RUN_NAME = "tuning-edutrack"
N_FOLDS         = 3

PARAM_GRID = {
    "n_estimators":  [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth":     [3, 5, 7],
}


def main():
    # ── Load & split data ─────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURES]
    y  = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Build all combinations ────────────────────────────────────────────
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    total  = len(combos)
    print(f"Grid search: {total} trials  ×  {N_FOLDS}-fold CV\n")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # ── MLflow setup ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLRUNS_DIR)
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_cv_mae  = float("inf")
    best_params  = {}
    best_model   = None

    with mlflow.start_run(run_name=PARENT_RUN_NAME) as parent_run:
        mlflow.log_param("search_type",   "grid")
        mlflow.log_param("n_folds",       N_FOLDS)
        mlflow.log_param("total_trials",  total)

        for i, combo in enumerate(combos, start=1):
            params = dict(zip(keys, combo))
            model  = GradientBoostingRegressor(random_state=42, **params)

            # ── Cross-validation ──────────────────────────────────────────
            fold_maes = []
            for train_idx, val_idx in kf.split(X_train):
                Xf_tr, Xf_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                yf_tr, yf_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                model.fit(Xf_tr, yf_tr)
                fold_maes.append(mean_absolute_error(yf_val, model.predict(Xf_val)))

            cv_mae = float(np.mean(fold_maes))

            # ── Log each trial as a nested run ────────────────────────────
            with mlflow.start_run(run_name=f"trial_{i:02d}", nested=True):
                for k, v in params.items():
                    mlflow.log_param(k, v)
                mlflow.log_metric("cv_mae", round(cv_mae, 4))

            print(f"  Trial {i:02d}/{total}  params={params}  cv_mae={cv_mae:.4f}")

            if cv_mae < best_cv_mae:
                best_cv_mae = cv_mae
                best_params = params
                # Re-fit on full training set with best params found so far
                best_model  = GradientBoostingRegressor(random_state=42, **params)

        # ── Train best on full training set, evaluate on held-out test ────
        best_model.fit(X_train, y_train)
        best_test_mae = float(mean_absolute_error(y_test, best_model.predict(X_test)))

        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_mae",   round(best_cv_mae,   4))
        mlflow.log_metric("best_test_mae", round(best_test_mae, 4))
        mlflow.sklearn.log_model(best_model, artifact_path="tuned_best_model")

    # ── Persist tuned model ───────────────────────────────────────────────
    joblib.dump(best_model, os.path.join(MODEL_DIR, "tuned_model.pkl"))

    # ── Write results JSON ────────────────────────────────────────────────
    output = {
        "search_type":     "grid",
        "n_folds":         N_FOLDS,
        "total_trials":    total,
        "best_params":     best_params,
        "best_mae":        round(best_test_mae, 4),
        "best_cv_mae":     round(best_cv_mae,   4),
        "parent_run_name": PARENT_RUN_NAME,
    }

    out_path = os.path.join(RESULT_DIR, "step2_s2.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅  Task 2 complete.")
    print(f"    Best params : {best_params}")
    print(f"    CV-MAE      : {round(best_cv_mae, 4)}")
    print(f"    Test-MAE    : {round(best_test_mae, 4)}")
    print(f"    Output      : {out_path}")


if __name__ == "__main__":
    main()
