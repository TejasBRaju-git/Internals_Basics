"""
Task 3 — Docker CLI Predictor (8 marks)
========================================
Usage (inside container):
  python src/predict_cli.py \
      --course_hours 76 \
      --quizzes_count 24 \
      --difficulty_level 3 \
      --learner_experience 2

Prefers tuned_model.pkl (Task 2); falls back to best_model.pkl (Task 1).
Prints prediction as JSON and writes results/step3_s3.json when the
canonical test input is used.
"""

import argparse
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

TUNED_PATH    = os.path.join(MODEL_DIR, "tuned_model.pkl")
BASELINE_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

FEATURES = ["course_hours", "quizzes_count", "difficulty_level", "learner_experience"]

# Canonical test input defined in the question paper
CANONICAL = {
    "course_hours":       76.0,
    "quizzes_count":      24.0,
    "difficulty_level":   3.0,
    "learner_experience": 2.0,
}


def load_model():
    if os.path.exists(TUNED_PATH):
        return joblib.load(TUNED_PATH)
    if os.path.exists(BASELINE_PATH):
        return joblib.load(BASELINE_PATH)
    raise FileNotFoundError(
        "No trained model found in models/. "
        "Run  python src/train.py  (and optionally  python src/tune.py) first."
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="EduTrack — predict course completion days"
    )
    p.add_argument("--course_hours",       type=float, required=True,
                   help="Total course hours (10–200)")
    p.add_argument("--quizzes_count",      type=float, required=True,
                   help="Number of quizzes (5–50)")
    p.add_argument("--difficulty_level",   type=float, required=True,
                   help="Difficulty level (1–5)")
    p.add_argument("--learner_experience", type=float, required=True,
                   help="Learner experience level (1–5)")
    return p.parse_args()


def main():
    args  = parse_args()
    model = load_model()

    input_dict = {
        "course_hours":       args.course_hours,
        "quizzes_count":      args.quizzes_count,
        "difficulty_level":   args.difficulty_level,
        "learner_experience": args.learner_experience,
    }

    # Build a DataFrame so feature names match training
    X = pd.DataFrame([input_dict], columns=FEATURES)
    prediction = round(float(model.predict(X)[0]), 4)

    result = {
        "image_name": "edutrack-predictor",
        "image_tag":  "v1",
        "base_image": "python:3.10-slim",
        "test_input": input_dict,
        "prediction": prediction,
    }

    # Always print to stdout
    print(json.dumps(result, indent=2))

    # Write step3_s3.json when called with the canonical test values
    is_canonical = all(
        abs(input_dict[k] - CANONICAL[k]) < 1e-6 for k in CANONICAL
    )
    if is_canonical:
        out_path = os.path.join(RESULT_DIR, "step3_s3.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[info] step3_s3.json written → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
