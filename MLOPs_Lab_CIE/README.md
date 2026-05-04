# Internals_Basics — MLOPs_Lab_CIE

**USN:** 1BM23AI196 | **Code:** mlops-cie-170 | **Date:** 04 May 2026
**Course:** MLOps (24AM6AEMLO) | **Marks:** 30

---

## Repository Structure

```
Internals_Basics/
└── MLOPs_Lab_CIE/
    ├── data/
    │   ├── training_data.csv       ← 25 rows  (do not modify)
    │   └── new_data.csv            ← 20 rows  (do not modify)
    ├── src/
    │   ├── train.py                ← Task 1
    │   ├── tune.py                 ← Task 2
    │   ├── predict_cli.py          ← Task 3 (runs inside Docker)
    │   └── retrain.py              ← Task 4
    ├── models/                     ← auto-generated .pkl files
    ├── results/
    │   ├── step1_s1.json           ← Task 1 proof
    │   ├── step2_s2.json           ← Task 2 proof
    │   ├── step3_s3.json           ← Task 3 proof
    │   └── step4_s8.json           ← Task 4 proof
    ├── Dockerfile
    ├── requirements.txt
    └── .gitignore
```

---

## Setup in VSCode

### 1. Open the project
```
File → Open Folder → select  Internals_Basics/
```

### 2. Create & activate a virtual environment
Open the **Terminal** panel in VSCode (`Ctrl + `` ` ```) and run:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
cd MLOPs_Lab_CIE
pip install -r requirements.txt
```

---

## Run the Tasks (in order)

### Task 1 — Experiment Tracking & Model Comparison
```bash
python src/train.py
```
**What it does:**
- Trains **SVR** and **GradientBoosting** on `data/training_data.csv`
- Logs all hyperparams + MAE / RMSE / R² / MAPE to MLflow
- Tags runs: `experiment_type = "baseline_comparison"`
- Experiment: `"edutrack-completion-days"`
- Picks best model by MAE → saves `models/best_model.pkl`
- **Output:** `results/step1_s1.json`

---

### Task 2 — Hyperparameter Tuning
```bash
python src/tune.py
```
**What it does:**
- Grid-searches `n_estimators × learning_rate × max_depth` (27 trials)
- 3-fold CV per trial; each trial = nested MLflow run under `"tuning-edutrack"`
- Saves best tuned model → `models/tuned_model.pkl`
- **Output:** `results/step2_s2.json`

---

### Task 3 — Docker Packaging

**Step 1 — Build the image:**
```bash
docker build -t edutrack-predictor:v1 .
```

**Step 2 — Run the canonical test:**
```bash
docker run edutrack-predictor:v1 \
  --course_hours 76 \
  --quizzes_count 24 \
  --difficulty_level 3 \
  --learner_experience 2
```

> `results/step3_s3.json` is already pre-generated from local execution.
> Re-run `python src/predict_cli.py --course_hours 76 --quizzes_count 24 --difficulty_level 3 --learner_experience 2` to regenerate it locally.

---

### Task 4 — Retraining Pipeline
```bash
python src/retrain.py
```
**What it does:**
- Combines `training_data.csv` (25) + `new_data.csv` (20) → 45 rows
- Retrains champion model type on combined data
- Compares retrained vs champion on the same test split
- Promotes only if RMSE improves by ≥ 1.0
- **Output:** `results/step4_s8.json`

---

## Results Summary

| Task | File | Key Result |
|------|------|-----------|
| 1 | `step1_s1.json` | Best model: **GradientBoosting** (MAE = 12.2887) |
| 2 | `step2_s2.json` | Best: `n_est=100, lr=0.01, depth=3` (CV-MAE = 13.6867) |
| 3 | `step3_s3.json` | Prediction = **51.8029 days** |
| 4 | `step4_s8.json` | Action = **kept_champion** |

---

## Push to GitHub

### First time setup

```bash
# 1. Go to github.com → New repository
#    Name: Internals_Basics   Visibility: Public   (no README, no .gitignore)

# 2. In your terminal (inside the Internals_Basics/ folder):
git init
git branch -M main
git add .
git commit -m "MLOps CIE Lab — Tasks 1-4 complete (USN: 1BM23AI196)"

# 3. Link to your GitHub repo and push
git remote add origin https://github.com/TejasBRaju-git/Internals_Basics.git
git push -u origin main
```

> **Tip:** If prompted for credentials, use your GitHub **Personal Access Token**
> (Settings → Developer settings → Personal access tokens → Generate new token).

### Verify on GitHub
- Repo name: `Internals_Basics` ✓
- Folder inside: `MLOPs_Lab_CIE/` ✓
- Repo visibility: **Public** ✓
- All `results/*.json` files present ✓
