# MLX 2.0 Regression Challenge
> Predicting Song Popularity Scores using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)](https://xgboost.readthedocs.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-MLX%202.0-20BEFF)](https://www.kaggle.com/)

---

## 📌 Overview

This project is a solution for the **MLX 2.0 Regression Challenge** hosted on Kaggle as part of the CS3111 Introduction to Machine Learning course. The objective is to predict a song's **popularity score (0–100)** using real-world music data including audio features, artist statistics, and track metadata from Billboard-tracked releases.

This is a **supervised regression** problem — the model must output a continuous numerical value rather than a category. Two machine learning approaches were implemented, evaluated, and submitted to Kaggle.

**Competition Metric:** Root Mean Squared Error (RMSE) — lower is better.

---

## 📁 File Structure

```
mlx-regression/
│
├── data/                              # Raw dataset files (from Kaggle)
│   ├── train.csv                      # 61,609 songs with known popularity scores (62 columns)
│   ├── test.csv                       # 41,074 songs to predict (61 columns, no target)
│   └── sample_submission.csv          # Example of required submission format
│
├── submissions/                       # Generated prediction files ready for Kaggle upload
│   ├── submission1_random_forest.csv  # Predictions from Approach 1
│   └── submission2_xgboost.csv        # Predictions from Approach 2
│
├── approach1_random_forest.py         # Approach 1: Random Forest Regressor
├── approach2_xgboost.py               # Approach 2: XGBoost Regressor
└── README.md                          # This file
```

---

## 📊 Dataset Description

Each row in the dataset represents a single music release. The features are grouped into four categories:

| Category | Features |
|----------|----------|
| **Track Identification** | `track_identifier`, `creator_collective`, `publication_timestamp`, `album_component_count` |
| **Core Audio Features** | `rhythmic_cohesion` (danceability), `intensity_index` (energy), `organic_texture` (acousticness), `beat_frequency` (tempo), `harmonic_scale` (key), `tonal_mode`, `duration_ms`, `time_signature` |
| **Derived Audio Metrics** | `emotional_charge` (valence × energy), `groove_efficiency` (energy/danceability), `organic_immersion` (acousticness × duration), `duration_consistency`, `tempo_volatility`, `key_variety` |
| **Contextual Features** | `album_name_length`, `artist_count`, `weekday_of_release`, `season_of_release`, `lunar_phase` |

- **Target variable:** `target` — continuous popularity score (0–100), present only in `train.csv`
- **Missing values:** represented as empty cells or `NaN`
- **Categorical features:** stored as strings, require encoding before model training

---

## ⚙️ Requirements

### Python Version
Python 3.11 or higher is recommended.

### Dependencies

Install all required libraries with:

```bash
pip install pandas numpy scikit-learn xgboost
```

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Random Forest model, preprocessing, evaluation metrics |
| `xgboost` | XGBoost model |

> **Mac users only:** XGBoost requires the OpenMP runtime library. If you get an import error, run:
> ```bash
> brew install libomp
> ```
> If Homebrew is not installed, first run:
> ```bash
> /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
> ```

---

## 🔧 Data Preprocessing

Both scripts apply the same preprocessing pipeline before training:

1. **Label Encoding** — Text columns (song names, artist names, weekday labels, etc.) are converted to integers using `LabelEncoder`. The encoder is fitted on the combined train and test data to ensure consistent mapping across both datasets.

2. **Force Numeric Conversion** — All columns are explicitly cast to numeric types using `pd.to_numeric(errors='coerce')`. Any value that cannot be converted becomes `NaN`.

3. **Infinity Handling** — Derived metrics such as `groove_efficiency` (energy ÷ danceability) can produce infinity values when danceability is zero. All `inf` and `-inf` values are replaced with `NaN`.

4. **Missing Value Imputation** — All remaining `NaN` values are filled with the **column mean calculated from the training data**. The same training means are applied to the test set to prevent data leakage.

5. **Train-Validation Split** — Training data is split 80/20 into a training set (49,287 samples) and a local validation set (12,322 samples) using `random_state=42` for reproducibility.

---

## 🚀 How to Run

Make sure you are inside the `mlx-regression/` directory in your terminal before running either script.

### Approach 1 — Random Forest

```bash
python approach1_random_forest.py
```

**What it does:**
- Loads and preprocesses `train.csv` and `test.csv`
- Trains a `RandomForestRegressor` with 100 trees on 80% of training data
- Evaluates on the 20% validation set and prints RMSE, MAE, and R²
- Retrains on the full training data for best performance
- Saves predictions to `submissions/submission1_random_forest.csv`

**Expected output:**
```
Loading data...
Train shape: (61609, 62)
Test shape:  (41074, 61)
Encoding text columns...
Filling missing values...
Training samples:   49287
Validation samples: 12322
Training Random Forest model...
Training complete!
--- Validation Results (Approach 1: Random Forest) ---
RMSE (lower is better): 11.3068
MAE  (lower is better): 6.6698
R²   (higher is better, max=1): 0.7255
Submission saved to: submissions/submission1_random_forest.csv
Total predictions: 41074
```

---

### Approach 2 — XGBoost

```bash
python approach2_xgboost.py
```

**What it does:**
- Applies the same preprocessing pipeline as Approach 1
- Trains an `XGBRegressor` with 500 trees, learning rate 0.05, max depth 6
- Evaluates on the 20% validation set and prints RMSE, MAE, and R²
- Retrains on the full training data and saves predictions to `submissions/submission2_xgboost.csv`

**Expected output:**
```
Loading data...
Train shape: (61609, 62)
Test shape:  (41074, 61)
Encoding text columns...
Filling missing values...
Training samples:   49287
Validation samples: 12322
Training XGBoost model...
Training complete!
--- Validation Results (Approach 2: XGBoost) ---
RMSE (lower is better): 12.8336
MAE  (lower is better): 9.0323
R²   (higher is better, max=1): 0.6464
Submission saved to: submissions/submission2_xgboost.csv
Total predictions: 41074
```

---

## 📈 Results

### Local Validation (20% holdout set)

| Model | RMSE ↓ | MAE ↓ | R² ↑ |
|-------|--------|-------|------|
| Random Forest | 11.3068 | 6.6698 | 0.7255 |
| XGBoost | 12.8336 | 9.0323 | 0.6464 |

### Kaggle Leaderboard

| Model | Public RMSE ↓ | Private RMSE ↓ |
|-------|--------------|----------------|
| Random Forest | **10.8188** | **10.7959** |
| XGBoost | 12.5064 | 12.5300 |

**Random Forest achieved the best score** with a Kaggle public RMSE of **10.8188**, explaining approximately 72.5% of the variance in song popularity scores (R² = 0.7255).

Both models scored slightly better on Kaggle than on the local validation set, confirming that neither model overfitted to the training data.

---

## 🔮 Possible Improvements

- Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`
- Feature engineering from `publication_timestamp` (extract year, month, day of year)
- Experimenting with LightGBM or CatBoost
- Stacking or blending predictions from multiple models
- Dropping low-importance features to reduce noise
