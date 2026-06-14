# Hackathon Project — Rapid ML Prototype

> **Time-boxed hackathon solution built under competitive constraints, demonstrating rapid problem framing, fast baseline construction, and iterative model improvement within strict time limits.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

---

## Overview

This repository contains a solution developed during a competitive **ML hackathon**. The challenge tested the ability to rapidly understand an unfamiliar problem, build a working baseline quickly, and iteratively improve it within a strict time budget. The workflow prioritizes speed-to-submission over exhaustive experimentation.

---

## Hackathon Workflow

```
T+0:00  Problem statement released
   ↓
T+0:30  EDA complete, baseline model running
   ↓
T+1:30  Feature engineering, model selection
   ↓
T+2:30  Hyperparameter tuning, ensemble
   ↓
T+3:00  Final submission
```

---

## Rapid Baseline Strategy

### Step 1: Fast EDA (30 minutes)
```python
print(df.info())           # dtypes, nulls
print(df.describe())       # distributions
print(df.isnull().sum())   # missing values
print(df['target'].value_counts())  # class balance
```

### Step 2: Instant Baseline
```python
# Fill all nulls with median (fast, good enough for baseline)
df.fillna(df.median(), inplace=True)

# Encode all categoricals with label encoding
for col in df.select_dtypes('object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# XGBoost default — usually top-3 algorithm without tuning
model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### Step 3: Feature Importance Sprint
```python
# Drop features with near-zero importance
importances = model.feature_importances_
drop_cols = [col for col, imp in zip(features, importances) if imp < 0.001]
X_train.drop(columns=drop_cols, inplace=True)
```

### Step 4: Quick Ensemble
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier([
    ('xgb', XGBClassifier(n_estimators=200)),
    ('rf', RandomForestClassifier(n_estimators=200)),
    ('lgbm', LGBMClassifier(n_estimators=200))
], voting='soft')
```

---

## Key Lessons

1. **Baseline first, optimize second** — a working XGBoost default beats a perfect unfinished model
2. **Nulls: median/mode imputation is fast and 90% as good as complex imputation in time-boxed settings**
3. **Feature importance culling** removes noise quickly and often improves CV score
4. **Soft-voting ensemble** of 3 diverse models almost always beats any single model
5. **Monitor leaderboard CV gap** — if local CV >> leaderboard, you have a data leakage issue

---

## Installation

```bash
git clone https://github.com/tamer017/Hackathon.git
cd Hackathon
pip install scikit-learn xgboost lightgbm pandas numpy jupyter
jupyter notebook
```

---

## Skills & Concepts

`Rapid Prototyping` `Hackathon` `XGBoost` `LightGBM` `Ensemble Methods` `Feature Engineering` `Time-Constrained ML` `Voting Classifier` `EDA` `Competitive ML`

---

## Author

**Ahmed Tamer Assy** — [GitHub](https://github.com/tamer017) | Machine Learning Researcher @ Volkswagen AG
