# Agricultural Water Consumption Optimization — Hackathon ML Solution

> **Time-boxed hackathon solution for the [Agricultural Data for Rajasthan, India (2018-2019)](https://www.kaggle.com/datasets/suraj520/agricultural-data-for-rajasthan-india-2018-2019) challenge — rapid problem framing, fast baseline construction, and iterative model improvement under strict time limits.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.6+-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This project aims to **optimize water consumption in agriculture** by developing an AI solution based on the agricultural dataset for Rajasthan, India. The solution uses machine learning techniques to predict and optimize water usage based on soil analysis and crop production data — developed under competitive hackathon constraints.

The dataset includes information on crop production, soil analysis, and water usage, covering the 2018–2019 agricultural season for the state of Rajasthan, India.

---

## Dataset

- **Source:** [Agricultural Data for Rajasthan, India (2018-2019)](https://www.kaggle.com/datasets/suraj520/agricultural-data-for-rajasthan-india-2018-2019) on Kaggle
- **Features:** Crop production records, soil analysis measurements, water usage statistics
- **Preprocessing:** Merged multi-source dataset, handled missing values, encoded categoricals

---

## Project Structure

```
├── data/               # Dataset files (crop production, soil analysis, water usage)
├── notebooks/          # Jupyter notebooks for EDA, preprocessing, and modeling
├── src/
│   ├── data_preprocessing.py   # Loading, preprocessing, and merging the dataset
│   ├── modeling.py             # ML models, training and evaluation functions
│   └── utils.py                # Utility functions
├── results/            # Model evaluation results and visualizations
└── README.md
```

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

## Requirements

```bash
git clone https://github.com/tamer017/Hackathon.git
cd Hackathon
pip install -r requirements.txt
jupyter notebook
```

Dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `category_encoders`, `missingno`

---

## Results

Model evaluation results and visualizations are stored in the `results/` directory. Full findings can be found in the [project report](report/report.pdf).

---

## Skills & Concepts

`Rapid Prototyping` `Hackathon` `XGBoost` `LightGBM` `Ensemble Methods` `Feature Engineering` `Time-Constrained ML` `Voting Classifier` `EDA` `Agricultural ML` `Competitive ML` `Water Optimization`

---

## Contributors

- Ahmed Tamer Assy — [GitHub](https://github.com/tamer017)

---

## Acknowledgements

Dataset provided by [Suraj520](https://www.kaggle.com/datasets/suraj520/agricultural-data-for-rajasthan-india-2018-2019) on Kaggle.

---

## Author

**Ahmed Tamer Assy** — [GitHub](https://github.com/tamer017) | Machine Learning Researcher @ Volkswagen AG
