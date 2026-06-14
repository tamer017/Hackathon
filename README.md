# Agricultural Crop Yield Prediction — Hackathon ML Pipeline

> A machine learning pipeline for predicting **crop yields** from agricultural data (Rajasthan, India 2018-2019) using an ensemble of gradient boosting, random forest, and XGBoost models.

[![Model](https://img.shields.io/badge/Model-XGBoost%20%2B%20RandomForest-blue?style=flat-square)]()
[![Dataset](https://img.shields.io/badge/Dataset-Rajasthan%20Agriculture%202018--19-green?style=flat-square)]()
[![Language](https://img.shields.io/badge/Language-Python%2FJupyter-orange?style=flat-square)]()

---

## Overview

This project was developed during a hackathon to tackle a real-world agricultural prediction challenge. Using the **Agricultural Data for Rajasthan, India (2018-2019)** dataset — which spans crop production records, soil analysis data, and water usage metrics — the pipeline trains and evaluates multiple regression models to predict crop yields.

The goal is to provide actionable insights for **farmers and agricultural policymakers**, enabling data-driven decisions about crop selection, irrigation planning, and resource allocation to improve food security.

---

## Dataset

| File | Contents |
|---|---|
| `crop_production_data.csv` | Historical crop yield per district, crop type, and season |
| `soil_analysis_data.csv` | Soil pH, nitrogen content, phosphorus, potassium per area |
| `water_usage_data.csv` | Irrigation water consumption per crop and district |

**Scope:** Rajasthan state, India | **Period:** 2018–2019 | **Granularity:** District-level

---

## Pipeline Architecture

```
[Multi-source CSV Input]
      crop_production_data.csv
      soil_analysis_data.csv
      water_usage_data.csv
              |
              v
   [loadData() — Merge on district key]
              |
              v
   [Preprocessing]
   • OneHotEncoder for categorical variables
     (crop type, district, season)
   • StandardScaler for numeric features
   • Missing value handling with missingno diagnostics
              |
              v
   [Model Training & Evaluation]
   ├── LinearRegression          (baseline)
   ├── RandomForestRegressor     (ensemble, n=100)
   ├── GradientBoostingRegressor (sklearn)
   └── XGBRegressor              (xgboost)
              |
              v
   [Results: RMSE, R² per model + Feature Importance]
```

---

## Technical Highlights

### Multi-Dataset Merge
The `loadData()` function performs a multi-key join across the three source datasets on district and year identifiers, producing a unified feature matrix for model training.

### Preprocessing Stack
- **`OneHotEncoder`:** Handles high-cardinality categorical features (crop species, district names, season type)
- **`StandardScaler`:** Normalizes numeric features (rainfall mm, temperature °C, soil NPK levels) for scale-sensitive models
- **`missingno`:** Visual diagnostics for missingness patterns before imputation

### Model Comparison
| Model | Type | Key Advantage |
|---|---|---|
| LinearRegression | Linear | Interpretable baseline |
| RandomForestRegressor | Ensemble (bagging) | Handles non-linearity |
| GradientBoostingRegressor | Ensemble (boosting) | Reduces bias iteratively |
| XGBRegressor | Optimized boosting | Speed + regularization |

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/tamer017/Hackathon.git
cd Hackathon

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost missingno jupyter

# Launch the notebook
jupyter notebook main.ipynb
```

> **Note:** Ensure `crop_production_data.csv`, `soil_analysis_data.csv`, and `water_usage_data.csv` are in the root directory.

---

## Skills Demonstrated

- **Machine Learning:** XGBoost, RandomForest, GradientBoosting, regression modeling
- **Data Engineering:** Multi-file dataset merging, categorical encoding, feature scaling
- **Domain Knowledge:** Agricultural analytics, crop yield modeling, soil science features
- **Python Stack:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, missingno
- **Rapid Prototyping:** Hackathon-paced end-to-end ML pipeline development

---

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Rajasthan Agriculture Department Data Portal](https://agriculture.rajasthan.gov.in/)
