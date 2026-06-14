# 🛡️ Fraud Detection System
> A two-layer fraud detection system built with XGBoost and a rule-based scoring engine.

**Author:** Your Name  
**Institution:** JUNIA ISEN — MSc Big Data  
**Date:** May 2026

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-red.svg)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-green.svg)](https://streamlit.io)

---

## Overview

Most fraud detection systems only check whether a credit card is stolen.
This project goes further by building **two independent layers** of defence:

- **Layer 1 — Credit Card Fraud:** An XGBoost ML model trained on 284,807 real
transactions that detects stolen cards at payment time using transaction patterns,
velocity, and card signals.

- **Layer 2 — Order Fraud:** A rule-based scoring engine that catches fraudulent
orders even when a stolen card passes Layer 1, by analysing behavioural signals
like account age, email type, shipping destination, and device patterns.

> In a production system, both layers would trigger automatically on every
> transaction. This demo allows independent testing of each layer for
> demonstration purposes.

---

## Live Demo

> 🚀 [Live Demo](https://fraud-detection-n6gn77sejezzadho7wqlkj.streamlit.app)

---
## Project Structure
fraud-detection/

├── data/

│   └── creditcard.csv          # Kaggle credit card fraud dataset

├── notebooks/

│   ├── 00_setup.ipynb          # Environment setup and verification

│   ├── 01_eda.ipynb            # Exploratory data analysis

│   ├── 02_preprocessing.ipynb  # Scaling, train/test split, SMOTE

│   ├── 03_modelling.ipynb      # Model training and evaluation

│   └── 04_order_fraud.ipynb    # Layer 2 rule-based scoring engine

├── models/

│   ├── fraud_model.pkl         # Trained XGBoost model

│   ├── threshold.pkl           # Optimal decision threshold

│   └── feature_columns.pkl     # Feature column names

├── app/

│   └── app.py                  # Streamlit web application

└── README.md
---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Transactions | 284,807 |
| Fraud cases | 492 (0.173%) |
| Features | 30 (V1–V28 PCA + Amount + Time) |
| Missing values | None |

**Key finding:** The dataset is severely imbalanced — a model that predicts
"legitimate" every time would be 99.8% accurate but completely useless.
This is why we use F1-score and AUC-ROC as our evaluation metrics, not accuracy.

---
## Methodology

### Layer 1 — Credit Card Fraud Detection

**Problem:** 492 fraud cases vs 284,315 legitimate — 0.173% fraud rate.

**Approach:**
1. Scale `Amount` and `Time` using StandardScaler
2. Stratified 80/20 train/test split — preserves fraud ratio in both sets
3. Apply SMOTE to training set only — balances fraud/legit from 394 vs 227,451 to 227,451 vs 227,451
4. Train and compare three models

**Results:**

| Model | AUC-ROC | F1 Score | Recall | Precision |
|---|---|---|---|---|
| Logistic Regression | 0.9698 | 0.1094 | 0.920 | 0.058 |
| **XGBoost** | **0.9771** | **0.2698** | **0.888** | **0.159** |
| Isolation Forest | N/A | 0.0000 | 0.000 | 0.000 |

**Winner: XGBoost** — highest AUC-ROC and F1 score.

**Most important feature: V14** — confirmed by both correlation analysis
in EDA and XGBoost feature importance. When V14 is low, fraud probability
rises sharply.

---

### Layer 2 — Order Fraud Detection

A rule-based scoring engine that assigns a risk score (0–100) based on
behavioural signals at the order level.

| Signal | Max points | Risk level |
|---|---|---|
| Linked to prior fraud | 25 | Critical |
| Account created today | 22 | High |
| Shipping to high-risk country | 22 | High |
| Multiple cards tried | 20 | High |
| Freight forwarder address | 20 | High |
| IP velocity (5+ orders today) | 20 | High |
| Disposable email | 18 | High |
| Address added recently | 18 | High |
| VPN / proxy detected | 15 | High |
| Name mismatch | 15 | High |

**Decision thresholds:**
- Score 0–34 → **APPROVE**
- Score 35–64 → **REVIEW**
- Score 65–100 → **DECLINE**

---

### Combined Score

| Layer | Weight | Reason |
|---|---|---|
| Layer 1 — XGBoost | 60% | Trained on real transaction data — more precise |
| Layer 2 — Rule engine | 40% | Catches behavioural fraud ML cannot see |

**Key scenario:** A stolen card with a low Layer 1 score (looks clean)
but high Layer 2 score (suspicious behaviour) still gets flagged for REVIEW.
This is the core value of the two-layer architecture.

---
## EDA Findings

| Finding | Implication |
|---|---|
| 492 fraud out of 284,807 (0.173%) | Severe class imbalance — cannot use accuracy as metric |
| Fraud median amount is $9.25 vs $22 legit | Fraudsters test stolen cards with small amounts first |
| Fraud spikes at hours 1–2am | Fraudsters operate when cardholders are asleep |
| V17, V14, V12, V10 strongly negative | Key features — model relies on these most |
| V11, V4, V2 positive correlation | Secondary fraud signals |
| Zero missing values | No cleaning needed — go straight to preprocessing |

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Fio90/fraud-detection.git
cd fraud-detection
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn joblib shap streamlit
```

### 3. Download the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and place it in the `data/` folder.

### 4. Run the notebooks in order
```
00_setup.ipynb
01_eda.ipynb
02_preprocessing.ipynb
03_modelling.ipynb
04_order_fraud.ipynb
```

### 5. Launch the Streamlit app
```bash
cd app
py -m streamlit run app.py
```

---

## Key Learnings

- **Class imbalance is the core challenge** — accuracy is a misleading metric
when 99.8% of data is one class
- **SMOTE must only be applied to training data** — applying it to test data
would give falsely optimistic results
- **Unsupervised methods struggle here** — Isolation Forest failed because
SMOTE-generated fraud samples do not look like natural anomalies
- **Two layers catch what one cannot** — a card that passes ML scoring can
still be caught by behavioural order signals
- **V14 is the single most powerful feature** — confirmed independently by
both correlation analysis and XGBoost feature importance

---

## Future Improvements

- [ ] Train on IEEE-CIS Fraud Detection dataset with richer order-level features
- [ ] Add SHAP explainability to show which features drove each prediction
- [ ] Build a real-time scoring pipeline using Kafka or Spark Streaming
- [ ] Connect Layer 1 and Layer 2 into a single automated pipeline
- [ ] Add a feedback loop — let analysts mark false positives to retrain the model

---

## Author

**Frederick Amartey-Fio**  
MSc Big Data — JUNIA ISEN  
[LinkedIn](https://www.linkedin.com/in/famarteyfio/) | [GitHub](https://github.com/Fio90)