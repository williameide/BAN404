# BAN404 Curriculum × Exam Map

> **How to use:** Find a topic, read its likelihood column, and match to your study priority. "HIGH" = nearly guaranteed; always prepare fully. "MED" = prepare a solid fallback. "LOW" = know the concept for written theory; unlikely to need R code.

---

## 1. Exam Format Reference

| Year | Type | Hours | Task 1 focus | Task 2 focus | Dataset |
|------|------|-------|--------------|--------------|---------|
| 2021 | Home | 8 | Bootstrap (SE, CI) + linear regression | Insurance claims prediction | `Smarket`, `dataCar` (ISLR/insuranceData) |
| 2022 | Home | 8 | Classification tree + threshold CV | GAM + bagging (regression) | `OJ` (ISLR), `Computers` (Ecdat) |
| 2023 | Home | 8 | OLS + Lasso + trees + RF (regression) | Bootstrap CI + logistic + RF (classification) | `Churn.csv` |
| 2024 | School | 6 | KNN local regression + LOOCV + backfitting | Airline satisfaction (two perspectives) | `airline.csv` |
| 2025 | School | 6 | Ridge + LOOCV + Bootstrap + GAM | Insurance claim prediction | `data_task1.csv` + `dataOhlsson` |
| **2026** | **School** | **6** | **Likely: explain code + tune method** | **Likely: EDA → logistic/tree → RF/boosting** | **Unknown CSV + package dataset** |

**Key 2026 constraint:** R retake version. Allowed: all written self-made material (no laptop restriction on notes, snippets pre-prepared). 6 hours. Submit `.rmd` or `.qmd`.

---

## 2. Topic × Exam Frequency Table

| Topic | Tested in | Task type | Points trend | Likelihood 2026 |
|-------|-----------|-----------|-------------|-----------------|
| **Bootstrap** (SE, CI, histogram) | 2021, 2023, 2025 | Task 1 code-explain + code-write | 10–15p | **HIGH** |
| **LOOCV / K-fold CV** | 2021, 2022, 2024, 2025 | Task 1 explain+use / Task 2 tuning | 10p | **HIGH** |
| **Ridge Regression** (code, tune, compare OLS) | 2025 | Task 1 all sub-parts | 30p | **HIGH** |
| **Lasso Regression** | 2023 | Task 1d–e | 10p | **HIGH** |
| **GAM / backfitting** | 2021, 2022, 2024, 2025 | Task 1 or Task 2 | 7–10p | **HIGH** |
| **OLS / linear regression** | 2022, 2023, 2025 | Baseline model + compare | 5–8p | **HIGH** |
| **Logistic Regression** | 2021, 2022, 2023, 2024, 2025 | Task 2 always | 10p | **HIGH** |
| **Random Forest** | 2022, 2023, 2024, 2025 | Task 2 always | 7–10p | **HIGH** |
| **Boosting (gbm)** | 2021, 2025 | Task 2 | 10p | **HIGH** |
| **Classification Tree** | 2022, 2024 | Task 2 | 5–8p | **HIGH** |
| **Regression Tree** | 2023 | Task 1 | 5–8p | **MED-HIGH** |
| **Confusion Matrix + Threshold** | 2021, 2022, 2023, 2024, 2025 | Task 2 evaluation | 5p | **HIGH** |
| **Descriptive Statistics / EDA** | ALL years | Task 2a always | 3–10p | **HIGH** |
| **Train/test split + data cleaning** | ALL years | Task 2a always | 3–5p | **HIGH** |
| **KNN (local regression)** | 2024 | Task 1 | 10p | **MED** |
| **Bagging** | 2022, 2024 | Task 2 | 5–10p | **MED** |
| **Variable Importance (RF/bagging)** | 2022, 2023, 2024 | Task 2 | 5p | **MED** |
| **R² / MSE computation** | 2025 (T1e) | Task 1 | 5p | **MED** |
| **Backfitting theory (explain)** | 2022 (T2f), 2024 (T1e) | Task 1 theory | 7p | **MED** |
| **Bagging theory (explain)** | 2022 (T2h) | Task 1 theory | 7p | **MED** |
| **Smoothing splines / polynomial** | 2021, 2022 | Task 1 or 2 | 7p | **LOW-MED** |
| **SVM** | not tested in exams | — | — | **LOW** |
| **PCA / PCR** | not tested in exams | — | — | **LOW** |
| **LDA / QDA** | not tested in exams | — | — | **LOW** |
| **K-means clustering** | not tested in exams | — | — | **LOW** |

---

## 3. Recurring Task Structures

### Task 1 pattern (50p, methodological)
Every exam since 2023 gives you **R functions to explain**, then asks you to:
1. **Explain what the function does** (name the method, state the formula, describe each line)
2. **Use the function** on given data (LOOCV loop, bootstrap loop, etc.)
3. **Extend or modify** (tune λ, add standardisation, compare to OLS)
4. **Optional follow-up** (bootstrap CI, GAM R², etc.)

**2024:** `f` was KNN local regression → explain → LOOCV for K → extend to multi-predictor → backfitting theory
**2025:** `f`+`g` was ridge → explain → compare OLS vs ridge → LOOCV for λ → bootstrap variance → GAM

### Task 2 pattern (50p, real data)
1. **Data cleaning** – factor encoding, remove leakage variables, train/test split
2. **EDA** – boxplots, cross-tables, prop.tables to find useful predictors
3. **Baseline model** – logistic regression (classification) or OLS/Lasso (regression)
4. **Evaluate** – confusion matrix, accuracy, or MSE on test data; threshold discussion
5. **Ensemble model** – random forest + variable importance
6. **Second ensemble** – boosting OR bagging
7. **Interpretation** – "What predicts Y?" written business answer

---

## 4. Methods by Priority

### TIER 1 — Must master completely (code + theory + interpretation)
- Ridge (objective function, demeaning, λ tuning via LOOCV)
- Bootstrap (SE, CI normal + percentile, histogram)
- LOOCV (inner loop structure, why demean, how to tune)
- Logistic regression (fit, interpret, threshold, confusion matrix)
- Random Forest (fit, mtry, variable importance, evaluate)
- Boosting/gbm (bernoulli/gaussian, n.trees, depth, shrinkage, evaluate)
- GAM with smoothing splines (detect nonlinearity, fit, compare MSE/R²)

### TIER 2 — Prepare well (likely appears, may be worth 10–15p)
- KNN local regression (distance, `order`, LOOCV for K)
- Backfitting theory (explain in plain English, iteration logic)
- Lasso (abs penalty, glmnet, variable selection)
- Regression/classification tree (fit, prune, interpret, plot)
- Confusion matrix + threshold tuning (cross-validation over thresholds)
- Bagging (bootstrap + average, variable importance)

### TIER 3 — Know for theory, skip code depth
- SVM, PCA/PCR, LDA/QDA, K-means
- Polynomial regression, step functions
- Naive Bayes

---

## 5. High-Probability 2026 Exam Scenarios

**Most likely Task 1 (code-explain format):**
- Option A: Lasso objective + LOOCV for λ + bootstrap for a coefficient SE
- Option B: KNN with multi-predictor distance + LOOCV + backfitting explanation
- Option C: Ridge (again) with different data + bootstrap for a different statistic + GAM

**Most likely Task 2 (dataset analysis):**
- Binary classification on a new CSV (employee turnover, loan default, medical event, etc.)
- EDA → logistic → RF → boosting pipeline (same as 2025, 2024, 2023 Task 2)
- Possible twist: regression task (predict continuous outcome like 2023 Task 1)
