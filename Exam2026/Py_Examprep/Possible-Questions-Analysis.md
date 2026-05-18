# BAN404 2026: Likely Exam Questions Based on `task1_data.csv`, `customer_data.csv`, and Prior Exams

## Scope and purpose

This note summarizes **high-probability question types** for BAN404 2026 based on:

- Prior exam style 2021–2025 (with stronger emphasis on 2024/2025 style)
- Current exam data files in this repository (`Exam2026/task1_data.csv`, `Exam2026/customer_data.csv`)
- Existing repository material (Kompendium, ExamPrep files, previous solved resources)

Use this as a reusable generator for future mock exams.

---

## 1) What the datasets naturally support

## `task1_data.csv` (continuous y, six x-variables)

Most likely themes:

1. **Regularisation theory + practice**
   - Explain Ridge vs LASSO objective and shrinkage behavior.
   - Fit OLS vs penalised model and compare coefficients.
   - Tune lambda with CV and interpret CV curve.

2. **Resampling methods**
   - Bootstrap distribution of `Var(y)` (or SD of `y`).
   - Bootstrap CI: normal approximation vs percentile.

3. **Non-linearity diagnostics**
   - Identify curved effects by scatter/loess.
   - Compare OLS vs GAM and discuss MSE/R² gains.

4. **Validation choice discussion**
   - Why K-fold over LOOCV for large n.
   - Leakage discussion in fold-specific preprocessing.

## `customer_data.csv` (binary churn)

Most likely themes:

1. **Applied binary prediction workflow**
   - Data cleaning and recoding factors.
   - Address perfect collinearity (minutes vs charges).
   - Train/test split and class imbalance interpretation.

2. **Interpretability track ("why churn?")**
   - Cross-tabs and grouped summaries.
   - Logistic regression coefficient interpretation (odds perspective).

3. **Prediction track ("who will churn?")**
   - Logistic + threshold tuning.
   - Random Forest + importance.
   - Boosting and staged performance/early stopping.

4. **Evaluation emphasis**
   - Accuracy alone is insufficient under class imbalance.
   - Sensitivity/specificity trade-off with business cost argument.

---

## 2) Recency-weighted likely structure (2024/2025 bias)

Highest-probability full exam structure:

- **Task 1 (methodology/theory with code reading):**
  - Explain a custom function (regularisation, CV, bootstrap, or KNN-like logic)
  - Apply it on `task1_data.csv`
  - Add one extension: GAM/non-linearity or tuning logic

- **Task 2 (applied modeling):**
  - A practical prediction assignment on `customer_data.csv`
  - Two lenses: explanation + prediction
  - Compare at least two model families (logit vs tree-ensemble)

This mirrors the 2024 methodology + applied split and the 2025 regularisation + insurance/churn style.

---

## 3) High-probability subquestion bank

Use these as interchangeable blocks when generating new mock exams.

### Task 1 bank

- Explain objective function and identify method (Ridge/LASSO).
- Show effect of increasing lambda on coefficients.
- Tune lambda with K-fold CV and plot test error.
- Explain why LOOCV may be inefficient here.
- Bootstrap variance/SD with CI and histogram.
- Diagnose non-linearity and improve with GAM.
- Compare training vs test perspective and justify chosen metric.

### Task 2 bank

- Recode factors and remove unrealistic predictors.
- Demonstrate and fix collinearity among charge/minute variables.
- Descriptive churn analysis via grouped rates and plots.
- Fit logistic regression and interpret 2–3 key coefficients.
- Pick threshold with explicit false-positive/false-negative tradeoff.
- Fit Random Forest; compute and interpret variable importance.
- Fit Boosting; select tree count via staged test performance.
- Provide final model recommendation under business constraints.

---

## 4) Model selection logic likely rewarded in grading

- Start from a transparent baseline (OLS/logit).
- Upgrade only when data pattern supports it (non-linearity, interactions).
- Keep evaluation aligned with objective:
  - explanatory question -> model interpretation + stable effects
  - prediction question -> held-out metrics + threshold strategy
- Explicitly mention assumptions and limitations.

---

## 5) Practical assumptions likely considered realistic

For prospective churn prediction, likely acceptable assumptions:

- Available at prediction time: subscription plan variables, historical usage summaries, service calls.
- Potentially not available: outcome-derived or post-event variables.
- Remove duplicated deterministic variables to reduce instability and improve interpretability.

---

## 6) How to reuse this file to build additional mock exams

1. Pick 4–5 blocks from Task 1 bank.
2. Pick 4–5 blocks from Task 2 bank.
3. Ensure one block demands explicit model comparison.
4. Ensure one block demands business-oriented threshold argument.
5. Keep point balance near 50/50 and include both code and interpretation requirements.

This gives an unlimited pipeline of realistic BAN404-style exams on the current datasets.

---
## 7) Master lookup: all mock-exam questions created so far (Exams 1–10)
This table gives a direct lookup of every subquestion used across all generated mock exams.

| Exam | Task | Q | Question title |
|---:|---|:--:|---|
| 1 | Task 1 | a | Decode the objective: Ridge variant (10 points) |
| 1 | Task 1 | b | Baseline vs shrinkage fit comparison (10 points) |
| 1 | Task 1 | c | Ten-fold CV tuning with robust choice rule (10 points) |
| 1 | Task 1 | d | Bootstrap uncertainty for variance (10 points) |
| 1 | Task 1 | e | Linear vs additive model improvement (10 points) |
| 1 | Task 2 | a | Prediction-ready data design (8 points) |
| 1 | Task 2 | b | EDA for churn risk framing (8 points) |
| 1 | Task 2 | c | Logistic model with FN-cost thresholding (12 points) |
| 1 | Task 2 | d | Random forest hyperparameter tuning (12 points) |
| 1 | Task 2 | e | Boosting selection and deployment recommendation (10 points) |
| 2 | Task 1 | a | Decode LASSO optimization code (10 points) |
| 2 | Task 1 | b | OLS vs LASSO coefficient path analysis (10 points) |
| 2 | Task 1 | c | Repeated K-fold tuning and LOOCV discussion (10 points) |
| 2 | Task 1 | d | Bootstrap uncertainty for SD(y) (10 points) |
| 2 | Task 1 | e | Polynomial vs GAM flexibility check (10 points) |
| 2 | Task 2 | a | Leakage-safe preprocessing and split (8 points) |
| 2 | Task 2 | b | Class imbalance and churn profile EDA (8 points) |
| 2 | Task 2 | c | Logistic threshold by expected cost (12 points) |
| 2 | Task 2 | d | Random forest vs logistic interpretation (12 points) |
| 2 | Task 2 | e | Gradient boosting for campaign targeting (10 points) |
| 3 | Task 1 | a | Identify Elastic Net objective mechanics (10 points) |
| 3 | Task 1 | b | Ridge vs LASSO vs Elastic Net comparison (10 points) |
| 3 | Task 1 | c | Validation discipline under preprocessing risk (10 points) |
| 3 | Task 1 | d | Bootstrap coefficient stability (10 points) |
| 3 | Task 1 | e | GAM smoothness interpretation (10 points) |
| 3 | Task 2 | a | Data preparation and deployment assumptions (8 points) |
| 3 | Task 2 | b | Segment-centric exploratory analysis (8 points) |
| 3 | Task 2 | c | Logistic baseline with odds interpretation (12 points) |
| 3 | Task 2 | d | Random forest interaction gains (12 points) |
| 3 | Task 2 | e | Boosting for final scorecard (10 points) |
| 4 | Task 1 | a | Explain K-fold CV pseudocode and leakage points (10 points) |
| 4 | Task 1 | b | Leakage-safe scaling inside folds (10 points) |
| 4 | Task 1 | c | Compare tuned Ridge and LASSO under common folds (10 points) |
| 4 | Task 1 | d | Bootstrap CI interpretation quality check (10 points) |
| 4 | Task 1 | e | Backfitting intuition and GAM implementation (10 points) |
| 4 | Task 2 | a | Data pipeline integrity (8 points) |
| 4 | Task 2 | b | EDA to model-hypothesis bridge (8 points) |
| 4 | Task 2 | c | Logistic model under threshold sweep (12 points) |
| 4 | Task 2 | d | Random forest robust tuning (12 points) |
| 4 | Task 2 | e | Boosting selection with early stopping logic (10 points) |
| 5 | Task 1 | a | Identify methods from function snippets (10 points) |
| 5 | Task 1 | b | Full regularization workflow (10 points) |
| 5 | Task 1 | c | Combine tuning and uncertainty evidence (10 points) |
| 5 | Task 1 | d | GAM enhancement + residual diagnostics (10 points) |
| 5 | Task 1 | e | Exam-style interpretation summary (10 points) |
| 5 | Task 2 | a | Build a production-ready churn dataset (8 points) |
| 5 | Task 2 | b | Risk-pattern EDA and intervention ideas (8 points) |
| 5 | Task 2 | c | Logistic baseline and threshold selection (12 points) |
| 5 | Task 2 | d | Ensemble comparison: RF vs boosting (12 points) |
| 5 | Task 2 | e | Final exam recommendation memo (10 points) |
| 6 | Task 1 | a | Read method from objective function (10 points) |
| 6 | Task 1 | b | OLS vs Ridge vs LASSO (10 points) |
| 6 | Task 1 | c | K-fold CV + one-standard-error rule (10 points) |
| 6 | Task 1 | d | Bootstrap variance uncertainty (10 points) |
| 6 | Task 1 | e | Linear vs smooth effects (10 points) |
| 6 | Task 2 | a | Prediction-time data design (8 points) |
| 6 | Task 2 | b | EDA with actionable hypotheses (8 points) |
| 6 | Task 2 | c | Logistic regression + threshold strategy (12 points) |
| 6 | Task 2 | d | Random forest tuning and interpretation (12 points) |
| 6 | Task 2 | e | Boosting and final recommendation (10 points) |
| 7 | Task 1 | a | Method identification from custom code (10 points) |
| 7 | Task 1 | b | OLS vs LASSO sparsity path (10 points) |
| 7 | Task 1 | c | Repeated K-fold CV (10 points) |
| 7 | Task 1 | d | Bootstrap SD(y) and CI comparison (10 points) |
| 7 | Task 1 | e | Basis expansion vs GAM (10 points) |
| 7 | Task 2 | a | Data cleaning and leakage prevention (8 points) |
| 7 | Task 2 | b | EDA and imbalance diagnosis (8 points) |
| 7 | Task 2 | c | Logistic model with cost matrix (12 points) |
| 7 | Task 2 | d | Random forest (12 points) |
| 7 | Task 2 | e | Gradient boosting (10 points) |
| 8 | Task 1 | a | Explain CV pseudocode (10 points) |
| 8 | Task 1 | b | Regularized model comparison (10 points) |
| 8 | Task 1 | c | Bootstrap coefficient stability (10 points) |
| 8 | Task 1 | d | Variance CI (10 points) |
| 8 | Task 1 | e | GAM diagnostics (10 points) |
| 8 | Task 2 | a | Data prep with collinearity handling (8 points) |
| 8 | Task 2 | b | EDA and interpretable segments (8 points) |
| 8 | Task 2 | c | Logistic baseline + interpretation (12 points) |
| 8 | Task 2 | d | Random forest nonlinear interactions (12 points) |
| 8 | Task 2 | e | Boosting + deployment recommendation (10 points) |
| 9 | Task 1 | a | Theory: LOOCV vs K-fold (10 points) |
| 9 | Task 1 | b | Leakage-safe preprocessing (10 points) |
| 9 | Task 1 | c | Ridge and LASSO tuning (10 points) |
| 9 | Task 1 | d | Bootstrap CI for SD(y) (10 points) |
| 9 | Task 1 | e | OLS vs GAM fit quality (10 points) |
| 9 | Task 2 | a | Data prep and split (8 points) |
| 9 | Task 2 | b | EDA and candidate predictors (8 points) |
| 9 | Task 2 | c | Logistic model diagnostics (12 points) |
| 9 | Task 2 | d | Random forest (12 points) |
| 9 | Task 2 | e | Boosting and final choice (10 points) |
| 10 | Task 1 | a | Read algorithm from code and describe assumptions (10 points) |
| 10 | Task 1 | b | OLS baseline and diagnostics (10 points) |
| 10 | Task 1 | c | Penalized alternatives + tuning (10 points) |
| 10 | Task 1 | d | Bootstrap uncertainty (10 points) |
| 10 | Task 1 | e | GAM enhancement (10 points) |
| 10 | Task 2 | a | Data prep strategy (8 points) |
| 10 | Task 2 | b | EDA and business framing (8 points) |
| 10 | Task 2 | c | Logistic model (12 points) |
| 10 | Task 2 | d | Random forest (12 points) |
| 10 | Task 2 | e | Boosting and final recommendation (10 points) |

## 8) Duplicate-control audit (exact question reuse check)
- **Exact full-question block duplicates (heading + body):** `0`
- **Heading-level duplicates (same task/letter/title only):** `2`
- Heading-level duplicates found (not full-body duplicates):
  - `Task 2 (d)` — "Random forest (12 points)" appears in `3` exams.
  - `Task 2 (e)` — "Boosting and final recommendation (10 points)" appears in `2` exams.
- **Interpretation:** Full-question wording is unique across Exams 1–10, so exact reuse is currently avoided.

## 9) Coverage assessment against curriculum / compendium
Coverage was evaluated against `Task1_comp.qmd` (code-identification dictionary and method banks) and existing compendium patterns in BAN404 materials. A dedicated file named `Kompendie_New` was not found in this repository, so `Task1_comp.qmd` + the current compendium resources were used as the practical reference set.

### 9.1 Task 1 coverage status
- **Covered strongly:** Ridge/LASSO/Elastic Net objectives, K-fold/LOOCV/repeated CV reasoning, leakage-safe preprocessing, bootstrap CI (Var/SD), GAM/nonlinearity, coefficient stability, model-choice justification.
- **Covered moderately:** polynomial vs GAM tradeoffs, residual-diagnostic interpretation, fairness of fold reuse in comparisons.
- **Potentially add if you want one more variant later:** explicit KNN or tree-split pseudocode decoding from `Task1_comp.qmd` quick-lookup bank.

### 9.2 Task 2 coverage status
- **Covered strongly:** churn data preparation, collinearity removal, EDA hypothesis building, logistic threshold tuning (cost-sensitive), RF tuning/importance, boosting staged selection, final business recommendation.
- **Covered moderately:** calibration-focused assessment and explicit profit/uplift framing (optional extension).

### 9.3 Overall readiness judgment
Current mock set (1–10) now spans the major likely BAN404 2026 iteration space for the two available datasets, with distinct question wording and varied methodological emphasis. The remaining risk is not topic omission but exam-day adaptation speed, which is addressed by the new snippets file added in this update.
