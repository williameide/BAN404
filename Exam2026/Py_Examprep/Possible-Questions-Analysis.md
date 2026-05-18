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
