# BAN404 — Course Approval Assignment: Comprehensive Guide

> **Repository:** `williameide/BAN404`  
> **Course:** BAN404 – Business Analytics, NHH Spring 2026  
> **Teacher:** Jonas Andersson  
> **Textbook:** ISLP – *An Introduction to Statistical Learning* (James, Witten, Hastie, Tibshirani & Taylor, 2023) — [free PDF](https://statlearning.com/)  
> **Exam:** 6-hour digital school exam, 19 May 2026. **A passed compulsory assignment is required to sit the exam.**  
> **Assignment deadline:** Handed out 9 March, **due 16 March 2026**

---

## Table of Contents

1. [Repository Map](#1-repository-map)
2. [Course Structure & Topics](#2-course-structure--topics)
3. [The Compulsory Assignment – Overview](#3-the-compulsory-assignment--overview)
4. [Task-by-Task Guide](#4-task-by-task-guide)
   - [Task 1 – Exploratory Data Analysis](#task-1--exploratory-data-analysis--descriptive-statistics)
   - [Task 2 – Linear Regression & Validation Set](#task-2--linear-regression--validation-set-approach)
   - [Task 3 – Leave-One-Out Cross-Validation](#task-3--leave-one-out-cross-validation-loocv)
   - [Task 4 – KNN Regression](#task-4--knn-regression-with-standardisation--cross-validation)
   - [Task 5 – Flexible / Nonlinear Models](#task-5--nonlinear-models--gam--splines--trees)
5. [Key Concepts Summary per Topic](#5-key-concepts-summary-per-topic)
6. [Submission Checklist](#6-submission-checklist)
7. [Difficulties & Recommendations for Accessibility](#7-difficulties--recommendations-for-accessibility)

---

## 1. Repository Map

| File / Folder | What it contains |
|---|---|
| `Course_approval/compulsory_assignment.pdf` | **The actual assignment** (Bike Sharing Dataset, 5 tasks) |
| `schedule2026.pdf` | Full semester schedule with lecture dates, ISLP chapters, and exercise numbers |
| `lecture2 (1).pdf` | Linear regression, KNN, bias-variance tradeoff |
| `lecture3.pdf` | Classification – Logistic regression |
| `lecture4.pdf` | Classification – LDA, QDA |
| `lecture5.pdf` | Resampling – Cross-validation |
| `lecture6.pdf` | Resampling – Bootstrap |
| `lecture7.pdf` | Linear model selection – subset selection, Ridge, Lasso |
| `lecture8ho.pdf` | Non-linear methods – polynomial, step functions, splines |
| `lecture9.pdf` | Smoothing splines, GAM |
| `lecture10.pdf` | Tree-based methods – regression & classification trees |
| `lecture11.pdf` | Bagging, Random Forests, Boosting |
| `lecture12.pdf` | Support Vector Machines |
| `tutorial1.pdf` / `tutorial1_solutions.pdf` | Exercise set 1 – linear regression & KNN |
| `exercise3.pdf` / `exercise3_solutions.pdf` | Exercise set 3 – Ridge, Lasso, splines |
| `ex38.py` | Exercise 3.8 – Auto dataset, simple linear regression + KNN |
| `ex413_a_d.py` | Exercise 4.13 (a–d) – Logistic regression on Weekly data |
| `ex413_e_g.py` | Exercise 4.13 (e–g) – LDA, QDA, KNN on Weekly data |
| `ex55.py` | Exercise 5.5 – Validation set & LOOCV on Default dataset |
| `ex611.py` | Exercise 6.11 – Ridge & Lasso on Boston data |
| `ex76.ipynb` | Exercise 7.6 – Polynomial & step-function regression on Wage data |
| `ex88ac.ipynb` / `ex88.ipynb` | Exercise 8.8 – Decision trees |
| `ex97.ipynb` / `ex97.py` | Exercise 9.7 – SVM on Auto data |
| `gam_example.py` / `gam_example.ipynb` | GAM backfitting example |
| `BAN404_exam_2024.pdf` | Previous exam (airline dataset, classification + regression) |
| `airline.csv` / `stocks.csv` | Data files used in exercises / exam |
| `BAN404_Kompendium_Sander.pdf` | Student-written compendium (supplementary) |
| `ban404-candiadte-164.pdf` | Example student submission from previous year |

---

## 2. Course Structure & Topics

The course follows ISLP chapters 1–12 and covers the following topics:

| Week | Topic | ISLP Chapter | Relevant exercises in repo |
|---|---|---|---|
| Jan 12–26 | Linear regression, KNN, bias-variance tradeoff | Ch 1–3 | `ex38.py`, `tutorial1.pdf` |
| Jan 28–Feb 11 | Classification: Logistic regression, LDA, QDA, KNN | Ch 4 | `ex413_a_d.py`, `ex413_e_g.py` |
| Feb 4–9 | Resampling: Cross-validation | Ch 5 | `ex55.py` |
| Feb 9 | Resampling: Bootstrap | Ch 5 | — |
| Feb 16–23 | Linear model selection: Ridge, Lasso, splines | Ch 6–7 | `ex611.py`, `exercise3.pdf` |
| Feb 23–Mar 2 | Smoothing splines, GAM | Ch 7 | `gam_example.py`, `ex76.ipynb` |
| Mar 4–9 | Tree-based methods, Bagging, Boosting | Ch 8 | `ex88ac.ipynb` |
| Mar 16 | SVM | Ch 9 | `ex97.py` |
| Mar 18–25 | Deep Learning, Unsupervised Learning | Ch 10, 12 | — |

---

## 3. The Compulsory Assignment – Overview

### Dataset
**Bike Sharing Dataset** (UCI Machine Learning Repository)  
- URL: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset  
- **File to download:** `day.csv`  
- **Outcome variable:** `cnt` – total number of daily bike rentals  

### Format
- Submit as a **Quarto document (`.qmd`) or Jupyter notebook (`.ipynb`)**
- All code must be included and **run without errors** (given packages installed)
- Work in **groups of at most 4 students** (or individually); one submission per group
- **Narrative text ≤ 3,000 words** (excludes code, figures, tables, printed output)
- Each task answered in **2–3 short paragraphs maximum**

### Starter Code (provided in the assignment)
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

np.random.seed(123)

data = pd.read_csv("day.csv")

X = data.drop(columns=["cnt"])
y = data["cnt"]
```

### Grading philosophy (from the assignment)
> *"You are not evaluated on achieving the lowest possible prediction error. The main focus is on correct use of models and methods, clear reasoning, and interpretation of results."*

---

## 4. Task-by-Task Guide

---

### Task 1 – Exploratory Data Analysis & Descriptive Statistics

**Assignment text:**
> Describe relevant features of the output variable `cnt` using descriptive statistics, tables, and graphs. Use descriptive analysis to investigate which predictors appear promising for predicting daily bike rentals. Remove variables that cannot reasonably be assumed to be available at the time a prediction is made, and briefly justify your choices.

#### Key Insights

1. **Variable removal is important and graded.** You must think about which columns would realistically be available *before* the bike rental day happens. This is a causal/leakage question.
   - **Remove:** `casual` and `registered` (they are components of `cnt` — perfect leakage)
   - **Remove (or argue about):** `instant` (just a row index), `dteday` (date string, redundant with other time variables)
   - **Potential to remove:** `atemp` if you consider it might correlate too much with `temp`, or vice versa. Justify any choice.
   - **Keep:** Weather-based predictors (`temp`, `windspeed`, `hum`), time-based predictors (`season`, `mnth`, `weekday`, `holiday`, `workingday`, `yr`)

2. **Variable types matter.** The assignment specifically says: *"Variables should be of the correct type, numeric for quantitative variables and categorical where appropriate."*  
   - `season`, `weathersit`, `weekday`, `mnth`, `holiday`, `workingday`, `yr` should be treated as **categorical** (use `pd.Categorical` or convert to dummy variables)
   - `temp`, `atemp`, `hum`, `windspeed` are **quantitative**

3. **What to show descriptively:**
   - Distribution of `cnt` (histogram, boxplot)
   - Time trends (cnt over time, by season, by weather)
   - Scatterplots of `cnt` vs continuous predictors
   - Mean `cnt` by categorical predictors

#### Suggested Code Pattern

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("day.csv")

# Distribution of cnt
data["cnt"].describe()
data["cnt"].hist(bins=30)

# Scatterplots vs continuous predictors
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, var in enumerate(["temp", "hum", "windspeed"]):
    axes[i].scatter(data[var], data["cnt"], alpha=0.4)
    axes[i].set_xlabel(var)
    axes[i].set_ylabel("cnt")
plt.tight_layout()

# Mean cnt by season
data.groupby("season")["cnt"].mean()

# Remove leakage/irrelevant columns
data_clean = data.drop(columns=["casual", "registered", "instant", "dteday"])

# Convert categoricals
for col in ["season", "weathersit", "weekday", "mnth", "holiday", "workingday", "yr"]:
    data_clean[col] = data_clean[col].astype("category")

X = data_clean.drop(columns=["cnt"])
y = data_clean["cnt"]
```

#### What to write (2–3 paragraphs)
- What does `cnt` look like? (range, mean, skew)
- Which predictors show the strongest association with `cnt`?
- Which variables were removed and why?

---

### Task 2 – Linear Regression & Validation Set Approach

**Assignment text:**
> Use linear regression to predict `cnt`. Estimate at least one multiple linear regression model and one more flexible model (for example, a polynomial regression). Use the validation set approach to evaluate predictive performance on test data using an appropriate error measure, and discuss the results.

#### Key Insights

1. **Validation set approach** = split data into training and test sets, fit on training, evaluate on test. This is the most basic resampling method (ISLP Ch. 5). The lecture slide code from `ex55.py` and `ex38.py` show the pattern clearly.

2. **Error measure**: Use **Mean Squared Error (MSE)** = mean of (y_actual - y_predicted)²  
   You can also report **RMSE** (Root MSE) for interpretability. For bike counts, RMSE in "number of bikes" is intuitive.

3. **At least two models required:**
   - **Multiple linear regression:** e.g., `cnt ~ temp + hum + windspeed + season + yr + ...`
   - **More flexible model:** e.g., polynomial terms (e.g., `temp²`), or more predictors

4. **Categorical variables** need to be properly handled:
   - With `sklearn`: use `pd.get_dummies()` or `sklearn.preprocessing.OneHotEncoder`
   - With `statsmodels`: use formula interface with `C(season)` for categoricals

5. **Describe the train/test split** clearly (e.g., 70/30 or 50/50, how it was done).

#### Suggested Code Pattern

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

np.random.seed(123)

# Convert categoricals to dummies
X_dummies = pd.get_dummies(X, drop_first=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, test_size=0.3, random_state=123)

# Model 1: Multiple linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = np.mean((y_test - y_pred_lr)**2)
print("Linear regression test MSE:", round(mse_lr, 1))
print("Linear regression test RMSE:", round(np.sqrt(mse_lr), 1))

# Model 2: Polynomial regression (add temperature squared)
X_poly_train = X_train.copy()
X_poly_test = X_test.copy()
X_poly_train["temp_sq"] = X_poly_train["temp"]**2
X_poly_test["temp_sq"] = X_poly_test["temp"]**2

lr2 = LinearRegression()
lr2.fit(X_poly_train, y_train)
y_pred_lr2 = lr2.predict(X_poly_test)
mse_lr2 = np.mean((y_test - y_pred_lr2)**2)
print("Polynomial regression test MSE:", round(mse_lr2, 1))
```

#### What to write (2–3 paragraphs)
- How was the data split? (seed, proportions)
- What are the test MSE values for each model?
- Which model performs better, and why might that be?

---

### Task 3 – Leave-One-Out Cross-Validation (LOOCV)

**Assignment text:**
> Use leave-one-out cross-validation to estimate test error for at least two competing models. Compare the results to those obtained using the validation set in Task 2, and comment on the stability of the estimates.

#### Key Insights

1. **LOOCV** = each observation is used once as the test set, the model is trained on all others. With n=731 rows (day.csv), this means fitting 731 models. In practice, `sklearn` can do this efficiently.

2. **What LOOCV does well:** It uses almost all data for training each time, so it has lower bias than validation set. But it can have higher variance if observations are correlated. Also, for linear models, there is a shortcut that makes LOOCV extremely fast (no need to refit 731 times).

3. **Key conceptual point** (likely to be in discussion): LOOCV gives a *more stable* estimate than the validation set approach because it doesn't depend on a single random split. The validation set estimate can vary significantly depending on which observations end up in the test set.

4. **With `sklearn`:** Use `LeaveOneOut` or `cross_val_score` with `cv=LeaveOneOut()` or `cv=len(X_train)` for LOOCV.

#### Suggested Code Pattern

```python
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

loo = LeaveOneOut()

# Model 1: Linear regression
lr = LinearRegression()
scores_lr = cross_val_score(lr, X_dummies, y, cv=loo, scoring="neg_mean_squared_error")
loocv_mse_lr = -scores_lr.mean()
print("LOOCV MSE (linear regression):", round(loocv_mse_lr, 1))

# Model 2: Polynomial
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("lr", LinearRegression())
])
scores_poly = cross_val_score(poly_model, X_dummies[["temp", "hum", "windspeed"]], y,
                              cv=loo, scoring="neg_mean_squared_error")
loocv_mse_poly = -scores_poly.mean()
print("LOOCV MSE (polynomial):", round(loocv_mse_poly, 1))
```

> **Note:** LOOCV on the full feature set with `get_dummies` may be very slow. Consider using fewer predictors, or use 5-fold or 10-fold CV as an approximation.

#### What to write (2–3 paragraphs)
- How is LOOCV implemented? (explicitly mention each observation used as test)
- How do the LOOCV MSE values compare to the validation set MSE values from Task 2?
- Is the estimate more or less stable? Why? (key: depends on random split vs full CV)

---

### Task 4 – KNN Regression with Standardisation & Cross-Validation

**Assignment text:**
> Predict `cnt` using a K-nearest neighbors (KNN) regression model. Standardize the predictors before fitting the model. Use cross-validation to determine a suitable value of K. Evaluate the predictive performance and compare it to the linear regression results.

The assignment provides a **specific code block** to explain and use:

```python
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor())
])

cv = KFold(n_splits=10, shuffle=True, random_state=1)

Ks = range(1, 41)
cv_mse = []

for k in Ks:
    pipe.set_params(knn__n_neighbors=k)
    fold_errors = []

    for train_idx, test_idx in cv.split(X):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

        pipe.fit(X_train_cv, y_train_cv)
        y_pred = pipe.predict(X_test_cv)
        fold_errors.append(np.mean((y_test_cv - y_pred)**2))

    cv_mse.append(np.mean(fold_errors))
```

**You must explain what this code is doing.**

#### Key Insights

1. **Why standardise?** KNN computes distances between observations. If predictors are on very different scales (e.g., `temp` ∈ [0,1] vs. `windspeed` ∈ [0,0.8] vs. a dummy variable ∈ {0,1}), predictors with larger values will dominate the distance. Standardising (z-score: subtract mean, divide by std) puts all predictors on the same scale.

2. **Pipeline:** The `Pipeline` chains `StandardScaler` → `KNeighborsRegressor`. This ensures scaling is fit on training data only and applied consistently to test folds (prevents data leakage).

3. **Explanation of the provided code:**
   - `KFold(n_splits=10, shuffle=True, random_state=1)`: 10-fold cross-validation with random shuffling
   - The outer loop iterates over K = 1 to 40
   - The inner loop manually does 10-fold CV for each K value
   - `pipe.set_params(knn__n_neighbors=k)`: sets K on the KNN step inside the pipeline
   - The result is a list of 40 CV-MSE values, one per K
   - **The optimal K** is the one with the lowest CV-MSE

4. **Plot CV-MSE vs K** to find the elbow / minimum.

5. **Compare KNN test MSE** (using optimal K, evaluated on held-out test set) to linear regression from Task 2.

#### Suggested Code Pattern (complete Task 4)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

# Set up Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor())
])

cv = KFold(n_splits=10, shuffle=True, random_state=1)

# Only use numeric predictors for KNN (or encode all)
X_num = pd.get_dummies(X, drop_first=True).astype(float)

Ks = range(1, 41)
cv_mse = []

for k in Ks:
    pipe.set_params(knn__n_neighbors=k)
    fold_errors = []
    for train_idx, test_idx in cv.split(X_num):
        X_train_cv = X_num.iloc[train_idx]
        X_test_cv  = X_num.iloc[test_idx]
        y_train_cv = y.iloc[train_idx]
        y_test_cv  = y.iloc[test_idx]
        pipe.fit(X_train_cv, y_train_cv)
        y_pred = pipe.predict(X_test_cv)
        fold_errors.append(np.mean((y_test_cv - y_pred)**2))
    cv_mse.append(np.mean(fold_errors))

# Find optimal K
best_k = Ks[np.argmin(cv_mse)]
print(f"Optimal K: {best_k}, CV-MSE: {min(cv_mse):.1f}")

# Plot
plt.plot(list(Ks), cv_mse)
plt.xlabel("K")
plt.ylabel("CV MSE")
plt.title("KNN: CV MSE vs K")
plt.axvline(best_k, color="red", linestyle="--", label=f"Optimal K={best_k}")
plt.legend()
plt.show()

# Evaluate on test set with optimal K
pipe.set_params(knn__n_neighbors=best_k)
pipe.fit(X_train_cv, y_train_cv)  # refit on a train/test split
y_pred_knn = pipe.predict(X_test_cv)
mse_knn = np.mean((y_test_cv - y_pred_knn)**2)
print(f"KNN test MSE (K={best_k}): {mse_knn:.1f}")
```

#### What to write (2–3 paragraphs)
- **Explain the provided code** (this is explicitly asked)
- Why is standardisation necessary for KNN?
- What is the optimal K? How does KNN compare to linear regression?

---

### Task 5 – Nonlinear Models (GAM / Splines / Trees)

**Assignment text:**
> Argue whether there are indications of nonlinear relationships between `cnt` and one or more predictors. Fit a flexible model, such as a generalized additive model (GAM), smoothing spline, or regression tree. Evaluate the predictive performance and compare it with earlier models.

#### Key Insights

1. **Argue for nonlinearity first.** Look at your scatterplots from Task 1. `temp` vs `cnt` likely shows a clear non-linear (inverted-U) relationship — bike use peaks at mild temperatures and drops in extreme heat or cold. `hum` may also show nonlinearity.

2. **Choose one or more flexible models:**

   **Option A – Regression Tree** (simplest to implement, covered in `ex88ac.ipynb`):
   ```python
   from sklearn.tree import DecisionTreeRegressor
   from sklearn.model_selection import GridSearchCV, KFold
   
   kfold = KFold(5, shuffle=True, random_state=10)
   tree = DecisionTreeRegressor()
   cv_tree = tree.cost_complexity_pruning_path(X_train, y_train)
   grid = GridSearchCV(tree, {'ccp_alpha': cv_tree.ccp_alphas}, cv=kfold,
                       scoring="neg_mean_squared_error", refit=True)
   grid.fit(X_train, y_train)
   best_tree = grid.best_estimator_
   ```

   **Option B – GAM with pygam** (covered in `gam_example.py`, `lecture9.pdf`):
   ```python
   from pygam import LinearGAM, s, l
   
   # s(i) = spline for predictor i, l(i) = linear for predictor i
   gam = LinearGAM(s(0) + s(1) + s(2)).fit(X_train_num, y_train)
   y_pred_gam = gam.predict(X_test_num)
   mse_gam = np.mean((y_test - y_pred_gam)**2)
   ```

   **Option C – Polynomial regression** (already touched in Task 2 — valid to extend here with cross-validation to select degree, as in `ex76.ipynb`).

3. **What GAM does:** Fits `y = β₀ + f₁(x₁) + f₂(x₂) + ... + fₚ(xₚ) + ε`, where each `fⱼ` is a smooth function estimated by a smoothing spline or local regression. The **backfitting** algorithm (shown in `gam_example.py`) iteratively estimates each `fⱼ` while keeping the others fixed.

4. **Tree interpretation:** Decision trees partition the predictor space into rectangular regions. The `plot_tree` function can visualise the tree structure. Pruning is done via cost-complexity pruning (controlled by `ccp_alpha`).

5. **Compare all models** (Tasks 2, 3, 4, 5) in a final summary table of test MSE values.

#### Suggested Code Pattern (Regression Tree)

```python
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV, KFold
import matplotlib.pyplot as plt

# Use same train/test split as Task 2
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_dummies, y, test_size=0.3, random_state=123
)

# Pruning path
tree_base = DecisionTreeRegressor(random_state=0)
tree_base.fit(X_train_d, y_train_d)
path = tree_base.cost_complexity_pruning_path(X_train_d, y_train_d)

kfold = KFold(5, shuffle=True, random_state=10)
grid = GridSearchCV(
    DecisionTreeRegressor(random_state=0),
    {"ccp_alpha": path.ccp_alphas},
    cv=kfold,
    scoring="neg_mean_squared_error",
    refit=True
)
grid.fit(X_train_d, y_train_d)
best_tree = grid.best_estimator_

y_pred_tree = best_tree.predict(X_test_d)
mse_tree = np.mean((y_test_d - y_pred_tree)**2)
print(f"Decision tree test MSE: {mse_tree:.1f}")

# Plot tree
fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(best_tree, feature_names=X_dummies.columns, precision=2, fontsize=7, ax=ax)
plt.tight_layout()
plt.show()
```

#### What to write (2–3 paragraphs)
- Which predictors show nonlinear relationships with `cnt`? (reference Task 1 plots)
- Which flexible model did you choose and why?
- What is the test MSE? How does it compare to linear regression and KNN?

---

## 5. Key Concepts Summary per Topic

### Linear Regression
- **OLS**: minimise RSS = Σ(yᵢ − ŷᵢ)²
- **MSE** = RSS/n; used to compare models (lower = better)
- **Training MSE always decreases** with flexibility; Test MSE follows a U-shape (bias-variance tradeoff)
- **From lectures:** `sm.OLS`, `smf.ols`, `sklearn.linear_model.LinearRegression`
- **Key file:** `ex38.py` — shows full workflow from scatter plot → OLS fit → prediction → residual plot

### KNN
- Non-parametric, predicts as average of K nearest neighbours
- Works for both regression and classification
- **Must standardise** predictors when using multiple predictors (different scales distort distances)
- Small K → overfit; Large K → underfit. Use CV to select K.
- **Key file:** `ex38.py` (KNN function from scratch), `ex413_e_g.py` (KNN classifier)

### Validation Set / Cross-Validation
- **Validation set:** simple split, computationally fast, but estimate varies with split
- **LOOCV:** low bias, high variance in estimate, computationally expensive for non-linear models
- **K-fold CV (K=5 or 10):** good tradeoff, standard in practice
- **Key file:** `ex55.py` — shows validation set and LOOCV on Default dataset

### Ridge Regression & Lasso
- Both penalise large coefficients to prevent overfitting
- **Ridge:** λΣβⱼ² — shrinks all coefficients, never sets to zero
- **Lasso:** λΣ|βⱼ| — can set coefficients exactly to zero (variable selection)
- **Tuning λ** via cross-validation (RidgeCV, LassoCV)
- **Key file:** `ex611.py` — full Ridge and Lasso workflow on Boston data

### Nonlinear Methods
- **Polynomial:** regress Y on X, X², X³,... — still OLS but with transformed predictors
- **Smoothing splines:** minimise RSS + λ∫g''(t)² — λ tuned by CV
- **GAM:** y = β₀ + f₁(x₁) + f₂(x₂) + ... — additive smooth functions, fitted by backfitting
- **Key files:** `ex76.ipynb` (polynomial, step functions, ISLP Wage data), `gam_example.py` (GAM backfitting), `lecture8ho.pdf`, `lecture9.pdf`

### Decision Trees / Bagging / Boosting
- **Trees:** recursive binary splitting to minimise RSS; pruning with cost-complexity parameter α
- **Bagging:** average predictions from B bootstrap resampled trees → reduces variance
- **Random Forests:** like bagging but each split considers only m < p predictors → decorrelates trees
- **Boosting:** sequentially fit trees on residuals; each tree corrects errors of previous
- **Key file:** `ex88ac.ipynb` — decision tree fitting, pruning, and visualisation

### Support Vector Machines (SVM)
- **Linear SVC:** separates classes with a maximal margin hyperplane
- **Soft margin:** allows some misclassifications (controlled by C)
- **Kernels:** polynomial, radial (RBF) — allow nonlinear boundaries
- **Key file:** `ex97.py` — SVM on Auto data with linear, polynomial, and radial kernels

---

## 6. Submission Checklist

Use this list before submitting:

- [ ] **Data loading:** `day.csv` is read from file (not hardcoded values)
- [ ] **Leakage variables removed:** `casual`, `registered` dropped (and `instant`, `dteday` or justified)
- [ ] **Variable types correct:** season, weathersit, weekday, mnth etc. treated as categorical
- [ ] **Random seed set:** `np.random.seed(123)` at the top
- [ ] **Task 1:** Descriptive statistics, histogram/distribution of `cnt`, scatterplots, boxplots by category, correlation analysis
- [ ] **Task 2:** ≥2 regression models fitted, validation set documented (size, seed), MSE reported, results discussed
- [ ] **Task 3:** LOOCV implemented for ≥2 models, compared to Task 2 results, stability discussed
- [ ] **Task 4:** KNN pipeline with StandardScaler, CV over K=1..40, optimal K found, code explained, test MSE reported
- [ ] **Task 5:** Nonlinearity argued with evidence from Task 1, flexible model fitted and evaluated, comparison to earlier models
- [ ] **Word limit:** narrative text ≤ 3,000 words
- [ ] **Code runs without errors** from top to bottom
- [ ] **All group member names and student numbers stated** in the document
- [ ] **File format:** `.qmd` or `.ipynb`

---

## 7. Difficulties & Recommendations for Accessibility

During the analysis of this repository, the following difficulties and observations were noted. Addressing these would significantly improve the quality of AI-assisted work in future sessions:

### ✅ What worked well
- **Python source files** (`.py`) were fully readable and contain well-commented code solutions from exercises
- **Jupyter notebooks** (`.ipynb`) were readable via JSON parsing and contain structured, runnable code
- **PDF extraction** worked well for most lecture slides and exercise sets (text-based PDFs)
- **The compulsory assignment PDF** (`Course_approval/compulsory_assignment.pdf`) was fully extractable and contains all 5 tasks with clear descriptions

### ⚠️ Difficulties Encountered

#### 1. Data File Not Present (`day.csv`)
- The compulsory assignment requires `day.csv` (Bike Sharing Dataset from UCI), but **this file is not in the repository**
- This means it was not possible to actually run or test any of the assignment code
- **Recommendation:** Add `day.csv` to the repository (or a link/script to download it automatically). This would enable automated testing and AI-assisted code generation

#### 2. Some PDFs are slide-presentation format (dense, layout-heavy)
- The lecture PDFs render well overall, but some mathematical formulas, especially matrices and multi-line equations, are partially garbled in text extraction (e.g., equations appear without spacing, symbols are lost)
- **Recommendation:** If possible, add text-format lecture notes or a markdown/HTML summary of key formulas per lecture — this would significantly help an AI agent (and students) quickly reference key mathematical concepts

#### 3. No example `.qmd` or `.ipynb` assignment template
- There is no template for the submission format (Quarto or Jupyter)
- **Recommendation:** Add a starter template file (e.g., `Course_approval/assignment_template.ipynb`) with the required structure, imports, and section headings. This reduces formatting work and ensures correct submission format

#### 4. The previous student submission (`ban404-candiadte-164.pdf`) is in PDF format
- This example of a previous year's submission is in PDF, making it hard to extract the code or structure from it
- **Recommendation:** If possible, include a `.ipynb` or `.qmd` version of the example submission so students can see the expected code + text structure directly

#### 5. Package dependencies not documented
- The exercises use: `numpy`, `pandas`, `matplotlib`, `statsmodels`, `sklearn`, `ISLP`, `pygam`
- There is no `requirements.txt` or `environment.yml` in the repository
- **Recommendation:** Add a `requirements.txt` with all required packages. This enables one-command setup (`pip install -r requirements.txt`) and ensures reproducibility

#### 6. Exercise numbering references ISLP chapters (e.g., "ex413_a_d.py" = Exercise 4.13a-d)
- This naming convention is efficient for course use but not immediately obvious to newcomers
- **Recommendation:** Add a brief comment at the top of each `.py` file mapping the file to the ISLP chapter/section and briefly stating what it covers — most files already have comments, but standardising would help

---

### Summary of Recommended Repository Additions

| Addition | Priority | Benefit |
|---|---|---|
| `day.csv` (Bike Sharing data) | 🔴 High | Assignment can be run and tested |
| `requirements.txt` | 🟡 Medium | Easy environment setup |
| `Course_approval/assignment_template.ipynb` | 🟡 Medium | Correct submission format, starter code |
| Markdown formula summary per lecture | 🟢 Low | Better AI/student accessibility of math |
| Text/notebook version of example submission | 🟢 Low | Better understanding of expected output |

---

*Report generated by AI assistant based on full repository analysis, March 2026.*
