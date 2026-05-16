# BAN404 — Python Command Reference for the Exam

> **Purpose:** A structured, exam-oriented reference of every Python command, function, and pattern covered in BAN404.  
> Commands are drawn from the lecture exercises (ex38, ex413, ex55, ex611, ex76, gam_example, ex88, ex97, ames), the compulsory assignment, and the course overview. Importance is rated on a three-star scale:
>
> | Rating | Meaning |
> |--------|---------|
> | ★★★ | Critical — appears in multiple exercises/exams, central to the course |
> | ★★  | Important — covered explicitly, likely tested |
> | ★   | Good to know — implied by context or useful support command |
>
> **Exam note:** The exam is closed-book. Focus on *understanding the pattern*, not copy-pasting. Each command below includes a minimal, memorable syntax example.

---

## General Setup (All Modules)

These imports and utilities appear in virtually every exercise.

| Command / Pattern | Importance | Syntax Example | Notes |
|---|---|---|---|
| `import numpy as np` | ★★★ | `import numpy as np` | Core numerical library |
| `import pandas as pd` | ★★★ | `import pandas as pd` | DataFrames, data manipulation |
| `import matplotlib.pyplot as plt` | ★★★ | `import matplotlib.pyplot as plt` | Plotting |
| `import seaborn as sns` | ★★ | `import seaborn as sns` | Enhanced plots (heatmaps, histograms) |
| `from ISLP import load_data` | ★★★ | `df = load_data("Auto")` | Load ISLP datasets (Auto, Boston, Wage, etc.) |
| `np.random.seed(123)` | ★★★ | `np.random.seed(123)` | Set seed for reproducibility — always do this first |
| `df.head()` | ★★ | `df.head()` | Preview first 5 rows |
| `df.describe()` | ★★ | `df.describe(include='all').round(3)` | Summary statistics |
| `df.dtypes` | ★ | `df.dtypes` | Check column types |
| `df.columns` | ★ | `df.columns` | List all column names |
| `df.shape` | ★★ | `X.shape[1]` | Rows, columns — `shape[1]` gives number of features |
| `df['col'].value_counts()` | ★★ | `df['Direction'].value_counts()` | Count occurrences per category |
| `df.groupby('col')['y'].mean()` | ★★ | `df.groupby('Direction')['Volume'].mean()` | Group-level statistics |
| `df.corr()` | ★★ | `df.corr()` | Correlation matrix |
| `pd.get_dummies(df, drop_first=True)` | ★★★ | `X = pd.get_dummies(X, drop_first=True)` | Encode categorical variables (avoids dummy trap) |
| `df.drop(columns=['col'])` | ★★★ | `X = df.drop(columns='Sales')` | Remove columns |
| `df.fillna(0)` | ★★ | `X = X.fillna(0)` | Fill missing values |
| `df.isna().sum()` | ★★ | `X.isna().sum().sort_values(ascending=False)` | Check missing values |
| `df.astype(float)` | ★ | `ames_small = ames_small.astype(float)` | Convert all columns to numeric |
| `pd.crosstab(y, pred)` | ★★★ | `pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])` | Confusion matrix |
| `np.mean(y == pred)` | ★★★ | `accuracy = np.mean(y_test == pred_test)` | Classification accuracy |
| `np.trace(conf_mat) / n` | ★★ | `accuracy = np.trace(conf_mat) / ntrain` | Accuracy from confusion matrix |
| `np.mean((y - yhat)**2)` | ★★★ | `MSE = np.mean((y_test - pred)**2)` | Mean squared error (manual) |
| `plt.scatter(x, y)` | ★★ | `plt.scatter(Auto['horsepower'], Auto['mpg'])` | Scatter plot |
| `plt.show()` | ★★ | `plt.show()` | Display plot |
| `fig, ax = plt.subplots(1, 2, figsize=(10,4))` | ★★ | `fig, axs = plt.subplots(1, 2, figsize=(10, 4))` | Multiple subplots |
| `ax.plot(x, y)` | ★★ | `ax.plot(Weekly['Today'])` | Line plot on an axis |
| `sns.histplot(data, bins=30, kde=True)` | ★★ | `sns.histplot(data['cnt'], bins=30, kde=True)` | Histogram with KDE overlay |
| `sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')` | ★★ | `sns.heatmap(corr[['cnt']].sort_values(by='cnt'), annot=True)` | Heatmap |
| `np.argsort(x)` | ★★ | `o = np.argsort(x); plt.plot(x[o], y[o])` | Sort indices — needed for smooth line plots |
| `np.logspace(-4, 4, 200)` | ★★★ | `alphas = np.logspace(-4, 4, 200)` | Grid of regularization values (log scale) |
| `pd.read_csv("file.csv")` | ★★ | `ames = pd.read_csv("AmesHousing.csv")` | Read CSV file |

---

## Module 1: Linear Models and Regression (Lectures 2–3)

### Lecture 2: Linear Regression (Exercise 3.8 — Auto dataset)

| Command / Pattern | Importance | Syntax Example | Notes |
|---|---|---|---|
| `from ISLP.models import summarize, ModelSpec as MS` | ★★★ | `from ISLP.models import summarize, ModelSpec as MS` | ISLP-specific model tools |
| `import statsmodels.api as sm` | ★★★ | `import statsmodels.api as sm` | Core stats library for OLS, GLM |
| `MS(['col']).fit(data)` | ★★★ | `design = MS(['horsepower']); design = design.fit(Auto)` | Create + fit a model specification (ISLP) |
| `design.transform(data)` | ★★★ | `X = design.transform(Auto)` | Apply the spec to get design matrix |
| `sm.OLS(y, X).fit()` | ★★★ | `results = sm.OLS(y, X).fit()` | Fit OLS (Ordinary Least Squares) — **sm.add_constant not needed when using MS** |
| `results.summary()` | ★★★ | `results.summary()` | Full regression table (coefficients, p-values, R²) |
| `summarize(results)` | ★★★ | `summarize(results)` | Cleaner ISLP summary |
| `results.params` | ★★★ | `results.params[0]` (intercept), `results.params[1]` (slope) | Extract coefficients |
| `results.fittedvalues` | ★★★ | `results.fittedvalues` | Fitted (predicted) values on training data |
| `results.resid` | ★★★ | `results.resid` | Residuals |
| `results.predict(X_new)` | ★★★ | `pred = results.predict(X_new)` | Point predictions |
| `results.get_prediction(X_new).conf_int(alpha=0.05)` | ★★★ | `ci = results.get_prediction(X_new).conf_int(alpha=0.05)` | **Confidence interval** (uncertainty in the mean) |
| `results.get_prediction(X_new).conf_int(obs=True, alpha=0.05)` | ★★★ | `pi = results.get_prediction(X_new).conf_int(obs=True, alpha=0.05)` | **Prediction interval** (uncertainty for a single new obs) |
| `sm.add_constant(X)` | ★★★ | `X_sm = sm.add_constant(X)` | Prepend intercept column (needed with raw arrays) |
| `ax.axline(xy1=[0, b0], slope=b1, color='red')` | ★★ | `ax.axline(xy1=[0, results.params[0]], slope=results.params[1])` | Draw regression line |
| `sm.qqplot(resid, line='45', ...)` | ★★★ | `sm.qqplot(results.resid, line='45', loc=..., scale=..., ax=ax)` | QQ-plot for normality check |
| `axs[0].scatter(results.fittedvalues, results.resid)` | ★★★ | Residuals vs fitted plot | Check for heteroscedasticity / non-linearity |
| `axs[0].axhline(0, color='black')` | ★★ | Adds horizontal line at 0 on residual plot | |
| `import statsmodels.formula.api as smf` | ★★★ | `import statsmodels.formula.api as smf` | Formula-based API |
| `smf.ols('y ~ x1 + x2', data=df).fit()` | ★★★ | `smf.ols('wage ~ age', data=Wage).fit()` | Formula-based OLS |
| `np.vander(x, k+1, increasing=True)` | ★★ | `smf.ols(f'wage ~ np.vander(age, {k+1}, increasing=True)', data=train)` | Polynomial regression via formula |
| `pd.cut(x, k)` | ★★ | `df['age_cut'] = pd.cut(df['age'], k)` | Step function / binning |
| Custom KNN function | ★★ | `def knn(x0, x, y, K): d=np.abs(x-x0); idx=np.argsort(d)[:K]; return np.mean(y[idx])` | Manual KNN for regression (Lecture 2 exercise) |

### Lecture 3: Logistic Regression (Exercise 4.13 a–d — Weekly dataset)

| Command / Pattern | Importance | Syntax Example | Notes |
|---|---|---|---|
| `smf.logit('y ~ x1 + x2', data=df).fit()` | ★★★ | `m1 = smf.logit('y ~ Volume + Lag1 + Lag2', data=Weekly).fit()` | Logistic regression (formula API) |
| `smf.glm('y ~ x', data=df, family=sm.families.Binomial()).fit()` | ★★★ | `smf.glm('default ~ income + balance', data=Default, family=sm.families.Binomial()).fit()` | GLM with binomial family — equivalent to logistic |
| `model.predict()` | ★★★ | `prob = m1.predict()` | Predicted probabilities (0–1) |
| `model.predict(test)` | ★★★ | `prob_test = m1_train.predict(test)` | Probabilities on new data |
| `prob > 0.5` | ★★★ | `pred = prob > 0.5` | Classify at threshold 0.5 |
| `df['y'] = (df['Direction'] == 'Up').astype(int)` | ★★★ | Encode binary target as 0/1 | |
| `pd.crosstab(y_true, y_pred)` | ★★★ | Confusion matrix | |
| Time-based train/test split | ★★★ | `ind = Weekly['Year'] < 2009; train = Weekly[ind]; test = Weekly[~ind]` | Temporal split — common in exam questions |
| `from scipy import stats` | ★ | Two-sample t-test: `tval = (x.mean()-y.mean())/np.sqrt(x.var()/nx + y.var()/ny)` | Manual t-test |
| `stats.norm.cdf(-np.abs(tval)) * 2` | ★ | `pval = 2*stats.norm.cdf(-np.abs(tval))` | Two-sided p-value from z-test |

---

## Module 2: Classification Methods (Lecture 4)

*Exercise 4.13 e–g — Weekly dataset (continuation of Lecture 3 data preparation)*

| Command / Pattern | Importance | Syntax Example | Notes |
|---|---|---|---|
| `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis` | ★★★ | `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis` | LDA |
| `from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis` | ★★★ | `from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis` | QDA |
| `from sklearn.neighbors import KNeighborsClassifier` | ★★★ | `from sklearn.neighbors import KNeighborsClassifier` | KNN classifier |
| `X_train = train[['Lag2']].values` | ★★★ | 2D array for sklearn | sklearn expects 2D feature matrix |
| `lda.fit(X_train, y_train)` | ★★★ | `lda1 = LinearDiscriminantAnalysis(); lda1.fit(X_train, y_train)` | Fit LDA |
| `lda.predict(X_test)` | ★★★ | `pred_class = lda1.predict(X_test)` | Predicted class labels |
| `lda.predict_proba(X_test)` | ★★★ | `posterior = lda1.predict_proba(X_test)` | Posterior probabilities — `[:,1]` for class 1 |
| `posterior[:,1] > threshold` | ★★★ | `pred = lda1.predict_proba(X_test)[:,1] > 0.54` | Adjust decision threshold |
| `np.where(condition, 'Up', 'Down')` | ★★ | `pred_class = np.where(pred5_class, 'Up', 'Down')` | Convert boolean to string labels |
| `qda.fit(X_train, y_train)` | ★★★ | `qda1 = QuadraticDiscriminantAnalysis(); qda1.fit(X_train, y_train)` | Fit QDA (same API as LDA) |
| `knn.fit(X_train, y_train)` | ★★★ | `knn = KNeighborsClassifier(n_neighbors=3); knn.fit(X_train, y_train)` | Fit KNN |
| `knn.predict_proba(X_test)` | ★★★ | `prob7 = knn.predict_proba(X_test)` | KNN probabilities |
| `pd.crosstab(y_test, pred, normalize='index')` | ★★★ | `prop_table = pd.crosstab(y_test, pred, normalize='index')` | Row-proportional confusion table |
| `(pred == y_test).mean()` | ★★★ | `accuracy = (pred_class == y_test).mean()` | Accuracy — the standard evaluation metric |

---

## Module 3: Resampling Methods (Lecture 5)

*Exercise 5.5 — Default dataset; Bootstrap — Lecture 6 / stocks.csv*

### Cross-Validation and Validation Set

| Command / Pattern | Importance | Syntax Example | Notes |
|---|---|---|---|
| `np.random.choice(n, size=ntrain, replace=False)` | ★★★ | `ind = np.random.choice(n, size=n//2, replace=False)` | Random validation split |
| `df.iloc[ind]` / `df.drop(ind)` | ★★★ | `train = Default.iloc[ind]; test = Default.drop(ind)` | Index-based train/test split |
| `from sklearn.model_selection import train_test_split` | ★★★ | `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)` | Sklearn train/test split |
| `from sklearn.model_selection import KFold` | ★★★ | `kfold = KFold(n_splits=10, shuffle=True, random_state=1)` | K-Fold cross-validation object |
| `from sklearn.model_selection import cross_val_score` | ★★★ | `scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')` | CV score — note **negative** MSE |
| LOOCV (manual loop) | ★★★ | `for i in range(n): train=data.drop(i); test=data.iloc[i:i+1]; ...` | Leave-One-Out cross-validation (implemented manually in ex55) |
| `from sklearn.model_selection import LeaveOneOut` | ★★ | `loo = LeaveOneOut(); for train_idx, test_idx in loo.split(X): ...` | Sklearn LOOCV — mentioned as the efficient alternative |
| `np.trace(conf_mat) / n` | ★★★ | Accuracy from a confusion matrix | |

### Bootstrap

| Command / Pattern | Importance | Syntax Example | Notes |
|---|---|---|---|
| `np.random.choice(n, size=n, replace=True)` | ★★★ | `indboot = np.random.choice(n, size=n, replace=True)` | Draw bootstrap sample **with replacement** |
| Bootstrap loop | ★★★ | `for b in range(B): ind = np.random.choice(n, n, replace=True); aboot[b] = statistic(x[ind], y[ind])` | Repeat B times, compute statistic each time |
| `np.quantile(aboot, [0.025, 0.975])` | ★★★ | `ci = np.quantile(aboot, [0.025, 0.975])` | Bootstrap 95% percentile confidence interval |
| `np.var(x, ddof=1)` | ★★ | `vx = np.var(x, ddof=1)` | Sample variance (ddof=1 = unbiased) |
| `np.cov(x, y, ddof=1)[0, 1]` | ★★ | `cxy = np.cov(x, y, ddof=1)[0, 1]` | Sample covariance |
| `np.std(x, ddof=1)` | ★★ | `s = np.std(x, ddof=1)` | Sample standard deviation |
| `np.mean(x)` | ★★★ | Mean of array | Needed for normal CI: `m - 1.96*s/sqrt(n)` |

---

## Module 4: Time Series and Regularization (Lectures 6–7)

### Lecture 6: Time Series / Bootstrap (stocks.csv)

| Command / Pattern | Importance | Syntax Example | Notes |
|---|---|---|---|
| `pd.read_csv("stocks.csv", sep=';')` | ★ | Load delimited CSV | |
| `ax.plot(series)` | ★★ | `ax.plot(Weekly['Today'])` | Time series plot |
| Custom `alpha_min(x, y)` function | ★★★ | `def alpha_min(x,y): vx=np.var(x,ddof=1); vy=np.var(y,ddof=1); cxy=np.cov(x,y,ddof=1)[0,1]; return (vy-cxy)/(vx+vy-2*cxy)` | Optimal portfolio weight — the exam's bootstrap example |
| Bootstrap SD estimate | ★★★ | `np.std(aboot, ddof=1)` or `np.sqrt(np.var(aboot, ddof=1))` | Standard error of a bootstrapped statistic |

### Lecture 7: Regularization — Ridge and Lasso (Exercise 6.11 — Boston dataset)

| Command / Pattern | Importance | Syntax Example | Notes |
|---|---|---|---|
| `import sklearn.linear_model as skl` | ★★★ | `import sklearn.linear_model as skl` | Alias for sklearn linear models |
| `skl.RidgeCV(alphas=alphas)` | ★★★ | `ridge_cv = skl.RidgeCV(alphas=np.logspace(-4,4,200)); ridge_cv.fit(X_train, y_train)` | Ridge with built-in CV to find best alpha |
| `ridge_cv.alpha_` | ★★★ | `alphamin = ridge_cv.alpha_` | Best regularization parameter from CV |
| `skl.Ridge(alpha=alphamin).fit(X, y)` | ★★★ | `ridgemin = skl.Ridge(alpha=alphamin); ridgemin.fit(X_train, y_train)` | Fit final Ridge model |
| `ridgemin.coef_` | ★★★ | `ridge_coef = ridgemin.coef_` | Ridge coefficients (no intercept term) |
| `ridgemin.intercept_` | ★★ | `np.concatenate(([ridgemin.intercept_], ridgemin.coef_))` | Prepend intercept to coef array |
| `skl.LassoCV(alphas=alphas)` | ★★★ | `lasso_cv = skl.LassoCV(alphas=np.logspace(-4,4,200)); lasso_cv.fit(X_train, y_train)` | Lasso with CV — finds best alpha |
| `lasso_cv.alpha_` | ★★★ | `lassoalpha = lasso_cv.alpha_` | Best alpha |
| `skl.Lasso(alpha=lassoalpha).fit(X, y)` | ★★★ | `lasso = skl.Lasso(alpha=lassoalpha); lasso.fit(X_train, y_train)` | Fit final Lasso |
| `lasso.coef_` | ★★★ | `lasso.coef_` | Lasso coefficients — **zeroed-out = selected away** |
| `from sklearn.preprocessing import StandardScaler` | ★★★ | `scaler = StandardScaler(with_mean=True, with_std=True)` | Standardize features — **required before Ridge/Lasso** |
| `scaler.fit_transform(X_train)` | ★★★ | `X_train_stand = scaler.fit_transform(X_train)` | Fit + transform training data |
| `scaler.transform(X_test)` | ★★★ | `X_test_stand = scaler.transform(X_test)` | Transform test data using **training** mean/std |
| `model.predict(X_test)` | ★★★ | `pred_ridge = ridgemin.predict(X_test)` | Predictions from Ridge/Lasso |
| MSE comparison | ★★★ | `MSE = (ytest - pred)**2).mean()` | Compare OLS vs Ridge vs Lasso on test set |
| `sm.OLS(y, sm.add_constant(X)).fit()` | ★★★ | OLS baseline for comparison | |
| `pd.DataFrame(coef, columns=['OLS','Ridge','Lasso'])` | ★★ | Side-by-side coefficient comparison | Shows shrinkage effect |

---

## Module 5: Non-Linear Methods (Lectures 8–9)

### Lecture 8: Polynomial Regression and Step Functions (Exercise 7.6 — Wage dataset)

| Command / Pattern | Importance | Syntax Example | Notes |
|---|---|---|---|
| `np.vander(x, k+1, increasing=True)` | ★★★ | Used inside formula: `smf.ols(f'wage ~ np.vander(age, {k+1}, increasing=True)', data=data)` | Creates polynomial features of degree k |
| `smf.ols(formula, data).fit()` | ★★★ | `reg = smf.ols('wage ~ np.vander(age, 3, increasing=True)', data=train).fit()` | Polynomial regression |
| `pd.cut(df['age'], k)` | ★★★ | `df['age_cut'] = pd.cut(df['age'], k)` | Step function — cut continuous var into k bins |
| `smf.ols('wage ~ age_cut', data=df).fit()` | ★★★ | Step function OLS using cut variable | |
| `reg.predict(test)` | ★★★ | `pred = reg.predict(test)` | Predict on test data (formula models handle new data) |
| `reg.predict(test.assign(age_cut=pd.cut(test['age'], k)))` | ★★★ | Must supply `age_cut` for new test data when using step regression | |
| `from sklearn.preprocessing import PolynomialFeatures` | ★★★ | `poly = PolynomialFeatures(degree=2, include_bias=False)` | sklearn polynomial features |
| `poly.fit_transform(X_train)` | ★★★ | `X_train_poly = poly.fit_transform(X_train)` | Fit and create polynomial design matrix |
| `poly.transform(X_test)` | ★★★ | `X_test_poly = poly.transform(X_test)` | Apply same transformation to test set |
| MSE loop over degrees | ★★★ | `for k in range(1, K+1): reg=polyreg(k,train); pred=reg.predict(test); MSE[k-1]=np.mean((test['wage']-pred)**2)` | Find optimal polynomial degree via test MSE |
| `np.argmin(MSE) + 1` | ★★★ | `optimal_k = np.argmin(MSE) + 1` | Best degree (note +1 because 0-indexed) |
| `plt.step(x[o], pred[o], where='mid')` | ★ | Step function plot | |
| `plt.scatter(age, wage, color='gray')` | ★ | Gray scatter for data background | |

### Lecture 9: GAMs — Backfitting Algorithm (gam_example)

| Command / Pattern | Importance | Syntax Example | Notes |
|---|---|---|---|
| GAM backfitting (manual KNN smoother) | ★★★ | `yd = y - y.mean(); f1 = b0 + np.array([knn(x0, x1, yd, K) for x0 in x1])` | Backfitting algorithm: cycle between predictors, fixing others |
| Residual computation | ★★★ | `res = y - f1; res = res - res.mean()` | Residuals for next predictor; **demean** before next step |
| Iteration to convergence | ★★★ | Outer `for i in range(10): ...` — repeat until `f1`, `f2` stop changing | Confirms convergence of backfitting |
| `from pygam import LinearGAM, s, f` | ★★ | `gam = LinearGAM(s(0) + s(1)).fit(X, y)` | pygam library for smooth GAMs (if used) |
| Partial effects plot | ★★ | `gam.plot_partial_dependence()` | Visualize individual smoother contributions |

---

## Module 6: Tree-Based Methods (Lectures 10–11)

### Lecture 10: Decision Trees (Exercise 8.8 a–c — Carseats dataset)

| Command / Pattern | Importance | Syntax Example | Notes |
|---|---|---|---|
| `from sklearn.tree import DecisionTreeRegressor, plot_tree` | ★★★ | `from sklearn.tree import DecisionTreeRegressor, plot_tree` | Regression trees |
| `from sklearn.metrics import mean_squared_error` | ★★★ | `from sklearn.metrics import mean_squared_error` | Standard error metric |
| `train_test_split(df, test_size=0.5, random_state=123)` | ★★★ | `train, test = train_test_split(Carseats, test_size=0.5, random_state=123)` | Split whole DataFrame |
| `X_train = train.drop(columns='Sales')` | ★★★ | Extract features | |
| `pd.get_dummies(X, drop_first=True)` | ★★★ | Encode categoricals before feeding to sklearn | |
| `DecisionTreeRegressor(random_state=123, min_samples_leaf=30)` | ★★★ | `tree1 = DecisionTreeRegressor(random_state=123, min_samples_leaf=30)` | Key params: `min_samples_leaf`, `max_depth`, `random_state` |
| `tree.fit(X_train, y_train)` | ★★★ | `tree1.fit(X_train, y_train)` | Fit tree |
| `plot_tree(tree, feature_names=X.columns, precision=2, fontsize=10, ax=ax)` | ★★★ | `plot_tree(tree1, feature_names=X_train.columns, precision=2, fontsize=10, ax=ax)` | Visualize tree structure |
| `tree.predict(X_test)` | ★★★ | `pred1 = tree1.predict(X_test)` | Predictions |
| `mean_squared_error(y_test, pred)` | ★★★ | `testMSE = mean_squared_error(y_test, pred1)` | Test MSE |
| **Tree pruning via cost-complexity** | ★★★ | See below | |
| `tree.cost_complexity_pruning_path(X, y)` | ★★★ | `cv_tree = tree2.cost_complexity_pruning_path(X_train, y_train)` | Get candidate alpha values for pruning |
| `cv_tree.ccp_alphas` | ★★★ | `{'ccp_alpha': cv_tree.ccp_alphas}` passed to `GridSearchCV` | Grid of complexity penalties |
| `GridSearchCV(tree, {'ccp_alpha': alphas}, cv=kfold, scoring='neg_mean_squared_error')` | ★★★ | `grid = GridSearchCV(tree2, {'ccp_alpha': cv_tree.ccp_alphas}, refit=True, cv=kfold, scoring='neg_mean_squared_error')` | CV to find best pruning strength |
| `grid.fit(X_train, y_train)` | ★★★ | `G = grid.fit(X_train, y_train)` | Runs CV + refits best model |
| `grid.best_estimator_` | ★★★ | `pruned_tree = G.best_estimator_` | Best (pruned) tree |
| `grid.best_params_['ccp_alpha']` | ★★ | `best_alpha = grid.best_params_['ccp_alpha']` | Optimal alpha value |
| `KFold(5, shuffle=True, random_state=10)` | ★★★ | `kfold = KFold(5, shuffle=True, random_state=10)` | 5-fold CV object |

### Lecture 11: Ensemble Methods (Exercise 8.8 d–e — Carseats, continued)

| Command / Pattern | Importance | Syntax Example | Notes |
|---|---|---|---|
| `from sklearn.ensemble import RandomForestRegressor` | ★★★ | `from sklearn.ensemble import RandomForestRegressor` | Both bagging and random forests |
| **Bagging** = Random Forest with `max_features = p` (all features) | ★★★ | `bag = RandomForestRegressor(n_estimators=500, max_features=X_train.shape[1], random_state=123)` | Use ALL features per split |
| **Random Forest** = `max_features = m < p` | ★★★ | `rf = RandomForestRegressor(n_estimators=500, max_features=3, random_state=123)` | Use subset of features — reduces correlation between trees |
| `model.fit(X_train, y_train)` | ★★★ | `bag.fit(X_train, y_train)` | Fit ensemble |
| `model.predict(X_test)` | ★★★ | `pred_bag = bag.predict(X_test)` | Predictions |
| `model.feature_importances_` | ★★★ | `importances = pd.Series(bag.feature_importances_, index=X_train.columns)` | Variable importance (impurity-based) |
| `importances.sort_values().plot.barh()` | ★★★ | `importances.sort_values().plot.barh(); plt.title('Variable Importance')` | Horizontal bar chart of feature importances |
| `from sklearn.ensemble import GradientBoostingRegressor` | ★★★ | `from sklearn.ensemble import GradientBoostingRegressor` | Gradient boosting |
| `GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=2)` | ★★★ | `boost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=2, random_state=123)` | Key params: `n_estimators`, `learning_rate`, `max_depth` |
| `boost.fit(X_train, y_train)` | ★★★ | `boost.fit(X_train, Y_train)` | Fit boosting model |
| `boost.predict(X_test)` | ★★★ | `y_pred = boost.predict(X_test)` | Predictions |
| `np.mean((y_test - pred)**2)` | ★★★ | Compare MSE: bagging vs RF vs boosting | |

---

## Module 7: Support Vector Machines (Lecture 12)

*Exercise 9.7 — Auto dataset*

| Command / Pattern | Importance | Syntax Example | Notes |
|---|---|---|---|
| `from sklearn.svm import SVC` | ★★★ | `from sklearn.svm import SVC` | Support Vector Classifier |
| `Auto['high'] = (Auto['mpg'] > Auto['mpg'].median()).astype(int)` | ★★★ | Create binary target from continuous variable | |
| `Auto.drop(columns=['mpg'])` | ★★★ | Remove the variable used to create the target | |
| **Linear SVM:** `SVC(kernel='linear', C=10)` | ★★★ | `svm_linear = SVC(kernel='linear', C=10); svm_linear.fit(X, y)` | `C`: regularization — large C = less margin, more fit |
| **Polynomial SVM:** `SVC(kernel='poly', C=..., degree=...)` | ★★★ | `SVC(kernel='poly', degree=3, C=10)` | Non-linear kernel |
| **Radial (RBF) SVM:** `SVC(kernel='rbf', C=..., gamma=...)` | ★★★ | `SVC(kernel='rbf', C=10, gamma=1)` | Most powerful non-linear kernel |
| `GridSearchCV(SVC(kernel='linear'), {'C': [0.01,0.1,1,10,100]}, cv=5)` | ★★★ | `tuneC = GridSearchCV(SVC(kernel='linear'), {'C': [0.01,...,5000]}, cv=2, scoring='accuracy'); tuneC.fit(X, y)` | Tune C via cross-validation |
| `tuneC.best_params_` | ★★★ | `print(tuneC.best_params_)` | Best hyperparameters |
| `tuneC.best_estimator_` | ★★★ | `bestmod = tuneC.best_estimator_` | Best fitted model |
| Joint grid search (poly) | ★★★ | `param_poly = {'C': [...], 'degree': [1,2,3,4]}; GridSearchCV(SVC(kernel='poly'), param_poly, cv=5).fit(X, y)` | Tune both C and degree |
| Joint grid search (RBF) | ★★★ | `param_rbf = {'C': [...], 'gamma': [0.5,1,2,3,4]}; GridSearchCV(SVC(kernel='rbf'), param_rbf, cv=5).fit(X, y)` | Tune both C and gamma |
| `svm.predict(X)` | ★★★ | `predbest = bestmod.predict(X)` | Class predictions |
| `from sklearn.metrics import confusion_matrix` | ★★★ | `confusion_matrix(y, pred)` | Confusion matrix (sklearn version) |
| `from ISLP.svm import plot as plot_svm` | ★★ | `plot_svm(X, y, svm_model, features=(0,1), ax=ax)` | Decision boundary plot (ISLP helper) |
| `from ISLP import confusion_table` | ★★ | `confusion_table(pred, y)` | ISLP confusion table helper |
| `plt.scatter(x1, x2, c=model.predict(X))` | ★★ | Color scatter plot by predicted class | |

---

## Module 8: Neural Networks and Model Comparison (Lecture 13)

*Ames Housing dataset — comprehensive multi-model comparison*

| Command / Pattern | Importance | Syntax Example | Notes |
|---|---|---|---|
| `from sklearn.neural_network import MLPRegressor` | ★★★ | `from sklearn.neural_network import MLPRegressor` | Multi-layer perceptron |
| **Single hidden layer:** `MLPRegressor(hidden_layer_sizes=(15,), activation='relu', max_iter=100000)` | ★★★ | `nn = MLPRegressor(hidden_layer_sizes=(15,), activation='relu', max_iter=100000, random_state=0)` | One hidden layer with 15 neurons |
| **Multi-layer:** `MLPRegressor(hidden_layer_sizes=(15, 8), ...)` | ★★★ | `dnn = MLPRegressor(hidden_layer_sizes=(15,8), activation='relu', max_iter=100000, random_state=0)` | Two hidden layers (deep neural net) |
| `nn.fit(X_train, y_train)` | ★★★ | With **standardized** features; `nn.fit(X_train_stand, Y_train)` | **Always standardize before neural networks** |
| `nn.predict(X_test)` | ★★★ | `y_pred = nn.predict(X_test_stand)` | Predictions |
| Custom `MAE` function | ★★★ | `def MAE(ypred, y): return np.mean(np.abs(y - ypred))` | Mean Absolute Error — used in Lecture 13 |
| Full model comparison pipeline | ★★★ | OLS → Ridge → Lasso → Bagging → RF → Boosting → NN | The pattern of the Ames exercise: compare all methods on same train/test |
| `X.isna().sum().sort_values(ascending=False).head(10)` | ★★ | Check missing values in Ames data | |
| `X = X.fillna(0)` | ★★ | Fill missing structural values with 0 | |
| `pd.get_dummies(df, drop_first=True); df.astype(float)` | ★★★ | Encode all categoricals; convert to float | Required before sklearn models |

---

## Compulsory Assignment — Additional Patterns

The compulsory assignment (bike rentals dataset) reinforces and combines all modules.

| Command / Pattern | Importance | Syntax Example | Notes |
|---|---|---|---|
| `data['col'].describe().to_frame().T` | ★★ | `display(data['cnt'].describe().to_frame().T)` | Formatted single-variable summary |
| `df['temp'].corr(df['atemp'])` | ★★ | Check near-perfect collinearity between two predictors | |
| `df[col].astype(str)` | ★★ | `df['mnth'] = df['mnth'].astype(str)` | Force numeric code → categorical string before `get_dummies` |
| `from sklearn.linear_model import LinearRegression` | ★★★ | `mlr = LinearRegression(); mlr.fit(X_train, y_train)` | sklearn OLS |
| `from sklearn.metrics import mean_squared_error` | ★★★ | `mse = mean_squared_error(y_test, y_pred)` | |
| `np.sqrt(mse)` | ★★ | RMSE — more interpretable than MSE | |
| `from sklearn.pipeline import Pipeline` | ★★★ | `pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())])` | Chain preprocessing + model |
| `pipe.set_params(knn__n_neighbors=k)` | ★★★ | `pipe.set_params(knn__n_neighbors=k)` | Update pipeline params using `__` notation |
| `pipe.fit(X_train_cv, y_train_cv)` | ★★★ | Inside CV loop | |
| `pipe.predict(X_test_cv)` | ★★★ | Inside CV loop | |
| KNN K-selection loop | ★★★ | `for k in Ks: pipe.set_params(knn__n_neighbors=k); fold_errors=[...]; cv_mse.append(np.mean(fold_errors))` | Manual CV loop across K values |
| `from sklearn.neighbors import KNeighborsRegressor` | ★★★ | `KNeighborsRegressor(n_neighbors=k)` | KNN for regression |
| `PolynomialFeatures(degree=2, include_bias=False)` | ★★★ | `poly = PolynomialFeatures(degree=2, include_bias=False)` | Degree-2 expansion |
| `poly.fit_transform(X_train)` + `poly.transform(X_test)` | ★★★ | Fit only on train, transform both | **Never fit on test data** |

---

## Quick-Reference: Key Parameter Choices

These are the most exam-critical parameter decisions to internalize:

| Model | Key Parameter(s) | How to choose |
|---|---|---|
| Polynomial regression | `degree` (k) | CV/test MSE — minimize over k |
| Ridge regression | `alpha` (λ) | `RidgeCV` with log-spaced grid |
| Lasso regression | `alpha` (λ) | `LassoCV` with log-spaced grid |
| KNN (regression/classification) | `n_neighbors` (K) | CV loop, pick K that minimizes CV error |
| Decision Tree | `ccp_alpha` (pruning) | `cost_complexity_pruning_path` + `GridSearchCV` |
| Random Forest | `n_estimators`, `max_features` | `n_estimators=500`; `max_features=sqrt(p)` for classification, `p/3` for regression |
| Bagging | `max_features=p` | Use ALL features (= no random subspace) |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth` | Small `learning_rate` + large `n_estimators`; `max_depth=2` typically |
| SVM | `C`, `kernel`, (`gamma`/`degree`) | `GridSearchCV` over a grid |
| Neural Network | `hidden_layer_sizes`, `activation` | Experiment; `relu` standard; always standardize X |

---

## Quick-Reference: Confusion Matrix and Classification Metrics

```python
# pd.crosstab approach (ISLP style)
conf = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])

# sklearn approach
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)

# Accuracy
accuracy = np.mean(y_true == y_pred)
# or: np.trace(conf) / conf.values.sum()
```

---

## Quick-Reference: Full Pipeline Pattern (Exam Template)

```python
# 1. Load and prepare data
from ISLP import load_data
data = load_data("DatasetName")
X = data.drop(columns='target')
y = data['target']
X = pd.get_dummies(X, drop_first=True)  # if categoricals exist

# 2. Train/test split
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

# 3. (Optional) Standardize — required for Ridge, Lasso, KNN, NN
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)       # use transform (not fit_transform) on test!

# 4. Fit model
model = SomeModel(params...)
model.fit(X_train, y_train)          # or X_train_s for regularized models

# 5. Evaluate
pred = model.predict(X_test)
MSE  = np.mean((y_test - pred)**2)
MAE  = np.mean(np.abs(y_test - pred))
# For classifiers:
acc  = np.mean(y_test == pred)
```

---

## Quick-Reference: Cross-Validation Patterns

```python
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

# K-Fold CV score (negative MSE)
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
cv_mse = -scores.mean()

# GridSearchCV (tune hyperparameter)
grid = GridSearchCV(model, {'param': [v1, v2, v3]}, cv=kfold, scoring='neg_mean_squared_error', refit=True)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
best_param  = grid.best_params_

# LOOCV (manual — as in ex55)
for i in range(n):
    train = data.drop(i)
    test  = data.iloc[i:i+1]
    ...
```

---

## Quick-Reference: Bootstrap Pattern

```python
B = 1000
n = len(x)
stat_boot = np.zeros(B)

for b in range(B):
    idx = np.random.choice(n, size=n, replace=True)   # sample WITH replacement
    stat_boot[b] = my_statistic(x[idx], y[idx])

# Standard error
se_boot = np.std(stat_boot, ddof=1)

# 95% Confidence interval
ci = np.quantile(stat_boot, [0.025, 0.975])
```

---

*Built from: ex38.py (L2), ex413_a_d.py (L3), ex413_e_g.py (L4), ex55.py (L5), lecture_6_knowledge.txt (L6), ex611.py (L7), ex76.ipynb (L8), gam_example.py/.ipynb (L9), ex88ac.ipynb (L10), ex88.ipynb (L11), ex97.ipynb/.py (L12), ames.ipynb/.py (L13), course_app_w.qmd (Compulsory Assignment)*
