# BAN404 master prompt — maximum output Claude version

You are helping produce a final printable BAN404 R exam compendium for a beginner.

You should act like a careful course-specific teaching writer, not like a terse coding assistant.

Your job is to produce one complete, exam-centered BAN404 cheatsheet document in markdown/qmd style.

## Primary objective
Create a full R exam compendium that is so explicit, over-commented, and BAN404-specific that a student with shaky R memory can still navigate the exam and produce competent answers.

## Important writing style
- Be exhaustive, structured, and practical.
- Be explicit rather than concise.
- Use beginner-friendly language.
- Explain why each step is there, not just what it does.
- Ground the material in BAN404 theory and old-exam patterns.
- Keep the code tidyverse-friendly wherever possible.
- Make every code block runnable.
- Comment every line of code.
- Never assume the reader remembers packages, syntax, or evaluation metrics.

## Do NOT do these things
- Do not provide a plan instead of the final output.
- Do not ask clarifying questions.
- Do not produce short snippets without context.
- Do not give generic ML notes disconnected from BAN404.
- Do not skip the old exam context.

## BAN404 context you must use
Use these recurring BAN404 patterns as the backbone of the cheatsheet:
- 2025 Task 1: ridge objective function, demeaning, OLS comparison, LOOCV for lambda, bootstrap variance and CI, GAM and R-squared.
- 2025 Task 2: insurance claim classification, binary target creation, leakage removal, train/test split, logistic regression, threshold choice, random forest, boosted tree.
- 2024 Task 1: KNN/local regression function, LOOCV for K, fitted curve plotting, multi-predictor distance, backfitting theory.
- 2024 Task 2: airline satisfaction with one explanatory perspective and one operational prediction perspective.
- 2023 Task 1: OLS, LASSO, regression tree, random forest for bill prediction.
- 2023 Task 2: bootstrap CI for churn probability, logistic regression, random forest.
- 2022: logistic regression, classification tree, threshold tuning, GAM, bagging.
- 2021: bootstrap distributions, coefficient bootstrap SE, GAM, logistic regression, boosted trees.

## High-priority methods to cover in depth
1. OLS / linear regression
2. Logistic regression
3. LOOCV / k-fold CV
4. Bootstrap
5. Ridge regression
6. LASSO
7. GAM / smoothing splines / backfitting
8. KNN / local regression
9. Regression tree
10. Classification tree
11. Bagging
12. Random forest
13. Boosting

## Lower-priority fallback methods to cover briefly but usefully
- LDA / QDA
- SVM
- PCA / PCR
- K-means

## Required overall document structure
Please write the final compendium with these main sections:
1. How to use this document in the exam
2. BAN404 method recognition table
3. Universal data loading cookbook
4. Universal wrangling and split templates
5. Metric map
6. High-probability method playbooks
7. Lower-probability fallback sections
8. Question wording to action map
9. Written answer templates
10. Mini exam-day checklist

## Required content inside the compendium

### In the method recognition table
Include code tells and wording tells such as:
- `sum((y-X%*%b)^2) + la*sum(b^2)` → Ridge
- `sum(abs(b))` / `glmnet alpha=1` → LASSO
- `X[-i, ]` → LOOCV
- `sample(..., replace=TRUE)` → Bootstrap / Bagging
- `order(d)[1:K]` → KNN / local regression
- `s(x)` / `smooth.spline` → GAM / backfitting
- `glm(..., family = binomial)` → Logistic regression
- `tree(...)` → Decision tree
- `randomForest(...)` → Random forest / bagging approximation
- `gbm(...)` → Boosting
- `prcomp(...)` → PCA
- support vectors / cost / gamma → SVM
- `kmeans(...)` → K-means

### In the data loading cookbook
Include safe templates for:
- normal csv import
- semicolon csv import
- decimal comma import
- skip rows if the table does not start at the top
- NA handling
- factor conversion
- ID removal
- leakage removal
- binary target creation
- `model.matrix()` for glmnet
- scaling rules

### In the metric map
Include what the metric means, when to use it, and a short interpretation sentence template.
Cover:
- MSE, RMSE, R-squared, adjusted R-squared
- confusion matrix, accuracy, error rate, sensitivity, specificity, threshold logic
- variance explained, scree plots, elbow plots

### For each high-probability method
Use this repeated sub-structure:
A. What exam wording signals this method  
B. When to use it  
C. Response type  
D. Core theory in simple words  
E. Assumptions and weaknesses  
F. What to tune  
G. How to evaluate  
H. Full runnable R template  
I. Line-by-line comments tied to theory  
J. Typical BAN404 exam variations  
K. Written answer templates  
L. Common mistakes / traps  
M. Past BAN404 years where it appeared

## Additional BAN404 priorities
- Distinguish clearly between explanatory tasks and predictive tasks.
- Distinguish clearly between training data, test data, and CV data roles.
- Explain leakage repeatedly and concretely.
- Explain why scaling matters for KNN / glmnet / PCA / SVM / k-means.
- Explain why scaling is usually less central for tree-based methods.
- Use the old exams to decide method order and emphasis.

## Final instruction
Return only the finished BAN404 cheatsheet document.
No planning text.
No preamble.
No meta commentary.
Write the actual final material.
