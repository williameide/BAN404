# BAN404 master prompt — maximum output ChatGPT version

You are creating a single, printable, BAN404 exam-survival R cheatsheet for a complete beginner.

Your task is to write the actual final cheatsheet, not a plan, not notes, and not a short summary.

## Output goal
Produce one giant, exam-centered `.qmd` or `.md` style document that helps a BAN404 student with weak R memory still solve the exam well.

The document must let the student:
- load ugly exam csv files safely,
- clean and recode data,
- remove leakage,
- split train/test correctly,
- recognize which method the question is asking for,
- fit the model in R,
- tune the main tuning parameter,
- evaluate it with the correct metric,
- interpret it in words,
- handle small wording changes in the exam.

## Hard requirements
- Use tidyverse-friendly R style everywhere possible.
- Every code block must be runnable, not fragments.
- Every line of code must be commented in plain English.
- Every method section must explain:
  1. what the method does,
  2. why it is used,
  3. what assumptions/weaknesses it has,
  4. what to tune,
  5. how to evaluate it,
  6. where it has appeared in BAN404 exams before.
- Assume the reader may forget package names, formula syntax, dummy-variable handling, threshold logic, and evaluation metrics.
- Prefer the safest BAN404 exam workflow over clever or advanced alternatives.
- Use beginner-friendly wording.
- Be extremely explicit.
- Do not ask clarifying questions.
- Do not summarize what you will do.
- Just write the full cheatsheet.

## BAN404 exam context you must build from
The old exams show these recurring patterns:
- 2025 Task 1: ridge code explanation, OLS vs ridge, LOOCV for lambda, bootstrap variance/CI, GAM for nonlinearity.
- 2025 Task 2: claim variable creation, leakage removal, train/test split, logistic regression, threshold choice, random forest, boosted tree.
- 2024 Task 1: local KNN-style regression, LOOCV for K, fitted line plotting, multi-predictor distance, backfitting explanation.
- 2024 Task 2: airline satisfaction with both explanatory and operational prediction perspectives; logistic regression, tree, random forest, variable importance.
- 2023 Task 1: OLS, LASSO, regression tree, random forest for continuous response.
- 2023 Task 2: bootstrap CI for probability, logistic regression, random forest for churn.
- 2022: logistic regression, classification tree, threshold CV, GAM, bagging.
- 2021: bootstrap-heavy tasks, coefficient bootstrap SE, GAM, logistic regression, boosted trees.

## High-priority BAN404 methods
Write full sections for:
1. OLS / linear regression
2. Logistic regression
3. LOOCV / k-fold CV
4. Bootstrap
5. Ridge
6. LASSO
7. GAM / smoothing splines / backfitting
8. KNN / local regression
9. Regression tree
10. Classification tree
11. Bagging
12. Random forest
13. Boosting

Also add shorter fallback sections for:
- LDA / QDA
- SVM
- PCA / PCR
- K-means

## Mandatory structure
Use this exact high-level structure:
1. How to use this document in the exam
2. BAN404 model recognition table
3. Universal data loading cookbook
4. Universal wrangling and split templates
5. Metric map
6. High-probability method playbooks
7. Lower-probability fallback sections
8. Question wording to action map
9. Written answer templates
10. Mini exam-day checklist

## In the data loading cookbook you must include
- comma-separated CSV
- semicolon-separated CSV
- decimal comma case
- skip rows if header not on top row
- missing value handling
- factor conversion
- leakage removal
- ID removal
- creating binary target variables
- `model.matrix()` for glmnet
- when standardization is needed

## In the metric map you must include
For regression:
- MSE
- RMSE
- R-squared
- adjusted R-squared

For classification:
- confusion matrix
- accuracy
- error rate
- sensitivity
- specificity
- threshold discussion

For unsupervised:
- variance explained
- scree plot
- within-cluster sum of squares / elbow plot

## For each high-probability method section, use this sub-structure
A. What BAN404 wording signals the method  
B. When to use it  
C. Response type  
D. Core theory in simple words  
E. Assumptions and weaknesses  
F. What to tune  
G. How to evaluate  
H. Full runnable R template  
I. Line-by-line comments tied to theory  
J. Typical BAN404 exam twists and how code changes  
K. Short written answer templates  
L. Common mistakes  
M. Which BAN404 years used it

## Additional BAN404-specific rules
- Clearly separate explanatory questions from predictive/operational questions.
- Warn aggressively about leakage and future-unavailable variables.
- Explain scaling rules explicitly:
  - usually important for KNN, glmnet, PCA, SVM, k-means
  - usually not the main concern for trees, RF, boosting
- Prefer the exact BAN404-safe answer when several answers are possible.
- Tie examples to exam years whenever possible.

## Final instruction
Return only the final cheatsheet document.
Do not give a plan.
Do not explain your process.
Write the full document.
