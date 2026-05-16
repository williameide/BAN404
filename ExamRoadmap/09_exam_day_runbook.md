# BAN404 Exam-Day Runbook

> **Print this out or keep it open on exam day. Follow every step in order.**

---

## T-15 MINUTES: Before the Exam Starts

### Install/verify packages (run this once)
```r
pkgs <- c("MASS","ISLR2","car","class","e1071","leaps","glmnet",
          "pls","splines","gam","tree","randomForest","gbm",
          "ROCR","Matrix","insuranceData")
for(p in pkgs) if(!requireNamespace(p,quietly=TRUE)) install.packages(p)
```

### Open files in this order
1. Open `ExamRoadmap/templates/task2_exam_template.qmd` → Save As `exam_[date]_answers.qmd`
2. Open `ExamRoadmap/snippets/ban404_core_snippets.R` in a second editor tab
3. Open `ExamRoadmap/04_task1_mastery.qmd` and scroll to "Method Recognition Table"
4. Download the attached CSV file (if any) to your working directory

### Set up your answer file header
```r
# First cell in your exam answer file:
library(MASS); library(ISLR2); library(e1071); library(glmnet)
library(gam); library(tree); library(randomForest); library(gbm)
set.seed(123)
```

---

## EXAM MINUTES 0-10: Read and Plan

**Do NOT start coding yet.**

1. Read all questions in both tasks completely
2. For Task 1: identify the method from the code (use the Recognition Table below)
3. For Task 2: note what the target variable is, what dataset, and what models are requested
4. Assign rough time to each sub-question based on point value
5. Note any question you find easy (do those first)

### Method Recognition Table (Task 1 fast lookup)

| Code pattern | Method |
|-------------|--------|
| `la * sum(b1^2)` | Ridge regression |
| `la * sum(abs(b1))` | Lasso regression |
| `sample(..., replace=TRUE)` | Bootstrap |
| `X[-i,]` in a loop | LOOCV |
| `abs(x-x0)` + `order(d)[1:K]` | KNN local regression |
| `smooth.spline(x1, y-f2)` loop | Backfitting (GAM) |
| `exp(eta)/(1+exp(eta))` | Logistic sigmoid |
| `nlminb(b1start, f, ...)` | Numerical optimizer for a custom objective |
| `gam(y ~ s(x1) + ...)` | GAM with smoothing splines |
| `r <- y - fhat` in sequential loop | Boosting residual update |

---

## TASK 1 WORKFLOW (70 minutes budget for 50 points)

### Sub-question: "Explain what this R function does"
**Time budget: 15 minutes**

1. Scan code for recognition cues (table above)
2. Name the method in your first sentence
3. Explain each major step:
   - What is the input? What is the output?
   - What mathematical formula does it compute? (write the formula)
   - What does the loop do?
   - What does the return value represent?
4. Add 1 sentence on WHY this method is used

**Template:**
> "The function `[name]` implements **[METHOD]**. It computes [formula/objective]. The loop [describe loop logic]. The function returns [what]. This method is used because [1-sentence motivation]."

---

### Sub-question: "Use the function on this data"
**Time budget: 15 minutes**

1. Load the data as shown in the exam
2. Call the function with the given parameters
3. Print/display the result
4. Write 1-2 sentences interpreting the output

```r
# Template:
X <- as.matrix(Xy[, -1])
y <- as.matrix(Xy[, 1])
result <- function_from_exam(X, y, la=VALUE)
print(result)  # or hist(result), or cat("Optimal:", result)
```

---

### Sub-question: "Compare model A vs model B"
**Time budget: 15 minutes**

1. Fit both models
2. Compute the same metric for both (MSE, sum of squared coefficients, etc.)
3. Print a side-by-side table
4. Write a conclusion with specific numbers

```r
# Template:
b_ols   <- lm(y ~ X)$coef[-1]
b_ridge <- g(X, y, la=optimal_la)
cat("Sum sq OLS:", round(sum(b_ols^2), 4))
cat("Sum sq Ridge:", round(sum(b_ridge^2), 4))
# Ridge shrinks from X to Y — this demonstrates variance reduction
```

---

### Sub-question: "Find optimal tuning parameter"
**Time budget: 15 minutes**

1. Define a grid of values (start broad: `seq(0, 5000, length=15)`)
2. Run LOOCV for each value using `sapply`
3. Plot MSE vs parameter
4. Report the optimal value

```r
grid <- seq(0, 5000, length.out=15)
mses <- sapply(grid, function(param) loo_function(param, X, y))
plot(grid, mses, type="l", main="LOOCV MSE")
opt  <- grid[which.min(mses)]
cat("Optimal:", opt)
```

---

### Sub-question: "Bootstrap"
**Time budget: 15 minutes**

```r
set.seed(1); B <- 1000; n <- length(y); res <- numeric(B)
for(b in 1:B) res[b] <- STATISTIC(y[sample(1:n, replace=TRUE)])
hist(res, breaks=50, main="Bootstrap distribution")
cat("SE:", sd(res))
cat("95% CI (normal):", mean(res) + c(-1,1)*1.96*sd(res))
cat("95% CI (pct):", quantile(res, c(0.025, 0.975)))
```

---

## TASK 2 WORKFLOW (90 minutes budget for 50 points)

### Sub-question 2a: Data cleaning + EDA (15-20 minutes)

```r
# MUST DO IN ORDER:
# 1. Load
data <- read.csv("[filename]")

# 2. Inspect
str(data); summary(data); head(data, 3)

# 3. Create binary target if needed
# data$target <- as.numeric(data$count >= 1)

# 4. Remove leakage
data <- data[, !(names(data) %in% c("id", "raw_target", "post_event_var"))]

# 5. Factor encode
# data$cat_col <- as.factor(data$cat_col)

# 6. Train/test split
set.seed([SEED FROM EXAM]); n <- nrow(data)
idx <- sample(1:n, floor(n/2)); train <- data[idx,]; test <- data[-idx,]

# 7. Quick EDA
prop.table(table(train$target))  # Class balance
# Boxplots: boxplot(x ~ target, data=train)
# Cross-tables: prop.table(table(train$cat, train$target), margin=1)
```

**EDA written answer checklist:**
- [ ] State which variables were removed and why
- [ ] State which were converted to factors and why
- [ ] Name 2-3 most promising predictors from plots/tables with specific numbers

---

### Sub-question 2b: Logistic Regression (15 minutes)

```r
logreg <- glm(target ~ ., data=train, family=binomial())
summary(logreg)
exp(coef(logreg))   # Odds ratios
```

**Written answer checklist:**
- [ ] Interpret 2 coefficients (use odds ratio scale)
- [ ] Note which are significant

---

### Sub-question 2c: Evaluate Predictions (15 minutes)

```r
# 1. Check class balance → decide threshold
prop.table(table(train$target))

# 2. Predict
prob <- predict(logreg, newdata=test, type="response")
threshold <- 0.5    # ADJUST if imbalanced

# 3. Evaluate
pred <- ifelse(prob > threshold, 1, 0)
cm   <- table(Actual=test$target, Predicted=pred)
prop.table(cm, margin=1)
acc  <- sum(diag(cm)) / sum(cm)
```

**Written answer checklist:**
- [ ] State the threshold and justify it
- [ ] Report sensitivity (row 2 diagonal) and specificity (row 1 diagonal)
- [ ] Report overall accuracy

---

### Sub-question 2d: Random Forest (15 minutes)

```r
library(randomForest)
train$target <- as.factor(train$target)
test$target  <- as.factor(test$target)
rf <- randomForest(target ~ ., data=train, ntree=300, importance=TRUE)
pred_rf <- predict(rf, newdata=test)
cm_rf   <- table(Actual=test$target, Predicted=pred_rf)
prop.table(cm_rf, margin=1)
varImpPlot(rf)
```

**Written answer checklist:**
- [ ] Compare accuracy to logistic regression (specific numbers)
- [ ] Name top 2-3 important variables from the plot
- [ ] Explain what variable importance measures

---

### Sub-question 2e: Boosting (15 minutes)

```r
library(gbm)
train_bt <- train
train_bt$target <- as.numeric(as.character(train$target))
bt <- gbm(target ~ ., data=train_bt, distribution="bernoulli",
          n.trees=1000, interaction.depth=3, shrinkage=0.01)
prob_bt <- predict(bt, newdata=test, n.trees=1000, type="response")
pred_bt <- ifelse(prob_bt > threshold, 1, 0)
table(Actual=as.numeric(as.character(test$target)), Predicted=pred_bt)
```

---

## POST-ANSWER SANITY CHECKLIST

Before submitting:

### Code checks
- [ ] All code chunks have outputs (no empty outputs)
- [ ] No `Error` messages in any output
- [ ] File can be rendered/knitted without errors

### Content checks
- [ ] Task 1: Named the method in every explain-question
- [ ] Task 1: Showed the R code output, not just the code
- [ ] Task 2: Written answer after EVERY code chunk
- [ ] Task 2: Confusion matrix evaluated with row proportions (not just raw counts)
- [ ] Task 2: Final comparison of models with specific numbers
- [ ] Train/test split was used correctly (fit on train, evaluate on test)

### "Last 10 minutes" priority
If running out of time, prioritize in this order:
1. Finish written answers for questions you already have code for
2. For unfinished code questions: write "If I had time I would [describe approach] and expect [describe result]"
3. Never leave a question completely blank — partial answers get partial credit

---

## Common Emergency Fixes

| Problem | Fix |
|---------|-----|
| `factor has different levels` | `test$x <- factor(test$x, levels=levels(train$x))` |
| RF won't run on target | `train$target <- as.factor(train$target)` |
| gbm won't run on target | `train$target <- as.numeric(as.character(train$target))` |
| Confusion matrix is 1×2 (all same prediction) | Threshold too high/low — try 0.1 or 0.01 |
| `system is computationally singular` | `lm(y ~ . - problemcol, data=train)` |
| `predict.gam` not found | Make sure `library(gam)` is loaded, not `mgcv` |
| `nlminb` doesn't converge | Try different starting values: `rep(0.1, q)` instead of `rep(0, q)` |
| NAs in predictions | `na.omit(train)` before fitting |

---

## The Golden Rules

1. **Name the method first, then explain it** — never describe code without identifying the statistical method
2. **Test errors need test data** — always predict on `test`, not `train`, for evaluation
3. **Always justify threshold choice** — never use 0.5 without acknowledging class balance
4. **Partial credit exists** — write something for every question, even if incomplete
5. **Pragmatic over perfect** — "don't let the perfect model stand in the way for a good one" (exact quote from every BAN404 exam)
