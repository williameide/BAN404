# BAN404 Predicted Exam 2026 – R Overview

This file mirrors the original exam overview but all implementation examples are R-first and aligned with BAN404 course tooling.

## 1) Core BAN404 package stack

```r
library(tidyverse)
library(janitor)
library(glmnet)
library(gam)
library(tree)
library(randomForest)
library(gbm)
```

## 2) Typical Task 1 (methodology) workflow in R

- Build OLS baseline with `lm()`.
- Build regularized model with `glmnet()`.
- Tune lambda with cross-validation (`cv.glmnet`) or manual K-fold loops.
- Use bootstrap via `sample(..., replace = TRUE)` and `quantile()`.
- Handle non-linearity using `gam::gam()` with `s(x, df = 4)`.

Example skeleton:

```r
df <- readr::read_csv("task1_data.csv", show_col_types = FALSE) %>% janitor::clean_names()
X <- model.matrix(y ~ . , data = df)[, -1]
y <- df$y

ols <- lm(y ~ ., data = df)
cv_lasso <- cv.glmnet(X, y, alpha = 1)
gam_fit <- gam::gam(y ~ x1 + s(x2, df = 4) + s(x3, df = 4) + x4 + x5 + x6, data = df)
```

## 3) Typical Task 2 (applied classification) workflow in R

- Recode logical/character fields to factors.
- Remove perfectly collinear minute/charge duplicates.
- Split train/test with fixed seed.
- Fit `glm(..., family = binomial())`, `randomForest()`, and `gbm()`.
- Compare with accuracy + sensitivity + specificity + AUC.

Example skeleton:

```r
cust <- readr::read_csv("customer_data.csv", show_col_types = FALSE) %>% janitor::clean_names()
cust <- cust %>%
  mutate(
    churn = factor(churn, levels = c(FALSE, TRUE), labels = c("No", "Yes")),
    international_plan = factor(international_plan),
    voice_mail_plan = factor(voice_mail_plan)
  ) %>%
  select(-total_day_charge, -total_eve_charge, -total_night_charge, -total_intl_charge)

set.seed(42)
idx <- sample(seq_len(nrow(cust)), size = floor(0.8 * nrow(cust)))
train <- cust[idx, ]
test  <- cust[-idx, ]
```

## 4) Exam writing reminders

- Always explain what the model answers ("why" vs "predict").
- Always justify split/validation choice.
- Always discuss threshold trade-off (false positives vs false negatives).
- Always connect findings back to business context.
