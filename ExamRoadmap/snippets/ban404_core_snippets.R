# BAN404 core snippets — exam-first, tidyverse-first, and heavily commented
# Use this file when you need a quick runnable pattern to adapt.

library(tidyverse)      # gives dplyr, ggplot2, readr, tibble, purrr, forcats, stringr
library(randomForest)   # tree ensemble often useful in Task 2
library(gbm)            # boosting model used in several BAN404-style workflows

set.seed(123)           # always set a seed so your split and results are reproducible

# ============================================================
# 1. READ + FIRST CHECK
# ============================================================

# data <- read_csv("data.csv")                               # use for comma-separated exam files
# data <- read_csv2("data.csv")                              # use for semicolon-separated exam files

# glimpse(data)                                              # first look at variable types and example values
# summary(data)                                              # good for spotting strange ranges and missing values
# head(data, 5)                                              # good for quick visual inspection

# ============================================================
# 2. SIMPLE CLEANING PIPELINE
# ============================================================

# data <- data %>%
#   rename_with(~ gsub("\\.", "_", make.names(.x)), everything()) %>% # easier column names without requiring another package
#   mutate(across(where(is.character), as.factor))           # convert character columns to factors

# missing_overview <- data %>%
#   summarise(across(everything(), ~ sum(is.na(.x)))) %>%
#   pivot_longer(cols = everything(), names_to = "variable", values_to = "n_missing") %>%
#   arrange(desc(n_missing))

# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

# index <- sample(seq_len(nrow(data)), size = floor(0.5 * nrow(data)))
# train <- data %>% slice(index)
# test  <- data %>% slice(-index)

# ============================================================
# 4. EDA — DISTRIBUTIONS, GROUP DIFFERENCES, AND CORRELATION
# ============================================================

# Histogram: use for one numeric variable when you want shape / skew / outlier insight
# train %>%
#   ggplot(aes(x = age)) +
#   geom_histogram(bins = 30, fill = "steelblue", color = "white") +
#   labs(title = "Distribution of age", x = "Age", y = "Count")

# Boxplot: use for one numeric predictor against a binary or categorical outcome
# train %>%
#   ggplot(aes(x = target, y = premium, fill = target)) +
#   geom_boxplot(show.legend = FALSE) +
#   labs(title = "Premium by outcome group", x = "Outcome", y = "Premium")

# Proportion bar chart: use for categorical predictor vs target
# train %>%
#   count(region, target) %>%
#   group_by(region) %>%
#   mutate(prop = n / sum(n)) %>%
#   ungroup() %>%
#   ggplot(aes(x = region, y = prop, fill = target)) +
#   geom_col() +
#   labs(title = "Outcome mix within region", x = "Region", y = "Within-region proportion")

# Correlation heatmap: use to check multicollinearity among numeric predictors
# train %>%
#   select(where(is.numeric)) %>%
#   cor(use = "pairwise.complete.obs") %>%
#   as.data.frame() %>%
#   rownames_to_column("var1") %>%
#   pivot_longer(-var1, names_to = "var2", values_to = "correlation") %>%
#   ggplot(aes(x = var1, y = var2, fill = correlation)) +
#   geom_tile() +
#   scale_fill_gradient2(low = "firebrick", mid = "white", high = "navy", midpoint = 0) +
#   theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Rule of thumb:
# - high correlation between predictors can make OLS unstable
# - ridge is usually the safer choice for prediction with correlated predictors
# - lasso is useful when you also want variable selection

# ============================================================
# 5. BOOTSTRAP — FULL RUNNABLE PATTERN
# ============================================================

bootstrap_stat <- function(y, stat_fn, B = 1000, seed = 123) {
  set.seed(seed)                                              # keeps the bootstrap reproducible
  n <- length(y)                                              # number of observations in the original sample
  stat_values <- numeric(B)                                   # pre-allocate storage for speed and clarity

  for (b in seq_len(B)) {                                     # repeat B times to build the bootstrap distribution
    idx <- sample(seq_len(n), size = n, replace = TRUE)       # sample WITH replacement = bootstrap tell
    y_boot <- y[idx]                                          # bootstrap sample of the original observations
    stat_values[b] <- stat_fn(y_boot)                         # compute the statistic of interest on that sample
  }

  tibble(
    mean_stat = mean(stat_values),
    se_boot = sd(stat_values),
    ci_low = quantile(stat_values, 0.025),
    ci_high = quantile(stat_values, 0.975)
  )
}

# Example:
# bootstrap_stat(mtcars$mpg, mean)                            # mean with bootstrap SE and percentile CI
# bootstrap_stat(mtcars$mpg, var)                             # variance with bootstrap SE and percentile CI

# ============================================================
# 6. LOGISTIC REGRESSION — COMPLETE CLASSIFICATION PATTERN
# ============================================================

# Example setup you can adapt quickly:
# data_cls <- mtcars %>%
#   as_tibble(rownames = "car") %>%
#   mutate(am = factor(am, labels = c("automatic", "manual")))
#
# index <- sample(seq_len(nrow(data_cls)), size = floor(0.5 * nrow(data_cls)))
# train_cls <- data_cls %>% slice(index)
# test_cls  <- data_cls %>% slice(-index)
#
# log_fit <- glm(am ~ mpg + wt + hp, data = train_cls, family = binomial())
# summary(log_fit)                                            # coefficient table for significance and sign
# exp(coef(log_fit))                                          # odds ratios are easier to interpret in text
#
# log_prob <- predict(log_fit, newdata = test_cls, type = "response")
# log_pred <- if_else(log_prob > 0.5, "manual", "automatic") %>%
#   factor(levels = levels(test_cls$am))
#
# cm_log <- table(Actual = test_cls$am, Predicted = log_pred)
# cm_log
# prop.table(cm_log, margin = 1)                              # row proportions help discuss sensitivity / specificity
# sum(diag(cm_log)) / sum(cm_log)                             # simple accuracy measure

# ============================================================
# 7. RANDOM FOREST — COMPLETE CLASSIFICATION PATTERN
# ============================================================

# rf_fit <- randomForest(
#   am ~ mpg + wt + hp + cyl,
#   data = train_cls,
#   ntree = 500,
#   mtry = 2,
#   importance = TRUE
# )
#
# rf_pred <- predict(rf_fit, newdata = test_cls)
# cm_rf <- table(Actual = test_cls$am, Predicted = rf_pred)
# cm_rf
# prop.table(cm_rf, margin = 1)
# sum(diag(cm_rf)) / sum(cm_rf)
# importance(rf_fit)
# varImpPlot(rf_fit)

# ============================================================
# 8. BOOSTING — COMPLETE CLASSIFICATION PATTERN
# ============================================================

# train_bt <- train_cls %>%
#   mutate(am_num = if_else(am == "manual", 1, 0)) %>%
#   select(-am)
#
# test_bt <- test_cls %>%
#   mutate(am_num = if_else(am == "manual", 1, 0))
#
# boost_fit <- gbm(
#   am_num ~ mpg + wt + hp + cyl,
#   data = train_bt,
#   distribution = "bernoulli",
#   n.trees = 1000,
#   interaction.depth = 3,
#   shrinkage = 0.01,
#   bag.fraction = 0.8
# )
#
# boost_prob <- predict(boost_fit, newdata = test_bt, n.trees = 1000, type = "response")
# boost_pred <- if_else(boost_prob > 0.5, 1, 0)
# cm_boost <- table(Actual = test_bt$am_num, Predicted = boost_pred)
# cm_boost
# prop.table(cm_boost, margin = 1)
# sum(diag(cm_boost)) / sum(cm_boost)

# ============================================================
# 9. CORRELATED PREDICTORS — WHAT TO SAY
# ============================================================

# If you see strong multicollinearity in the correlation heatmap or unstable OLS coefficients:
# - say OLS may become unstable because highly correlated predictors share similar information
# - say ridge is often preferred when the goal is prediction and many predictors are correlated
# - say lasso is attractive when you also want a simpler model with some coefficients set to zero
