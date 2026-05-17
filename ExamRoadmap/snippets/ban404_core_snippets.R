# BAN404 Core R Snippets — full runnable mini-examples
# Purpose: copy, adjust, and run complete examples rather than isolated fragments.
# Style: tidyverse-first for reading, cleaning, summarising, and plotting.

library(tidyverse)                                                          # main data workflow used throughout the file

# ============================================================================
# 0. Shared demo datasets
# ============================================================================

set.seed(123)                                                               # reproducible splits and resampling

cars_cls <- mtcars |>
  tibble::rownames_to_column(var = "model") |>
  as_tibble() |>
  mutate(
    target      = factor(am, levels = c(0, 1), labels = c("automatic", "manual")), # binary classification target
    cylinders   = factor(cyl),                                                        # categorical predictor version
    engine_shape = factor(vs, levels = c(0, 1), labels = c("v-shaped", "straight"))
  ) |>
  select(model, target, mpg, disp, hp, wt, qsec, cylinders, engine_shape)

cars_reg <- mtcars |>
  as_tibble() |>
  mutate(
    cylinders = factor(cyl),                                               # keep one factor to show mixed-type workflows
    gears     = factor(gear)
  ) |>
  select(mpg, disp, hp, wt, qsec, cylinders, gears)

# ============================================================================
# 1. EDA starter block
# ============================================================================

cars_cls |> glimpse()                                                       # inspect types before plotting or modeling
cars_cls |> summary()                                                       # quick one-object summary

cars_cls |>
  group_by(target) |>
  summarise(
    mean_hp = mean(hp),                                                     # average horsepower per class
    mean_wt = mean(wt),                                                     # average weight per class
    n       = n(),                                                          # class count
    .groups = "drop"
  )

ggplot(cars_cls, aes(x = target, y = hp, fill = target)) +                 # numeric predictor vs binary target
  geom_boxplot(alpha = 0.8) +
  labs(title = "Horsepower by transmission", x = "Transmission", y = "Horsepower") +
  theme_minimal()

cars_cls |>
  count(cylinders, target) |>
  group_by(cylinders) |>
  mutate(prop = n / sum(n)) |>
  ungroup() |>
  ggplot(aes(x = cylinders, y = prop, fill = target)) +                     # categorical predictor vs binary target
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Transmission share within cylinder groups", x = "Cylinders", y = "Share") +
  theme_minimal()

corr_tbl <- cars_cls |>
  select(where(is.numeric)) |>
  cor() |>
  as.data.frame() |>
  tibble::rownames_to_column("var1") |>
  pivot_longer(-var1, names_to = "var2", values_to = "correlation")

ggplot(corr_tbl, aes(x = var1, y = var2, fill = correlation)) +             # quick collinearity scan
  geom_tile() +
  geom_text(aes(label = round(correlation, 2)), size = 3) +
  scale_fill_gradient2(low = "firebrick", mid = "white", high = "steelblue", midpoint = 0) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ============================================================================
# 2. Bootstrap example — sampling distribution of the mean
# ============================================================================

boot_mean <- function(x, B = 2000, seed = 123) {
  set.seed(seed)                                                            # same bootstrap results when you rerun
  n <- length(x)                                                            # original sample size
  boot_stats <- numeric(B)                                                  # pre-allocate storage for speed and clarity

  for (b in seq_len(B)) {
    x_boot <- sample(x, size = n, replace = TRUE)                           # replace = TRUE is what makes this bootstrap resampling
    boot_stats[b] <- mean(x_boot)                                           # save one bootstrap replicate of the statistic
  }

  tibble(
    estimate     = mean(x),                                                 # observed statistic from the original sample
    se_boot      = sd(boot_stats),                                          # bootstrap standard error
    ci_low_pct   = quantile(boot_stats, 0.025),                             # percentile lower bound
    ci_high_pct  = quantile(boot_stats, 0.975)                              # percentile upper bound
  )
}

boot_result <- boot_mean(cars_reg$mpg)                                      # run the bootstrap on mpg
boot_result

tibble(boot_stat = replicate(2000, mean(sample(cars_reg$mpg, replace = TRUE)))) |>
  ggplot(aes(x = boot_stat)) +                                              # visualise the bootstrap distribution
  geom_histogram(bins = 35, fill = "steelblue", color = "white") +
  labs(title = "Bootstrap distribution of mean(mpg)", x = "Bootstrap mean", y = "Count") +
  theme_minimal()

# ============================================================================
# 3. Ridge-style LOOCV example — full manual structure
# ============================================================================

f_ridge <- function(b1, X, y, lambda) {
  sum((y - X %*% b1)^2) + lambda * sum(b1^2)                                # RSS + lambda * L2 penalty
}

g_ridge <- function(X, y, lambda) {
  q <- ncol(X)                                                              # number of predictors
  y_centered <- y - mean(y)                                                 # remove intercept from y before optimization
  X_centered <- scale(X, center = TRUE, scale = FALSE)                      # demean columns but do not scale by sd
  start_vals <- rep(0, q)                                                   # optimization starts at zero coefficients
  opt <- nlminb(start = start_vals, objective = f_ridge, X = X_centered, y = y_centered, lambda = lambda)
  opt$par                                                                   # return the minimizing coefficient vector
}

loo_ridge <- function(lambda, X, y) {
  n <- nrow(X)                                                              # one loop per observation
  preds <- numeric(n)                                                       # store prediction for each left-out row

  for (i in seq_len(n)) {
    X_train <- X[-i, , drop = FALSE]                                        # training predictors for this split
    y_train <- y[-i]                                                        # training response for this split
    x_test_centered <- X[i, ] - colMeans(X_train)                           # center with training means only; using test-row statistics here would leak information
    y_train_mean <- mean(y_train)                                           # same intercept logic as inside g_ridge

    b1 <- g_ridge(X = X_train, y = y_train, lambda = lambda)                # fit ridge on the n - 1 training observations
    preds[i] <- y_train_mean + x_test_centered %*% b1                       # add back the training intercept for the left-out prediction
  }

  mean((y - preds)^2)                                                       # LOOCV MSE for this lambda on the original y scale
}

X_ridge <- cars_reg |>
  select(disp, hp, wt, qsec) |>
  as.matrix()                                                               # ridge code expects a numeric matrix

y_ridge <- cars_reg$mpg                                                     # regression target

lambda_grid <- seq(0, 200, by = 10)                                         # candidate penalty values
loo_tbl <- tibble(
  lambda = lambda_grid,
  loo_mse = map_dbl(lambda_grid, ~ loo_ridge(.x, X = X_ridge, y = y_ridge)) # compute one LOOCV error per lambda
)

loo_tbl |> filter(loo_mse == min(loo_mse))                                  # best lambda in the grid

ggplot(loo_tbl, aes(x = lambda, y = loo_mse)) +                             # visual tuning plot
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue") +
  labs(title = "LOOCV MSE by ridge penalty", x = "Lambda", y = "LOOCV MSE") +
  theme_minimal()

# ============================================================================
# 4. Logistic regression — full train/test pipeline
# ============================================================================

set.seed(123)
idx_cls <- sample.int(nrow(cars_cls), size = floor(0.5 * nrow(cars_cls)))   # 50/50 split for classification example
train_cls <- cars_cls |> slice(idx_cls)
test_cls  <- cars_cls |> slice(-idx_cls)

log_mod <- glm(target ~ mpg + hp + wt + cylinders,                           # baseline logistic model with mixed predictors
               data = train_cls,
               family = binomial())

summary(log_mod)                                                             # inspect coefficients and p-values
exp(coef(log_mod))                                                           # odds ratios are often easier to explain in words

prob_log <- predict(log_mod, newdata = test_cls, type = "response")         # probabilities for the positive class
threshold <- 0.50                                                            # default threshold; lower it if positives are rare
pred_log <- if_else(prob_log >= threshold, "manual", "automatic") |>        # convert probabilities to class labels
  factor(levels = levels(test_cls$target))

cm_log <- table(Actual = test_cls$target, Predicted = pred_log)              # confusion matrix for evaluation
cm_log
prop.table(cm_log, margin = 1)                                               # row proportions = class-wise performance
sum(diag(cm_log)) / sum(cm_log)                                              # overall accuracy

# ============================================================================
# 5. Random forest — compare against logistic regression
# ============================================================================

library(randomForest)                                                        # classic package used repeatedly in BAN404 material

rf_mod <- randomForest(target ~ mpg + hp + wt + cylinders + engine_shape,    # same task, more flexible model
                       data = train_cls,
                       ntree = 500,
                       mtry = 2,
                       importance = TRUE)

rf_mod                                                                        # prints OOB error and class information
pred_rf <- predict(rf_mod, newdata = test_cls)
cm_rf <- table(Actual = test_cls$target, Predicted = pred_rf)
cm_rf
prop.table(cm_rf, margin = 1)
sum(diag(cm_rf)) / sum(cm_rf)

varImpPlot(rf_mod)                                                           # variable importance plot for interpretation

# ============================================================================
# 6. GAM example — when you suspect non-linearity
# ============================================================================

library(gam)                                                                 # smooth terms via s()

gam_mod <- gam(mpg ~ s(hp) + s(wt) + qsec, data = cars_reg)                 # smooth hp and wt, keep qsec linear
summary(gam_mod)
plot(gam_mod, se = TRUE, col = "steelblue")                                 # inspect whether the smooth terms look curved

pred_gam <- predict(gam_mod, newdata = cars_reg)
mean((cars_reg$mpg - pred_gam)^2)                                            # in-sample MSE just for quick comparison

# ============================================================================
# 7. Threshold tuning table — useful when classes are imbalanced
# ============================================================================

threshold_tbl <- tibble(threshold = seq(0.10, 0.90, by = 0.05)) |>
  mutate(
    pred = map(threshold, ~ if_else(prob_log >= .x, "manual", "automatic")), # predicted labels for each threshold
    accuracy = map_dbl(pred, ~ mean(factor(.x, levels = levels(test_cls$target)) == test_cls$target))
  )

threshold_tbl                                                               # use this to justify a threshold choice if asked
