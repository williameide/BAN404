# BAN404 Core R Snippets — Copy-Paste Reference
# Use as RStudio snippets or paste directly during exam
# All code assumes: X = predictor matrix, y = response, train/test = data frames

# ============================================================
# 0. SETUP — Run once at start of every exam
# ============================================================
library(MASS); library(ISLR2); library(e1071); library(glmnet)
library(gam); library(tree); library(randomForest); library(gbm)
library(splines); library(pROC)

set.seed(123)  # Always set seed for reproducibility

# Read data
data <- read.csv("data.csv")       # comma-separated
data <- read.csv2("data.csv")      # semicolon-separated (European)

# Train/test split (50/50)
n     <- nrow(data)
idx   <- sample(1:n, size=floor(n/2))
train <- data[idx, ]
test  <- data[-idx, ]

# Convert categories to factors
# data$col <- as.factor(data$col)
# Check structure
str(data); summary(data)

# ============================================================
# 1. BOOTSTRAP — SE and 95% CI
# ============================================================
bootstrap_stat <- function(y, stat_fn, B=1000, seed=1) {
  set.seed(seed)
  n   <- length(y)
  res <- numeric(B)
  for(b in 1:B) {
    yb    <- y[sample(1:n, size=n, replace=TRUE)]
    res[b] <- stat_fn(yb)
  }
  list(
    se          = sd(res),
    ci_normal   = mean(res) + c(-1,1)*1.96*sd(res),
    ci_pct      = quantile(res, c(0.025, 0.975)),
    dist        = res
  )
}
# Usage: bootstrap_stat(y, var)      # Variance
#        bootstrap_stat(y, mean)     # Mean
#        bootstrap_stat(y, sd)       # Std dev (volatility)
# Plot: hist(result$dist, breaks=50)

# Bootstrap for regression coefficient
bootstrap_coef <- function(X, y, B=1000, coef_idx=2) {
  n   <- nrow(X); res <- numeric(B)
  for(b in 1:B) {
    idx   <- sample(1:n, replace=TRUE)
    mod   <- lm(y[idx] ~ X[idx,])
    res[b] <- coef(mod)[coef_idx]
  }
  list(se=sd(res), ci=quantile(res, c(0.025, 0.975)), dist=res)
}

# ============================================================
# 2. LOOCV — Leave-One-Out Cross Validation
# ============================================================
# Generic LOOCV for any model with lambda
loo_ridge <- function(la, X, y) {
  n  <- nrow(X); q <- ncol(X)
  b0 <- mean(y);  yd <- y - b0
  Xd <- X; for(k in 1:q) Xd[,k] <- X[,k] - mean(X[,k])
  preds <- numeric(n)
  for(i in 1:n) {
    b1       <- g_ridge(X=X[-i,], y=y[-i], la=la)
    preds[i] <- Xd[i,] %*% b1
  }
  mean((yd - preds)^2)
}

# Use it to find optimal lambda:
lavec <- seq(0, 5000, length.out=20)
mses  <- sapply(lavec, function(la) loo_ridge(la, X, y))
la_opt <- lavec[which.min(mses)]
plot(lavec, mses, type="l", main="LOOCV MSE vs lambda")

# ============================================================
# 3. K-FOLD CROSS VALIDATION
# ============================================================
kfold_cv <- function(X, y, K=5) {
  n      <- length(y)
  folds  <- sample(rep(1:K, length.out=n))
  errors <- numeric(K)
  for(i in 1:K) {
    tX <- X[folds != i, , drop=FALSE]; ty <- y[folds != i]
    vX <- X[folds == i, , drop=FALSE]; vy <- y[folds == i]
    b  <- solve(t(tX)%*%tX) %*% t(tX)%*%ty
    errors[i] <- mean((vy - vX%*%b)^2)
  }
  mean(errors)
}

# ============================================================
# 4. RIDGE REGRESSION — Manual (exam style)
# ============================================================
f_ridge <- function(b1, X, y, la) {
  sum((y - X%*%b1)^2) + la*sum(b1^2)
}

g_ridge <- function(X, y, la) {
  q  <- ncol(X); b0 <- mean(y); yd <- y - b0
  Xd <- X; for(k in 1:q) Xd[,k] <- X[,k] - mean(X[,k])
  opt <- nlminb(rep(0,q), f_ridge, X=Xd, y=yd, la=la)
  opt$par
}

# Compare OLS vs ridge
b_ols   <- lm(y ~ X)$coef[-1]
b_r0    <- g_ridge(X, y, la=0)
b_r1000 <- g_ridge(X, y, la=1000)
cbind(b_ols, b_r0, b_r1000)
sum(b_ols^2); sum(b_r1000^2)   # Ridge shrinks sum of squared coefficients

# ============================================================
# 5. LASSO — with glmnet
# ============================================================
# Requires X as matrix
X_mat <- as.matrix(X)
cv_lasso   <- cv.glmnet(X_mat, y, alpha=1)    # alpha=1 = Lasso
la_lasso   <- cv_lasso$lambda.min
lasso_fit  <- glmnet(X_mat, y, alpha=1, lambda=la_lasso)
coef(lasso_fit)                               # Zero coefficients = excluded

# Ridge via glmnet
cv_ridge2  <- cv.glmnet(X_mat, y, alpha=0)   # alpha=0 = Ridge
la_ridge2  <- cv_ridge2$lambda.min

# ============================================================
# 6. OLS — Linear Regression
# ============================================================
model_ols <- lm(y ~ ., data=train)
summary(model_ols)
pred_ols  <- predict(model_ols, newdata=test)
MSE_ols   <- mean((test$y - pred_ols)^2)
R2_ols    <- 1 - MSE_ols / var(test$y)

# Matrix formula
b_hat <- solve(t(X)%*%X) %*% t(X)%*%y

# ============================================================
# 7. KNN LOCAL REGRESSION (from 2024 exam)
# ============================================================
f_knn <- function(x0, x, y, K=3) {
  d  <- abs(x - x0)
  o  <- order(d)[1:K]
  xl <- x[o]; yl <- y[o]
  predict(lm(yl~xl, data=data.frame(xl,yl)), newdata=data.frame(xl=x0))
}

# LOOCV for best K
n    <- length(y)
Kgr  <- 1:15
mseK <- sapply(Kgr, function(K)
  mean(sapply(1:n, function(i) (y[i] - f_knn(x[i], x[-i], y[-i], K))^2))
)
K_opt <- Kgr[which.min(mseK)]

# Plot predictions for optimal K
x_seq  <- seq(min(x), max(x), length=200)
ypreds <- sapply(x_seq, function(x0) f_knn(x0, x, y, K=K_opt))
plot(x, y); lines(x_seq, ypreds, col="red", lwd=2)

# ============================================================
# 8. GAM — Generalized Additive Model
# ============================================================
library(gam)

# Fit: s(x) = smooth spline, linear terms without s()
gam_fit <- gam(y ~ s(x1) + x2 + x3, data=train)
MSE_gam  <- mean(residuals(gam_fit)^2)
plot(gam_fit, se=TRUE, col="blue")   # Visualise smooth terms

# Compare to OLS
MSE_null <- var(train$y)
R2_ols   <- 1 - mean(residuals(lm(y~., data=train))^2) / MSE_null
R2_gam   <- 1 - MSE_gam / MSE_null
cat("R2 OLS:", round(R2_ols,4), "  R2 GAM:", round(R2_gam,4))

# Backfitting (manual — if asked in Task 1)
backfit <- function(x1, x2, y, iters=20) {
  f1 <- f2 <- rep(0, length(y))
  for(i in 1:iters) {
    f1 <- smooth.spline(x1, y-f2)$y
    f2 <- smooth.spline(x2, y-f1)$y
  }
  f1 + f2
}

# ============================================================
# 9. LOGISTIC REGRESSION
# ============================================================
logreg <- glm(churn ~ ., data=train, family=binomial())
summary(logreg)

# Predict probabilities
prob  <- predict(logreg, newdata=test, type="response")
pred  <- ifelse(prob > 0.5, 1, 0)

# Confusion matrix
cm  <- table(Actual=test$churn, Predicted=pred)
acc <- sum(diag(cm)) / sum(cm)
prop.table(cm, margin=1)     # Sensitivity (row 2) and Specificity (row 1)

# Odds ratio interpretation
exp(coef(logreg))            # Multiplicative effect on odds

# ============================================================
# 10. THRESHOLD TUNING (classification)
# ============================================================
thresholds <- seq(0.01, 0.5, by=0.005)
acc_vec    <- sapply(thresholds, function(th) {
  pred_th <- as.numeric(prob > th)
  mean(pred_th == as.numeric(as.character(test$churn)))
})
best_th <- thresholds[which.max(acc_vec)]
plot(thresholds, acc_vec, type="l", main="Accuracy vs Threshold")
abline(v=best_th, col="red")

# ============================================================
# 11. RANDOM FOREST
# ============================================================
library(randomForest)

# Classification RF (y must be factor)
rf_cls <- randomForest(as.factor(churn) ~ ., data=train,
                       mtry=floor(sqrt(ncol(train)-1)),
                       ntree=500, importance=TRUE)

# Regression RF
rf_reg <- randomForest(y ~ ., data=train,
                       mtry=floor((ncol(train)-1)/3),
                       ntree=500, importance=TRUE)

# Evaluate
pred_rf  <- predict(rf_cls, newdata=test)
cm_rf    <- table(Actual=test$churn, Predicted=pred_rf)
prop.table(cm_rf, margin=1)

# Variable importance
varImpPlot(rf_cls)          # Plot
importance(rf_cls)          # Numbers

# ============================================================
# 12. BOOSTING (gbm)
# ============================================================
library(gbm)

# Classification: y must be 0/1 numeric
train_bt      <- train
train_bt$y    <- as.numeric(as.character(train$churn))

bt <- gbm(y ~ ., data=train_bt,
          distribution="bernoulli",
          n.trees=1000, interaction.depth=3,
          shrinkage=0.01, cv.folds=5)
best_trees <- gbm.perf(bt, method="cv")   # or which.min(bt$cv.error)

prob_bt  <- predict(bt, newdata=test, n.trees=best_trees, type="response")
pred_bt  <- ifelse(prob_bt > 0.5, 1, 0)
cm_bt    <- table(Actual=test$churn, Predicted=pred_bt)

# Regression boosting
bt_reg <- gbm(y ~ ., data=train,
              distribution="gaussian",
              n.trees=1000, interaction.depth=3, shrinkage=0.01)

# ============================================================
# 13. DECISION TREE
# ============================================================
library(tree)

# Regression
t_reg  <- tree(y ~ ., data=train)
plot(t_reg); text(t_reg, pretty=0)
pred_t <- predict(t_reg, newdata=test)
MSE_t  <- mean((test$y - pred_t)^2)

# Classification
t_cls  <- tree(as.factor(churn) ~ ., data=train)
plot(t_cls); text(t_cls, pretty=0)
pred_c <- predict(t_cls, newdata=test, type="class")
table(Actual=test$churn, Predicted=pred_c)

# Prune tree
cv_t   <- cv.tree(t_reg)                      # CV to find best size
prune_t <- prune.tree(t_reg, best=cv_t$size[which.min(cv_t$dev)])
plot(prune_t); text(prune_t)

# ============================================================
# 14. DESCRIPTIVE STATISTICS FOR EDA
# ============================================================
# Continuous predictor vs continuous response
plot(train$x1, train$y, xlab="x1", ylab="y")
cor(train$x1, train$y)

# Continuous predictor vs binary response (boxplot)
boxplot(x1 ~ churn, data=train, main="x1 by churn")

# Categorical predictor vs binary response (proportions)
prop.table(table(train$gender, train$churn), margin=1)

# Continuous predictors summary by group
tapply(train$x1, train$churn, summary)

# Full numeric summary
summary(train)
sapply(train[, sapply(train, is.numeric)], function(x) c(mean=mean(x), sd=sd(x)))

# ============================================================
# 15. EVALUATION METRICS SUMMARY
# ============================================================
# MSE and RMSE
MSE  <- mean((y_true - y_pred)^2)
RMSE <- sqrt(MSE)

# Accuracy (classification)
acc <- mean(pred == actual)

# Confusion matrix row proportions
prop.table(table(actual, pred), margin=1)
# Row 1 = sensitivity for class 0 (specificity for class 1)
# Row 2 = sensitivity for class 1 (true positive rate)

# R² formula
R2 <- 1 - mean((y_true - y_pred)^2) / var(y_true)
