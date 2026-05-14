
# Read data
library(ISLR2)

# First look at the data
hist(Boston$crim)
boxplot(Boston$crim)
plot(crim~as.factor(chas),data=Boston)
plot(crim~.,data=Boston)

# Split traning/test
set.seed(1234)
n <- nrow(Boston)
ntrain <- floor(n/2)
ind <- sample(1:n,size=ntrain)
train <- Boston[ind,]
test <- Boston[-ind,]
X <- as.matrix(Boston[,-1])
y <- Boston[,1]
Xtrain <- X[ind,]
Xtest <- X[-ind,]
ytrain <- y[ind]
ytest <- y[-ind]

# (a)
library(leaps)
bestsubset <- regsubsets(crim~.,data=train,nvmax = 12)
plot(bestsubset,scale="adjr2")

# Ridge
library(glmnet)
lambdamin <- cv.glmnet(Xtrain,ytrain,alpha=0,nfold=n)$lambda.min
ridge <- glmnet(Xtrain,ytrain,alpha=0,lambda = lambdamin)
coef(ridge)

# LASSO
lambdamin <- cv.glmnet(Xtrain,ytrain,alpha=1,nfold=n)$lambda.min
lasso <- glmnet(Xtrain,ytrain,alpha=1,lambda = lambdamin)
coef(lasso)

# OLS
ols <- lm(crim~.,data=train)
cbind(coef(ols),coef(ridge),coef(lasso))

# PCR
library(pls)
pcr.fit <- pcr(crim~.,data=train)
summary(pcr.fit)

# (b)
pred_ols <- predict(ols,newdata=test)
MSE_ols <- mean((ytest-pred_ols)^2)
MSE_ols

pred_ridge <- predict(ridge,newx=Xtest)
MSE_ridge <- mean((ytest-pred_ridge)^2)
MSE_ridge

pred_lasso <- predict(lasso,newx=Xtest)
MSE_lasso <- mean((ytest-pred_lasso)^2)
MSE_lasso

pcr.fit <- pcr(crim~.,data=train,ncomp=5)
summary(pcr.fit)
pred_pcr <- predict(pcr.fit,newdata=test)
MSE_pcr <- mean((ytest-pred_pcr)^2)
MSE_pcr























