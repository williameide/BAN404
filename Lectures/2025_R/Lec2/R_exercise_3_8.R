
# Load data
library(ISLR)

# First look at data
head(Auto)
sapply(Auto,class)
plot(mpg~horsepower,data=Auto)

# (a)
reg1=lm(mpg~horsepower,data=Auto)
summary(reg1)
# The null hypothesis that horsepower is not associated with mpg is rejected
# with a p-value smaller than 2e-16
# The strength of the relationship, measured as R^2, is 0.60.
# The relationship is negative. An increase with one horsepower is associated
# with a decrease in fuel usage of -0.16 gallon
pred1=predict(reg1,data.frame(horsepower=98))
pred1
# Confidence interval. Closeness of prediction and f(X)
pred2=predict(reg1,data.frame(horsepower=98),interval = "confidence")
pred2
# Prediction interval. Closeness of prediction and Y
pred3=predict(reg1,data.frame(horsepower=98),interval = "prediction")
pred3

# (b)
plot(mpg~horsepower,data=Auto)
abline(reg1,col="red")

# (c)
par(mfrow=c(1,2))
plot(reg1,which=c(1,2))
graphics.off()
# Systematic pattern in plot of residuals on predictions
# Heteroscedasticity and positive residuals for large and small 
# predictions. Nonlinear relationship? This might cause significance levels
# and levels of confidence intervals not being as they are supposed to be, e.g.,
# 5% or 95%.
# Some deviation from normality. Only a problem for prediction intervals.

# Extra: Plot predictions with confidence and prediction intervals
pred4=predict(reg1,interval="confidence")
pred5=predict(reg1,interval="prediction")
plot(mpg~horsepower,data=Auto,ylim=range(pred4,pred5,Auto$mpg))
o=order(Auto$horsepower)
x=Auto$horsepower[o]
lines(x,pred4[o,1],col="red")
lines(x,pred4[o,2],col="blue")
lines(x,pred4[o,3],col="blue")
lines(x,pred5[o,2],col="magenta")
lines(x,pred5[o,3],col="magenta")

# Using KNN
knn=function(x0,x,y,K=3)
{
  d=abs(x-x0)
  o=order(d)[1:K]
  ypred=mean(y[o])
  return(ypred)
}
x=Auto$horsepower
y=Auto$mpg
n=length(y)
o=order(x)
xo=x[o]
yo=y[o]
yp=matrix(0,n,1)
for(i in 1:n) yp[i]=knn(xo[i],xo,yo,K=20)
plot(x,y)
lines(xo,yp,col="blue")




