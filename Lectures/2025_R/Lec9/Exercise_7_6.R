
# Read data
library(ISLR)
summary(Wage)

# (a) Use validation set approach to determine best polynomial
set.seed(123) # Try with other seeds
n=nrow(Wage)
ntrain=floor(n/2)
ind=sample(1:n,size=ntrain)
train=Wage[ind,]
test=Wage[-ind,]
polyreg=function(k,age,wage) lm(wage~poly(age,k),data=train)
K=20
MSE=matrix(0,K,1)
for(k in 1:K)
{
  reg=polyreg(k,age,wage)
  pred=predict(reg,newdata=test)
  MSE[k]=mean((test$wage-pred)^2)
}
MSE
which.min(MSE)
# CV indicates a polynomial of order 7

reg1=polyreg(1,age,wage)
reg2=polyreg(2,age,wage)
reg3=polyreg(3,age,wage)
regK=polyreg(K,age,wage)
summary(regK)
# Hypothesis testing of coefficients indicates a polynomial of order 2

anova(reg1,reg2,reg3)
# ANOVA indicates that we reject the
# null hypothesis of order 1 against the alternative of order 2
# The null hypothesis of order two is not rejeted against
# the alternative of order 3. Conclusion: order 2.

plot(wage~age,data=Wage)
reg7=lm(wage~poly(age,7),data=Wage)
reg2=lm(wage~poly(age,2),data=Wage)
o=order(Wage$age)
lines(Wage$age[o],predict(reg7)[o],col="red")
lines(Wage$age[o],predict(reg2)[o],col="blue")

# (b) Step function regression. Use cross-validation to determine number
# of cuts
stepreg=function(k,age,wage) lm(wage~cut(age,k),data=train)
K=20
MSE=matrix(0,K-1,1)
for(k in 2:K)
{
  reg=stepreg(k,age,wage)
  pred=predict(reg,newdata=test)
  MSE[k-1]=mean((test$wage-pred)^2)
}
MSE
which.min(MSE)
step9=lm(wage~cut(age,9),data=Wage)
plot(wage~age,data=Wage)
lines(Wage$age[o],predict(step9)[o],type="s",col="red")
