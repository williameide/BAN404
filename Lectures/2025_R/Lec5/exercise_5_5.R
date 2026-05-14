
# READ DATA
library(ISLR)

# (a) 
logreg=glm(default~income+balance,data=Default,family = binomial())
summary(logreg)
# income and balance significant on the 1%-level

# (b)
# Split into training and test data
n=nrow(Default)
ntrain=floor(n/2)
ind=sample(1:n,size=ntrain)
train=Default[ind,]
test=Default[-ind,]
# Fit a logistic regression to the training data
logreg2=glm(default~income+balance,data=train,family=binomial())
summary(logreg2)
# Predict for the test observations
prob2=predict(logreg2,newdata=test,type="response")
pred2=prob2>0.5
# Compute the validation set error
conf_mat=table(test$default,pred2)
sum(diag(conf_mat))/ntrain

## (c)
# The error is around 0.97 each time but varies since the
# training/test division is not identical each time

# (d)
# As above but with default~income+balance+student
logreg3=glm(default~income+balance+student,data=train,family=binomial())
summary(logreg3)
# Predict for the test observations
prob3=predict(logreg3,newdata=test,type="response")
pred3=prob3>0.5
# Compute the validation set error
conf_mat=table(test$default,pred3)
sum(diag(conf_mat))/ntrain

# Extra task: LOOCV
n<-nrow(Default)
nsmall <- 500
small <- Default[sample(1:n,nsmall),]
pred4 <- matrix(0,nsmall,1)
# Takes a few minutes
for(i in 1:nsmall)
{
  training <- small[-i,]
  test <- small[i,]
  logreg4 <- glm(default~income+balance+student,data=train,family=binomial())
  prob4 <- predict(logreg4,newdata=test,type="response")
  pred4[i] <- prob4>0.5
}
conf_mat=table(small$default,pred4)
sum(diag(conf_mat))/nsmall









