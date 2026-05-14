
# Read data and needed packages
library(ISLR)
library(tree)

# (a) Split the data into training and test data
set.seed(123)
n=nrow(Carseats)
ntrain=floor(n/2)
ind=sample(1:n,size=ntrain)
train=Carseats[ind,]
test=Carseats[-ind,]

# (b) Fit tree to training data, plot and interpret, compute testMSE
tree1=tree(Sales~.,data=train)
plot(tree1)
text(tree1,pretty=0)
# Complex results. Follow the tree and go left if you answer "yes" to
# the questions, e.g., is shelve location bad or medium? Right if "no"
# The highest average sales, 12.02, is in the case when shelve location
# is good and price is below 109.5.
pred1=predict(tree1,newdata=test)
testMSE1=mean((test$Sales-pred1)^2)
testMSE1

# (c) Use CV to determine optimal tree complexity
cv.tree1 =cv.tree(tree1,FUN=prune.tree)
plot(cv.tree1$size,cv.tree1$dev,type='b')
tree2=prune.tree(tree1,best=5)
pred2=predict(tree2,newdata=test)
testMSE2=mean((test$Sales-pred2)^2)
testMSE2
# In this case pruning did not help

# Go into help function to show the different tuning parameters
# Check particularly tree.controll and mincut
# A "manual" approach using CV to determine minimum node size
# Validation set approach split train into train2/test2
set.seed(123)
ntrain2=floor(ntrain/2)
ind2=sample(1:ntrain,size=ntrain2)
train2=train[ind2,]
test2=train[-ind2,]
mseres=data.frame(mincut=2,testMSE=0)
for(k in 2:50)
{
  tree3=tree(Sales~.,data=train2,control
           =tree.control(nobs=ntrain2,mincut=k))
  pred3=predict(tree3,newdata=test2)
  testMSE3=mean((test2$Sales-pred3)^2)
  mseres[k-1,]=cbind(k,testMSE3)
}
plot(mseres,type="b")
# Check original test data with mincut=8
tree4=tree(Sales~.,data=train,control
           =tree.control(nobs=ntrain,mincut=12))
pred4=predict(tree4,newdata=test)
testMSE4=mean((test$Sales-pred4)^2)
testMSE4

# (d) Use bagging to predict and compute variable importance
MSE=function(mod,newdata=test)
{
  pred=predict(mod,newdata=test)
  testMSE=mean((test$Sales-pred)^2)
  testMSE
}
library(randomForest)
set.seed(123)
names(Carseats)
ncol(Carseats)
bag.Carseats=randomForest(Sales~.,data=train,mtry=10,importance=TRUE)
varImpPlot(bag.Carseats,type=1)
MSE5=MSE(bag.Carseats)
MSE5
testMSE1
MSE(tree1)

# (e) Random forests
rf.Carseats=randomForest(Sales~.,data=train,mtry=3,importance=TRUE)
par(mfrow=c(1,2))
varImpPlot(bag.Carseats,type=1)
varImpPlot(rf.Carseats,type=1)
MSE(rf.Carseats)
# Bagging seem to work better than RF for these data











