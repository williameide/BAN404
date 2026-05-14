
# READ DATA
library(ISLR)

# (a) First look at data
head(Weekly)
names(Weekly)
plot.ts(Weekly$Today)
table(Weekly$Direction)
attach(Weekly)
mean(Volume)
by(Volume,Direction,mean)
by(Volume,Direction,t.test)
plot(Volume~Direction,data=Weekly)
cor(Weekly[,c(2:6,8)])
acf(Weekly$Today)

# First try with a linear probability model
formula1=Direction~Volume+Lag1+Lag2+Lag3+Lag4+Lag5
linreg=lm(formula1,data=Weekly) # Problem with factor y
sapply(Weekly,class)
y=as.numeric(Weekly$Direction) # Create numerical variable
head(y)
y=y-1 # Create dummy (0-1) variable
Weekly$y=y
head(y)
head(Weekly$Direction)
formula2=y~Volume+Lag1+Lag2+Lag3+Lag4+Lag5
linreg=lm(formula2,data=Weekly)
summary(linreg)
head(predict(linreg))
pred1=predict(linreg)>0.5
table(pred1)
conf_matrix=table(Weekly$Direction,pred1)
conf_matrix
accuracy=sum(diag(conf_matrix))/sum(conf_matrix)
accuracy

# (b)
logreg=glm(formula1,data=Weekly,family = binomial())
summary(logreg)
# Lag2 significant on the 5%-level

# (c)
pred2=predict(logreg) 
head(pred2) # Negative numbers? Log-odds
pred3=predict(logreg,type="response")
head(pred2)
head(log(pred3/(1-pred3)))
conf_matrix=table(Weekly$Direction,pred3>0.5)
conf_matrix
accuracy=sum(diag(conf_matrix))/sum(conf_matrix)
accuracy
prop.table(conf_matrix,margin=1)

# (d)
# Split into training and test data
ind=Weekly$Year<=2008
train=Weekly[ind,] # Obs until 2008
test=Weekly[!ind,] # Obs after 2008
logreg2=glm(Direction~Lag2,data=train,family=binomial())
pred4=predict(logreg2,newdata=test,type="response")
conf_matrix=table(test$Direction,pred4>0.5)
conf_matrix
accuracy=sum(diag(conf_matrix))/sum(conf_matrix)
accuracy
# What if we always guess "up".
table(train$Direction)
table(test$Direction)
accuracy=61/104
accuracy
prop.table(conf_matrix,margin=1)
# Row percentages, e.g., P(predict up | actual up)=0.918

# (e)
# Linear discriminant analysis
library(MASS)
lda1=lda(Direction~Lag2,data=train)
pred5=predict(lda1,newdata = test)
names(pred5)
head(pred5$class)
head(pred5$posterior)
conf_matrix=table(test$Direction,pred5$class)
conf_matrix
accuracy=sum(diag(conf_matrix))/sum(conf_matrix)
accuracy
prop.table(conf_matrix,margin=1) 

# (f)
# Quadratic discriminant analysis
qda1=qda(Direction~Lag2,data=train)
pred6=predict(qda1,newdata = test)
names(pred6)
head(pred6$class)
head(pred6$posterior)
conf_matrix=table(test$Direction,pred6$class)
conf_matrix
accuracy=sum(diag(conf_matrix))/sum(conf_matrix)
accuracy
prop.table(conf_matrix,margin=1)

# (g)
library(class)
# Different structure of function, must supply x and y-variables
pred7=knn(as.matrix(train$Lag2),as.matrix(test$Lag2),
          train$Direction,k=1)
conf_matrix=table(test$Direction,pred7)
conf_matrix
prop.table(conf_matrix,margin=1)
accuracy=sum(diag(conf_matrix))/sum(conf_matrix)
accuracy

# (h)
# Logistic regression with 5 lags
logreg2=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5,family=
              binomial(),data=train)
pred8=predict(logreg2,newdata=test,type="response")>0.5
conf_matrix=table(test$Direction,pred8)
conf_matrix
prop.table(conf_matrix,margin=1)
accuracy=sum(diag(conf_matrix))/sum(conf_matrix)
accuracy
# Logistic with Lag2 only predicts better

# KNN with k=10
pred9=knn(as.matrix(train$Lag2),as.matrix(test$Lag2),
          train$Direction,k=10)
conf_matrix=table(test$Direction,pred9)
conf_matrix
prop.table(conf_matrix,margin=1)
accuracy=sum(diag(conf_matrix))/sum(conf_matrix)
accuracy
# Better than KNN with K=1 but worse than logistic regression and LDA