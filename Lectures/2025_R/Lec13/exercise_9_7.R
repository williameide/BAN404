
# Read data
library(ISLR)
summary(Auto)

# (a) Create 0-1 variable for mpg above median
Auto$high=as.factor((Auto$mpg>median(Auto$mpg)))
names(Auto)
Auto=Auto[,-1]

# (b) Use a support vector classifier
library(e1071)
dim(Auto)
formula1=high~horsepower+weight+year
svmfit=svm(formula1,data=Auto,kernel="linear"
           ,cost=10)
plot(svmfit,Auto,weight~year)
set.seed(1)
tuneC=tune(svm,formula1,data=Auto, kernel ="linear",
              ranges =list(cost=c(0.01,0.1,1,10,100,1000,5000)))
summary(tuneC)
bestmod=tuneC$best.model
summary(bestmod)

# (c) Use SVM with radial and polynomial basis kernels
set.seed(1)
tuneCd=tune(svm,formula1,data=Auto, kernel ="polynomial",
           ranges =list(cost=c(0.01,0.1,1,10,100,1000),d=c(1,2,3,4)))
summary(tuneCd)
set.seed(1)
tuneCg=tune(svm,formula1,data=Auto, kernel ="radial",
            ranges =list(cost=c(0.01,0.1,1,10,100,1000)
                         ,gamma=c(0.5,1,2,3,4)))
summary(tuneCg)

# (d) Plot the results
formula2=weight~year
plot(bestmod,Auto,formula2)
badmodel=svm(formula1,data=Auto,kernel="linear"
                    ,cost=0.01,scale=TRUE)
summary(badmodel)
plot(badmodel,Auto,formula2)
bestpoly=tuneCd$best.model
plot(bestpoly,Auto,formula2)
bestradial=tuneCg$best.model
plot(bestradial,Auto,formula2)
plot(bestradial,Auto,horsepower~weight)

# Compare (training) predictions, i.e. fit to training data
predbest=predict(bestmod,Auto)
table(truth=Auto$high,predict=predbest)
predbad=predict(badmodel,Auto)
table(truth=Auto$high,predict=predbad)
predpoly=predict(bestpoly,Auto)
table(truth=Auto$high,predict=predpoly)
predradial=predict(bestradial,Auto)
table(truth=Auto$high,predict=predradial)
