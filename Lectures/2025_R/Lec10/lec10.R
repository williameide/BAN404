
# (a) Training/test, forward stepwise
set.seed(123)
library(ISLR2)
n <- nrow(College)
ntrain <- floor(n/2)
ind <- sample(1:n,size=ntrain)
train <- College[ind,]
test <- College[-ind,]
library(leaps)
forward <- regsubsets(Outstate~.,data=train,
                      method="forward",nvmax=17)
plot(forward,scale="bic")
# Which variables: Private, Room.Board, Terminal,
# perc.alumni, Expend, Grad.Rate

# (b)
library(gam)
form1 <- Outstate ~ Private + s(Room.Board) + s(Terminal) +
  s(perc.alumni) + s(Expend) + s(Grad.Rate)
gam1 <- gam(form1,data=train)
plot(gam1,terms="s(Room.Board)",se=TRUE,col="red")

myplot <- function(var)
  {
    plot(gam1,terms=var,se=TRUE,col="red")
}

myplot("s(Room.Board)")
myplot("s(Terminal")
myplot("s(perc.alumni)")
myplot("s(Expend)")
myplot("Private")
myplot("s(Grad.Rate)")

# (c)
MSE <- function(pred) mean((test$Outstate-pred)^2)
pred_gam <- predict(gam1,newdata=test)
MSE(pred_gam)
# Compare with a linear model
form2 <- Outstate ~ Private + Room.Board + Terminal +
  perc.alumni + Expend + Grad.Rate
lm1 <- lm(form2,data=train)
pred_lm <- predict(lm1,newdata=test)
MSE(pred_lm)

# (d)
form3 <- Outstate ~ Private + s(Room.Board)  +
  perc.alumni + s(Expend) + Grad.Rate
gam2 <- gam(form3,data=train)
pred_gam2 <- predict(gam2,newdata=test)
MSE(pred_lm)
MSE(pred_gam)
MSE(pred_gam2)












