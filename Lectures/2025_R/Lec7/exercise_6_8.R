
# (a) Generate x-variable and error-term
n=100
x=rnorm(n) # 100 numbers drawn from a N(0,1)-variable
eps=rnorm(n)

# (b) Generate y = 1 + x - 0.5x^2 + 0.5x^3 + eps
b0=1
b1=1
b2=-0.5
b3=0.5
y=b0+b1*x+b2*x^2+b3*x^3+eps
plot(x,y) # Plot y vs x
lm(y~x+I(x^2)+I(x^3)) # Fit the model y=b0+b1x+b2x^2+b3x^2+eps

# (c) Best subset regression
library(leaps)
# Create dataset with y and powers of x
XY=data.frame(matrix(0,n,11)) # Initiate a 100 x 11 data frame
for(j in 1:10)
{
  XY[,j]=x^j
  colnames(XY)[j]=paste("x",j,sep="") # Name the variable x^j
                                      # "xj"
}
XY[,11]=y
colnames(XY)[11]="y"
summary(lm(y~.,data=XY)) # Run the full regression
bestsubset=regsubsets(y~.,data=XY,method="exhaustive",nvmax = 8)
plot(bestsubset,scale="adjr2") # Visualize different model
plot(bestsubset,scale="bic")   # against measures of fit
plot(bestsubset,scale="Cp")

bsum=summary(bestsubset)
names(bsum)
bsum$bic # The BIC for the best model of each size 0,1,...,10
which.min(bsum$bic) # Find the model with smallest BIC
coef(bestsubset,3)

# (d) Forward/backward stepwise
forward=regsubsets(y~.,data=XY,method="backward",nvmax = 8)
plot(forward,scale="Cp")
plot(forward,scale="adjr2")
plot(forward,scale="bic")
fsum=summary(forward)
fsum$bic
which.min(fsum$bic)
coef(forward,3)


