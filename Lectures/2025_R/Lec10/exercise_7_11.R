
# Generate artificial data
n <- 100
x1 <- rnorm(n)
x2 <- rnorm(n)
eps <- rnorm(n,sd=0.1)
y <- 1 + 2*x1 + 3*x2 + eps

# Fit model with OLS
lm(y~x1+x2)

beta1 <- 0

for(i in 1:5)
{
  a <- y - beta1*x1
  beta2 <- lm(a~x2)$coefficients[2]
  a <- y - beta2*x2
  beta1 <- lm(a~x1)$coefficients[2]
  print(cbind(beta1,beta2))
}
