# Kompendium NEW – Formula & Keyword Matrix

> Source: `Kompendium_NEW.pdf`
> Copyable matrix overview of formulas, key terms, and course structure.

## 1) Mathematical Formula Matrix

| Formula | Meaning | Page(s) |
|---|---|---|
| `+Var(ε)  ` | Expectation/variance expression | 5, 9 |
| `Y=f(X) +ε` | Mathematical expression | 5 |
| `ˆY= ˆf(X)` | Mathematical expression | 5 |
| `ˆyi Fitted/predicted value for observationi:ˆyi = ˆf(xi)` | Mathematical expression | 5 |
| `εRandom error term:E[ε] = 0, independent ofX` | Expectation/variance expression | 5 |
| `MSE= 1` | Mean Squared Error | 9 |
| `i=1` | Mathematical expression | 9, 10, 13, 14, 27, 33, 39, 51, 52, 62 |
| `1(yi̸= ˆyi)` | Mathematical expression | 10, 29 |
| `Bayes error rate= 1−E` | Mathematical expression | 10 |
| `Error rate= 1` | Mathematical expression | 10 |
| `Pr(Y=j\|X)` | Probability/posterior expression | 10 |
| `Pr(Y=j\|X=x 0)` | Probability/posterior expression | 10, 25 |
| `ˆy(x0) = arg max` | Optimization/class decision rule | 10 |
| `1(yi =j)` | Mathematical expression | 11 |
| `K= 1Very jagged, irregular Very low Very high Memorises training` | Mathematical expression | 11 |
| `K=nFlat (predicts majority class` | Mathematical expression | 11 |
| `KNN estimates this probability by:ˆP(Y=j\|X=x 0) = 1` | Mathematical expression | 11 |
| `Pr(Y=j\|X=x 0) = 1` | Probability/posterior expression | 11, 25 |
| `Since we cannot computePr(Y =j\|X=x0)directly, a simple approximation is to estimate it from theK` | Probability/posterior expression | 11 |
| `Var(ε). As model flexibility increases, bias decreases (the model can approximate more complex` | Expectation/variance expression | 11 |
| `i∈N0 1(yi =j)` | Mathematical expression | 11 |
| `j\|X=x0)` | Mathematical expression | 11 |
| `rate (in classification) or Var(ε)(in regression) is doing as well as theoretically possible` | Expectation/variance expression | 11 |
| `- n is larger (more data→more precise estimates). -Var(X)is larger (more spread inX→better-identified` | Expectation/variance expression | 13 |
| `H1 :β1̸= 0. The test statistic is:` | Model parameter expression | 13 |
| `RSS=` | Residual Sum of Squares | 13 |
| `Var(X) , ˆβ0 = ¯y−ˆβ1¯x` | Expectation/variance expression | 13 |
| `Y=β0 +β1X+ε` | Model parameter expression | 13 |
| `i=1(xi−¯x)(yi−¯y)∑n` | Mathematical expression | 13 |
| `i=1(xi−¯x)2` | Mathematical expression | 13 |
| `i=1(xi−¯x)2 = Cov(X,Y)` | Mathematical expression | 13 |
| `i=1(xi−¯x)2,SE( ˆβ0) = ˆσ` | Model parameter expression | 13 |
| `where¯x= 1` | Mathematical expression | 13 |
| `ˆy=ˆβ0 + ˆβ1x` | Model parameter expression | 13 |
| `•They areunbiased:E[ ˆβ0] =β0 andE[ ˆβ1] =β1` | Expectation/variance expression | 13 |
| `∑xi and¯y= 1` | Mathematical expression | 13 |
| `F= (TSS−RSS)/p` | Residual Sum of Squares | 14 |
| `H0 :β1 =β2 =···=βp = 0vs.H 1 :at least oneβj̸= 0` | Model parameter expression | 14 |
| `R2 = 1−RSS` | Residual Sum of Squares | 14 |
| `zero, we rejectH0 :β1 = 0at the 5% level` | Model parameter expression | 14 |
| `Confidence interval (CI)for themean responseatx 0, i.e., forE[Y\|X=x0] =β0 +β1x0:` | Expectation/variance expression | 15 |
| `Y=β0 +β1X1 +β2X2 +···+βpXp +ε` | Model parameter expression | 15 |
| `andβ= (β0,β1,...,βp)⊤` | Model parameter expression | 15 |
| `t-statistic ˆβj/SE( ˆβj)TestsH 0 :βj = 0` | Model parameter expression | 15 |
| `ˆβ= (X⊤X)−1X⊤Y` | Model parameter expression | 15 |
| `Example:X={A,B,C}⇒createDB =1(X=B), D C =1(X=C)` | Mathematical expression | 16 |
| `Y=β0 +β1X+β2X 2 +···+βdXd +ε` | Model parameter expression | 16 |
| `Y=β0 +β1X1 +β2X2 +β3X1X2 +ε` | Model parameter expression | 16 |
| `Detection: Studentised residualsri =e i/ˆσ(−i); values\|ri\|>3are flagged as potential outliers` | Mathematical expression | 17 |
| `Functional form Linear:ˆy= ˆβ0 + ˆβ1x1 +···None — adapts to local data` | Model parameter expression | 18 |
| `ˆf(x 0) = 1` | Mathematical expression | 18 |
| `A:OLS minimises the Residual Sum of Squares:RSS = ∑n` | Residual Sum of Squares | 19 |
| `i=1(yi−ˆβ0−ˆβ1xi)2. The resulting estimators` | Model parameter expression | 19 |
| `p(X) = eβ0+β1X` | Model parameter expression | 20 |
| `withY∈{0,1}, we modelp(X) = Pr(Y= 1\|X)` | Probability/posterior expression | 20 |
| `( Pr(Y=k\|X)` | Probability/posterior expression | 21 |
| `(Y= 0)` | Mathematical expression | 21 |
| `(Y= 1)` | Mathematical expression | 21 |
| `Pr(Y=K\|X)` | Probability/posterior expression | 21 |
| `Pr(Y=k\|X) = eβk0+β⊤` | Probability/posterior expression | 21 |
| `Predicted: Negative (ˆY= 0) Predicted: Positive ( ˆY= 1)` | Mathematical expression | 21 |
| `i:y i=0` | Mathematical expression | 21 |
| `i:y i=1` | Mathematical expression | 21 |
| `l X,Pr(Y=K\|X) = 1` | Probability/posterior expression | 21 |
| `l=1 eβl0+β⊤` | Model parameter expression | 21 |
| `p(X) = eβ0+β1X1+···+βpXp` | Model parameter expression | 21 |
| `to it. For each classk= 1,...,K−1:` | Mathematical expression | 21 |
| `AUC= Pr(ˆp(random positive)>ˆp(random negative))` | Area under ROC interpretation | 22 |
| `FP/(FP+TN) = 1−Specificity; rate of false alarms` | Mathematical expression | 22 |
| `Rate= FP/(FP+TN) = 1−Specificity` | Mathematical expression | 22 |
| `thresholds by plotting: -y-axis:True Positive Rate(Sensitivity) = TP/(TP +FN)- x-axis:False Positive` | Mathematical expression | 22 |
| `t∗= cFP` | Mathematical expression | 22 |
| `An alternative to directly modellingPr(Y =k\|X)(thediscriminativeapproach of logistic regression) is` | Probability/posterior expression | 23 |
| `Pr(Y=k\|X=x) = πk·fk(x)` | Probability/posterior expression | 23 |
| `X\|Y=k∼ N(µk,Σ)` | Mathematical expression | 23 |
| `Y=k)is theclass-conditional density(whatXlooks like within classk)` | Mathematical expression | 23 |
| `fk(x) = 1√` | Mathematical expression | 23 |
| `i:y i=k` | Mathematical expression | 23, 24 |
| `l=1` | Mathematical expression | 23 |
| `n ,ˆµ k = 1` | Mathematical expression | 23 |
| `xi,ˆσ 2 = 1` | Mathematical expression | 23 |
| `ˆπk = nk` | Mathematical expression | 23 |
| `δk(x) =x·µk` | Mathematical expression | 23 |
| `δk(x) =x⊤Σ−1µk−1` | Mathematical expression | 23 |
| `Pr(Y=l\|x) = logπk` | Probability/posterior expression | 24 |
| `X\|Y=k∼ N(µk,Σ k)` | Mathematical expression | 24 |
| `k=1` | Mathematical expression | 24, 37, 55, 63, 64 |
| `log Pr(Y=k\|x)` | Probability/posterior expression | 24 |
| `xi, ˆΣ= 1` | Mathematical expression | 24 |
| `ˆµk = 1` | Mathematical expression | 24 |
| `δk(x) =−1` | Mathematical expression | 24 |
| `1(yi =j),ˆy= arg max` | Optimization/class decision rule | 25 |
| `LDA Linear GaussianX\|Y=k; shared` | Mathematical expression | 25 |
| `Pr(X)` | Probability/posterior expression | 25 |
| `QDA Quadratic GaussianX\|Y=k;` | Mathematical expression | 25 |
| `j=1` | Mathematical expression | 25, 28, 29, 33, 34, 55, 62, 63, 64 |
| `• Logistic regressionisdiscriminative: it directly models Pr(Y = 1\|X)without any assumption on` | Probability/posterior expression | 25 |
| `(logit) scale:log(p/(1−p)) =β0 +β1X` | Model parameter expression | 26 |
| `(b) Predict observationi:ˆyi = ˆf (−i)(xi)` | Mathematical expression | 27 |
| `(c) Compute the squared error: MSEi = (yi−ˆyi)2` | Mean Squared Error | 27 |
| `1. For eachi= 1,2,...,n:` | Mathematical expression | 27 |
| `CV(n) = 1` | Mathematical expression | 27 |
| `Validation MSE= 1` | Mean Squared Error | 27 |
| `(k=n)` | Mathematical expression | 28 |
| `2. Forj= 1,2,...,k:` | Mathematical expression | 28 |
| `CV(k) = 1` | Mathematical expression | 28, 29 |
| `LOOCV is the special case wherek=n` | Mathematical expression | 28 |
| `equal-sizedfolds(groups). For each foldj= 1,...,k:` | Mathematical expression | 28 |
| `k= 10` | Mathematical expression | 28 |
| `k= 5` | Mathematical expression | 28 |
| `,where ¯ˆθ∗= 1` | Mathematical expression | 29 |
| `1. ChooseB(typicallyB= 1000or more)` | Mathematical expression | 29 |
| `2. Forb= 1,2,...,B:` | Mathematical expression | 29, 47 |
| `Errj,where Err j = 1` | Mathematical expression | 29 |
| `Pr(obsi /∈Z∗b) =` | Probability/posterior expression | 29 |
| `b=1` | Mathematical expression | 29, 45, 46 |
| `1. DrawB= 1000bootstrap samples from the(X,Y)data` | Mathematical expression | 30 |
| `TypicalBork k= 5or10B= 1000+` | Mathematical expression | 30 |
| `k= 5ork= 10typically recommended over LOOCV?` | Mathematical expression | 30 |
| `ˆα= ˆσ2` | Mathematical expression | 30 |
| `analytical formulaSE( ˆβj) = ˆσ` | Model parameter expression | 31 |
| `2. Fork= 0,1,...,p−1:` | Mathematical expression | 32 |
| `2. Fork=p,p−1,...,1:` | Mathematical expression | 32 |
| `j =RSS+λ∥β∥2` | Residual Sum of Squares | 33 |
| `•λ= 0: no penalty⇒OLS solution` | Model parameter expression | 33 |
| `•λ→∞: allˆβj→0(exceptˆβ0 = ¯y)` | Model parameter expression | 33 |
| `\|βj\|=RSS+λ∥β∥1` | Residual Sum of Squares | 34 |
| `ˆβ0 = ¯yafter centering` | Model parameter expression | 34 |
| `ˆβRidge = (X⊤X+λI)−1X⊤y` | Model parameter expression | 34 |
| `˜xij = xij−¯xj` | Mathematical expression | 34 |
| `ChoosingM: via cross-validation.M=precovers OLS;M= 1gives the most reduced model` | Mathematical expression | 35 |
| `EachZm = ∑p` | Mathematical expression | 35 |
| `reduces RSS. Therefore, RSS is non-increasing as predictors are added andR2 = 1−RSS/TSSis` | Residual Sum of Squares | 36 |
| `C0(X) =1(X <c 1), C 1(X) =1(c 1≤X <c2), ..., C K(X) =1(X≥cK)` | Mathematical expression | 37 |
| `yi =β0 +` | Model parameter expression | 37 |
| `yi =β0 +β1C1(xi) +β2C2(xi) +···+βKCK(xi) +εi` | Model parameter expression | 37 |
| `yi =β0 +β1xi +β2x2` | Model parameter expression | 37 |
| `•b k(x) =x k: polynomial regression of degreeK` | Mathematical expression | 37 |
| `Degrees of freedom for natural spline=K` | Mathematical expression | 38 |
| `Degrees of freedom=K+ 4` | Mathematical expression | 38 |
| `Net degrees of freedom:(4K+ 4)−3K=K+ 4` | Mathematical expression | 38 |
| `Start withK+ 1separate cubic polynomials:(K+ 1)×4 = 4K+ 4parameters` | Mathematical expression | 38 |
| `•b k(x) = (x−ξk)3` | Mathematical expression | 38 |
| `•b k(x) =1(c k−1≤x<ck): step function` | Mathematical expression | 38 |
| `In Python’spatsy, natural splines are created withcr(x, df=K)(cubic regression splines)` | Mathematical expression | 39 |
| `dfλ=tr(S λ)` | Model parameter expression | 39 |
| `(a) Compute thepartial residuals: ri =yi−ˆβ0−∑` | Model parameter expression | 40 |
| `1. Initialise: ˆfj≡0for allj;ˆβ0 = ¯y` | Model parameter expression | 40 |
| `2. Assign observation-specific weightswi =K(xi,x 0)based on distance: observations close tox0 receive` | Mathematical expression | 40 |
| `2. Cycle throughj= 1,2,...,p(repeat until convergence):` | Mathematical expression | 40 |
| `Y=β0 +f 1(X1) +f 2(X2) +···+fp(Xp) +ε` | Model parameter expression | 40 |
| `k̸=j` | Mathematical expression | 40 |
| `outliers with appropriate kernel. Best forp= 1orp= 2` | Mathematical expression | 41 |
| `[g′′(t)]2dt. Whenλ= 0there is no penalty` | Model parameter expression | 42 |
| `fj(Xj), givingY =β0 + ∑` | Model parameter expression | 42 |
| `λ= 0) down to 2 (straight line,λ→∞). In practice,λis chosen by LOOCV` | Model parameter expression | 42 |
| `R1(j,s) ={x\|xj <s}andR 2(j,s) ={x\|xj≥s}` | Mathematical expression | 43 |
| `m=1` | Mathematical expression | 43, 44 |
| `ˆyRm = 1` | Mathematical expression | 43 |
| `(oneˆpmk = 1);preferred for` | Mathematical expression | 44 |
| `Cross-entropy (deviance)D=− ∑K` | Entropy/deviance expression | 44 |
| `G=p(1−p) + (1−p)·p= 2p(1−p)` | Mathematical expression | 44 |
| `Gini index in depth: For a binary outcome (K= 2) withˆpm1 =p:` | Gini impurity expression | 44 |
| `Gini indexG= ∑K` | Gini impurity expression | 44 |
| `k=1 ˆpmk log(ˆpmk)Numerically similar to Gini; also` | Gini impurity expression | 44 |
| `k=1 ˆpmk(1−ˆpmk)Measures total variance across` | Mathematical expression | 44 |
| `i = 1` | Mathematical expression | 45 |
| `ˆfbag(x) = 1` | Mathematical expression | 45 |
| `Default:m= √p(classification), m=p/3(regression)` | Mathematical expression | 46 |
| `Importance(Xj) = 1` | Mathematical expression | 46 |
| `1. Initialise: ˆf(x) = 0and residualsr i =y i for alli` | Mathematical expression | 47 |
| `3. Output: ˆf(x) = ∑B` | Mathematical expression | 47 |
| `b=1νˆfb(x)` | Mathematical expression | 47 |
| `depth.d= 1` | Mathematical expression | 47 |
| `model;d= 2` | Mathematical expression | 47 |
| `(d= 1–6)` | Mathematical expression | 48 |
| `equally distributed. The Gini index isG = ∑` | Gini impurity expression | 48 |
| `Msubject toy i(β0 +β⊤xi)≥M∀i,∥β∥= 1` | Model parameter expression | 50 |
| `classifier based on the hyperplane predictsˆyi =sign(β0 +β⊤xi)` | Model parameter expression | 50 |
| `β0 +β1X1 +β2X2 +···+βpXp = 0` | Model parameter expression | 50 |
| `•Inp= 2dimensions, a hyperplane is aline` | Mathematical expression | 50 |
| `•Inp= 3dimensions, a hyperplane is aplane` | Mathematical expression | 50 |
| `ξi≤C,∥β∥= 1` | Model parameter expression | 51 |
| `f(x) =β0 +` | Model parameter expression | 52 |
| `(decision_function_shape='ovr')` | Mathematical expression | 53 |
| `Example: withK = 4classes, we train` | Mathematical expression | 53 |
| `is expensive. The polynomial kernelK(xi, xi′) = (1 +x⊤` | Mathematical expression | 53 |
| `ˆy= arg max` | Optimization/class decision rule | 53 |
| `mean whenξi = 0,0<ξi≤1, andξi >1?` | Mathematical expression | 54 |
| `, l= 1,...,L` | Mathematical expression | 55 |
| `Ak =h` | Mathematical expression | 55 |
| `h(l) =h` | Mathematical expression | 55 |
| `ˆY=g` | Mathematical expression | 55 |
| `ˆY=g 0 +` | Mathematical expression | 55 |
| `, k= 1,...,K(hidden layer)` | Mathematical expression | 55 |
| `1. The softmax functionh(zk) =ezk/ ∑` | Mathematical expression | 56 |
| `Linearh(z) =z(−∞,∞)Output layer for regression` | Mathematical expression | 56 |
| `ReLUh(z) = max(0,z) [0,∞)Hidden layers — default choice;` | Mathematical expression | 56 |
| `Sigmoidh(z) = 1` | Mathematical expression | 56 |
| `Softmaxh(z k) = ezk` | Mathematical expression | 56 |
| `Tanhh(z) = ez−e−z` | Mathematical expression | 56 |
| `k=1yik log ˆpik` | Mathematical expression | 56 |
| `l=1ezl` | Mathematical expression | 56 |
| `(b) Compute∂L/∂W(l) =∂L/∂z(l)·(h(l−1))⊤` | Mathematical expression | 57 |
| `2. For each layerl=L,L−1,...,1(going backward):` | Mathematical expression | 57 |
| `•One epoch =⌈n/Bmb⌉iterations` | Mathematical expression | 57 |
| `Lreg =L+ λ` | Model parameter expression | 58 |
| `PVEm = d2` | Mathematical expression | 62 |
| `Thefirst principal component loading vectorϕ1 = (ϕ11,ϕ21,...,ϕp1)⊤solves:` | Mathematical expression | 62 |
| `i=1x2` | Mathematical expression | 62 |
| `i=1z2` | Mathematical expression | 62 |
| `j=1 Var(Xj). After mean-centring, this equals ∑p` | Expectation/variance expression | 62 |
| `l=1d2` | Mathematical expression | 62 |
| `m=1d2` | Mathematical expression | 62 |
| `zim =ϕ⊤` | Mathematical expression | 62 |
| `ϕ1 = arg max` | Optimization/class decision rule | 62 |
| `ϕ:∥ϕ∥=1` | Mathematical expression | 62 |
| `ϕj1xij =ϕ⊤` | Mathematical expression | 62 |
| ` = arg max` | Optimization/class decision rule | 62 |
| `2. Compute the SVD of˜X=UDV ⊤` | Mathematical expression | 63 |
| `3. Loading vectors: columns ofV(i.e.ϕm =v m, them-th column)` | Mathematical expression | 63 |
| `4. Score matrix:Z= ˜XV(equivalently,z im = ˜x⊤` | Mathematical expression | 63 |
| `WCV(Ck) = 1` | Mathematical expression | 63 |
| `C(i) = arg mink∥xi−¯xk∥2` | Optimization/class decision rule | 64 |
| `s(i) = b(i)−a(i)` | Mathematical expression | 64 |
| `¯xk = 1` | Mathematical expression | 64 |
| `Completed(A,B) = max i∈A,j∈Bd(xi,xj)—` | Mathematical expression | 65 |
| `Singled(A,B) = min i∈A,j∈Bd(xi,xj)—` | Mathematical expression | 65 |
| `arg minA̸=Bd(A,B)` | Optimization/class decision rule | 65 |
| `dij =d(x i,xj)` | Mathematical expression | 65 |
| `• Euclidean distance:d(xi, xj) =∥xi−xj∥2 =` | Mathematical expression | 66 |
| `•Manhattan distance:d(x i,xj) = ∑` | Mathematical expression | 66 |
| `n (mean-centred) observations onto this direction:zi1 =ϕ⊤` | Mathematical expression | 67 |

## 2) Keyword Matrix (all TOC terms + locations)

| Module | Section | Keyword / Term | Page | Related concepts |
|---|---|---|---|---|
| Foundations | 1 | Chapter 1 – Introduction to Statistical Learning | 5 |  |
| Foundations | 1.1 | What is Statistical Learning? | 5 |  |
| Foundations | 1.2 | Notation | 5 |  |
| Foundations | 1.3 | Why Do We Estimatef? | 5 |  |
| Foundations | 1.3.1 | Prediction | 5 |  |
| Foundations | 1.3.2 | Inference | 6 |  |
| Foundations | 1.4 | Parametric vs. Non-Parametric Methods | 6 |  |
| Foundations | 1.4.1 | Parametric Methods | 6 |  |
| Foundations | 1.4.2 | Non-Parametric Methods | 6 |  |
| Foundations | 1.5 | Supervised vs. Unsupervised Learning | 7 |  |
| Model Assessment | 2 | Chapter 2 – Assessing Model Accuracy | 9 |  |
| Model Assessment | 2.1 | Measuring the Quality of Fit | 9 |  |
| Model Assessment | 2.1.1 | The Pattern of Training and Test MSE as Flexibility Increases | 9 |  |
| Model Assessment | 2.2 | The Bias-Variance Trade-off | 9 |  |
| Model Assessment | 2.3 | The Classification Setting | 10 |  |
| Model Assessment | 2.3.1 | The Bayes Classifier | 10 |  |
| Model Assessment | 2.3.2 | K-Nearest Neighbours Classifier | 11 |  |
| Model Assessment | 2.4 | Summary: Key Principles of Model Assessment | 11 |  |
| Linear Regression | 3 | Chapter 3 – Linear Regression | 13 |  |
| Linear Regression | 3.1 | Simple Linear Regression | 13 |  |
| Linear Regression | 3.1.1 | Estimating Coefficients — Ordinary Least Squares (OLS) | 13 |  |
| Linear Regression | 3.1.2 | Assessing the Accuracy of Coefficient Estimates | 13 |  |
| Linear Regression | 3.1.3 | Assessing Model Accuracy: RSE,R 2, and theF-statistic | 14 |  |
| Linear Regression | 3.1.4 | Confidence Intervals vs. Prediction Intervals | 15 |  |
| Linear Regression | 3.2 | Multiple Linear Regression | 15 |  |
| Linear Regression | 3.2.1 | Interpretation of Coefficients in Multiple Regression | 16 |  |
| Linear Regression | 3.2.2 | Key Questions in Multiple Regression | 16 |  |
| Linear Regression | 3.3 | Extensions of the Linear Model | 16 |  |
| Linear Regression | 3.3.1 | Interaction Terms | 16 |  |
| Linear Regression | 3.3.2 | Polynomial Regression | 16 |  |
| Linear Regression | 3.3.3 | Qualitative (Categorical) Predictors | 16 |  |
| Linear Regression | 3.4 | Potential Problems in Linear Regression | 17 |  |
| Linear Regression | 3.4.1 | 1. Non-linearity of the Relationship | 17 |  |
| Linear Regression | 3.4.2 | 2. Heteroscedasticity (Non-constant Error Variance) | 17 |  |
| Linear Regression | 3.4.3 | 3. Correlated Errors (Autocorrelation) | 17 |  |
| Linear Regression | 3.4.4 | 4. Outliers | 17 |  |
| Linear Regression | 3.4.5 | 5. High-Leverage Points | 17 |  |
| Linear Regression | 3.4.6 | 6. Multicollinearity | 18 |  |
| Linear Regression | 3.5 | KNN Regression vs. Linear Regression | 18 | distance-based prediction |
| Classification | 4 | Chapter 4 – Classification | 20 |  |
| Classification | 4.1 | Why Not Linear Regression for Classification? | 20 |  |
| Classification | 4.2 | Logistic Regression | 20 | log-odds link |
| Classification | 4.2.1 | The Logistic Model | 20 | log-odds link |
| Classification | 4.2.2 | Estimating Coefficients — Maximum Likelihood | 21 |  |
| Classification | 4.2.3 | Multiple Logistic Regression | 21 | log-odds link |
| Classification | 4.2.4 | Multinomial Logistic Regression (K >2Classes) | 21 | log-odds link |
| Classification | 4.3 | Evaluating Classifiers | 21 |  |
| Classification | 4.3.1 | The Confusion Matrix | 21 |  |
| Classification | 4.3.2 | The ROC Curve and AUC | 22 |  |
| Classification | 4.3.3 | Cost-Optimal Classification Threshold | 22 |  |
| Classification | 4.4 | Generative Models for Classification | 23 |  |
| Classification | 4.5 | Linear Discriminant Analysis (LDA) | 23 | generative Gaussian classifier |
| Classification | 4.5.1 | LDA for p= 1(Single Predictor) | 23 | generative Gaussian classifier |
| Classification | 4.5.2 | LDA for p>1(Multiple Predictors) | 23 | generative Gaussian classifier |
| Classification | 4.6 | Quadratic Discriminant Analysis (QDA) | 24 | quadratic boundary |
| Classification | 4.6.1 | LDA vs. QDA: When to Use Which | 24 | generative Gaussian classifier |
| Classification | 4.7 | Naive Bayes | 24 |  |
| Classification | 4.8 | K-Nearest Neighbours Classifier (Revisited) | 25 |  |
| Classification | 4.9 | Comparison of Classification Methods | 25 |  |
| Resampling | 5 | Chapter 5 – Resampling Methods | 27 |  |
| Resampling | 5.1 | Cross-Validation | 27 | model selection; error estimation |
| Resampling | 5.1.1 | The Validation Set Approach | 27 |  |
| Resampling | 5.1.2 | Leave-One-Out Cross-Validation (LOOCV) | 27 | model selection; error estimation |
| Resampling | 5.1.4 | Bias-Variance Trade-off for k-Fold CV | 28 |  |
| Resampling | 5.1.5 | CV for Model Selection | 28 |  |
| Resampling | 5.1.6 | Cross-Validation for Classification | 29 | model selection; error estimation |
| Resampling | 5.2 | The Bootstrap | 29 | sampling with replacement |
| Resampling | 5.2.1 | Core Idea and Algorithm | 29 |  |
| Resampling | 5.2.2 | What Observations End Up in a Bootstrap Sample? | 29 | sampling with replacement |
| Resampling | 5.2.3 | The Portfolio Example: When Bootstrap is Essential | 30 | sampling with replacement |
| Resampling | 5.3 | Cross-Validation vs. Bootstrap: A Critical Distinction | 30 | model selection; error estimation |
| Model Selection & Regularization | 6 | Chapter 6 – Linear Model Selection and Regularization | 32 |  |
| Model Selection & Regularization | 6.1 | Subset Selection | 32 |  |
| Model Selection & Regularization | 6.1.1 | Best Subset Selection | 32 |  |
| Model Selection & Regularization | 6.1.2 | Forward Stepwise Selection | 32 |  |
| Model Selection & Regularization | 6.1.3 | Backward Stepwise Selection | 32 |  |
| Model Selection & Regularization | 6.2 | Choosing the Optimal Model Size | 32 |  |
| Model Selection & Regularization | 6.2.1 | Information Criteria | 33 |  |
| Model Selection & Regularization | 6.2.2 | Cross-Validation for Model Selection | 33 | model selection; error estimation |
| Model Selection & Regularization | 6.3 | Shrinkage Methods: Ridge and Lasso | 33 | L2 shrinkage |
| Model Selection & Regularization | 6.3.1 | Ridge Regression (L2 Regularisation) | 33 | L2 shrinkage |
| Model Selection & Regularization | 6.3.2 | The Lasso (L1 Regularisation) | 34 | L1 shrinkage |
| Model Selection & Regularization | 6.3.3 | Ridge vs. Lasso: A Detailed Comparison | 34 | L2 shrinkage |
| Model Selection & Regularization | 6.3.4 | Selecting λby Cross-Validation | 35 | model selection; error estimation |
| Model Selection & Regularization | 6.4 | Dimension Reduction: Principal Components Regression (PCR) | 35 |  |
| Non-Linear Methods | 7 | Chapter 7 – Moving Beyond Linearity | 37 |  |
| Non-Linear Methods | 7.1 | Polynomial Regression | 37 |  |
| Non-Linear Methods | 7.2 | Step Functions | 37 |  |
| Non-Linear Methods | 7.3 | Basis Functions: A Unifying Framework | 37 |  |
| Non-Linear Methods | 7.4 | Regression Splines | 38 |  |
| Non-Linear Methods | 7.4.1 | Piecewise Polynomials | 38 |  |
| Non-Linear Methods | 7.4.2 | Cubic Splines | 38 |  |
| Non-Linear Methods | 7.4.3 | Natural Splines | 38 |  |
| Non-Linear Methods | 7.4.4 | Choosing the Number and Location of Knots | 39 |  |
| Non-Linear Methods | 7.5 | Smoothing Splines | 39 |  |
| Non-Linear Methods | 7.6 | Local Regression | 40 |  |
| Non-Linear Methods | 7.7 | Generalised Additive Models (GAMs) | 40 | additive smooth terms |
| Non-Linear Methods | 7.7.1 | Fitting GAMs: Backfitting | 40 | additive smooth terms |
| Non-Linear Methods | 7.7.2 | GAMs for Classification (Logistic GAMs) | 41 | log-odds link |
| Non-Linear Methods | 7.7.3 | Advantages and Disadvantages of GAMs | 41 | additive smooth terms |
| Tree-Based Methods | 8 | Chapter 8 – Tree-Based Methods | 43 |  |
| Tree-Based Methods | 8.1 | Regression Trees | 43 |  |
| Tree-Based Methods | 8.1.1 | Building a Regression Tree: Recursive Binary Splitting | 43 |  |
| Tree-Based Methods | 8.1.2 | Tree Pruning: Cost-Complexity Pruning | 43 |  |
| Tree-Based Methods | 8.2 | Classification Trees | 44 |  |
| Tree-Based Methods | 8.3 | Advantages and Disadvantages of Trees | 45 |  |
| Tree-Based Methods | 8.4 | Bagging (Bootstrap Aggregating) | 45 | sampling with replacement |
| Tree-Based Methods | 8.4.1 | Out-of-Bag (OOB) Error | 45 |  |
| Tree-Based Methods | 8.4.2 | Variable Importance from Bagged Trees | 46 |  |
| Tree-Based Methods | 8.5 | Random Forests | 46 | bagging + feature subsampling |
| Tree-Based Methods | 8.6 | Gradient Boosting | 47 | sequential bias reduction |
| Tree-Based Methods | 8.6.1 | The Boosting Idea | 47 | sequential bias reduction |
| Tree-Based Methods | 8.6.2 | The Three Hyperparameters of Boosting | 47 | sequential bias reduction |
| Tree-Based Methods | 8.6.3 | Why Does Boosting Work? The Bias Perspective | 48 | sequential bias reduction |
| Tree-Based Methods | 8.7 | Summary of Ensemble Methods | 48 |  |
| Support Vector Machines | 9 | Chapter 9 – Support Vector Machines | 50 |  |
| Support Vector Machines | 9.1 | What is a Hyperplane? | 50 |  |
| Support Vector Machines | 9.2 | The Maximal Margin Classifier (Hard Margin SVM) | 50 | margin + kernels |
| Support Vector Machines | 9.2.1 | Support Vectors | 50 |  |
| Support Vector Machines | 9.2.2 | Limitations of the Maximal Margin Classifier | 51 |  |
| Support Vector Machines | 9.3 | The Support Vector Classifier (Soft Margin SVM) | 51 | margin + kernels |
| Support Vector Machines | 9.3.1 | Interpreting the Slack Variables | 51 |  |
| Support Vector Machines | 9.3.2 | The Role ofC(Budget Parameter) | 51 |  |
| Support Vector Machines | 9.4 | The Support Vector Machine (Kernel Trick) | 51 |  |
| Support Vector Machines | 9.4.1 | The Dual Formulation and Inner Products | 52 |  |
| Support Vector Machines | 9.4.2 | Common Kernel Functions | 52 |  |
| Support Vector Machines | 9.4.3 | Choosing C and γby Cross-Validation | 52 | model selection; error estimation |
| Support Vector Machines | 9.5 | Multi-Class SVM | 53 | margin + kernels |
| Support Vector Machines | 9.5.1 | One-vs-One (OvO) | 53 |  |
| Support Vector Machines | 9.5.2 | One-vs-All (OvA) | 53 |  |
| Support Vector Machines | 9.6 | SVM vs. Other Classifiers | 53 | margin + kernels |
| Neural Networks / Deep Learning | 10 | Chapter 10 – Deep Learning and Neural Networks | 55 |  |
| Neural Networks / Deep Learning | 10.1 | From Linear to Non-Linear: The Need for Activation Functions | 55 |  |
| Neural Networks / Deep Learning | 10.2 | Architecture of a Multilayer Perceptron (MLP) | 55 |  |
| Neural Networks / Deep Learning | 10.3 | Activation Functions | 56 |  |
| Neural Networks / Deep Learning | 10.3.1 | Why ReLU is Preferred in Hidden Layers | 56 |  |
| Neural Networks / Deep Learning | 10.4 | Loss Functions | 56 |  |
| Neural Networks / Deep Learning | 10.5 | Training: Backpropagation and Gradient Descent | 57 |  |
| Neural Networks / Deep Learning | 10.5.1 | The Gradient Descent Update Rule | 57 |  |
| Neural Networks / Deep Learning | 10.5.2 | Backpropagation | 57 |  |
| Neural Networks / Deep Learning | 10.5.3 | Mini-Batch Stochastic Gradient Descent | 57 |  |
| Neural Networks / Deep Learning | 10.5.4 | Adaptive Optimisers | 57 |  |
| Neural Networks / Deep Learning | 10.6 | Hyperparameters of Neural Networks | 58 |  |
| Neural Networks / Deep Learning | 10.7 | Regularisation of Neural Networks | 58 |  |
| Neural Networks / Deep Learning | 10.7.1 | L2 Regularisation (Weight Decay) | 58 |  |
| Neural Networks / Deep Learning | 10.7.2 | Dropout | 58 | neural-net regularization |
| Neural Networks / Deep Learning | 10.7.3 | Early Stopping | 59 |  |
| Neural Networks / Deep Learning | 10.8 | The Bias-Variance Trade-off in Neural Networks | 59 |  |
| Neural Networks / Deep Learning | 10.9 | Neural Networks vs. Other Methods | 60 |  |
| Unsupervised Learning | 11 | Chapter 11 – Unsupervised Learning | 62 |  |
| Unsupervised Learning | 11.1 | Principal Component Analysis | 62 |  |
| Unsupervised Learning | 11.1.1 | Motivation and Intuition | 62 |  |
| Unsupervised Learning | 11.1.2 | Loading Vectors and Scores | 62 |  |
| Unsupervised Learning | 11.1.3 | Proportion of Variance Explained | 62 |  |
| Unsupervised Learning | 11.1.4 | Choosing the Number of Components | 63 |  |
| Unsupervised Learning | 11.1.5 | The Biplot | 63 |  |
| Unsupervised Learning | 11.1.6 | PCA for Regression: Principal Components Regression | 63 | dimension reduction |

## 3) Requested Terms Quick-Check

| Term | Page(s) found |
|---|---|
| sampling | 2, 27, 29, 30, 31, 48 |
| cross validation | Not found as exact token |
| cross-validation | 2, 3, 9, 11, 12, 16, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 40, 48, 49, 51, 52, 63 |
| bagging | 3, 30, 43, 45, 46, 47, 48, 49, 60 |
| knn | 1, 10, 11, 18, 19, 25, 40, 63 |
| bootstrapping | Not found as exact token |
