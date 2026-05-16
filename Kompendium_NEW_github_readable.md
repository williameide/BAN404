# Kompendium NEW (GitHub Readable)

> Source: `Kompendium_NEW.pdf`
> Total pages: 67

This markdown is a page-by-page text conversion for GitHub reading. Mathematical symbols are preserved from the PDF text layer where available.

## Page 1

BAN404 – Statistical Machine Learning and Data Analysis
Exam Compendium | Based on ISLP
Exam: May 19, 2026
Contents
1 Chapter 1 – Introduction to Statistical Learning 5
1.1 What is Statistical Learning? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
1.2 Notation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
1.3 Why Do We Estimatef? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
1.3.1 Prediction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
1.3.2 Inference . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
1.4 Parametric vs. Non-Parametric Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
1.4.1 Parametric Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
1.4.2 Non-Parametric Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
1.5 Supervised vs. Unsupervised Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
2 Chapter 2 – Assessing Model Accuracy 9
2.1 Measuring the Quality of Fit . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
2.1.1 The Pattern of Training and Test MSE as Flexibility Increases . . . . . . . . . . . . . . 9
2.2 The Bias-Variance Trade-off . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
2.3 The Classification Setting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
2.3.1 The Bayes Classifier . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
2.3.2 K-Nearest Neighbours Classifier . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
2.4 Summary: Key Principles of Model Assessment . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
3 Chapter 3 – Linear Regression 13
3.1 Simple Linear Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
3.1.1 Estimating Coefficients — Ordinary Least Squares (OLS) . . . . . . . . . . . . . . . . . 13
3.1.2 Assessing the Accuracy of Coefficient Estimates . . . . . . . . . . . . . . . . . . . . . . . 13
3.1.3 Assessing Model Accuracy: RSE,R 2, and theF-statistic . . . . . . . . . . . . . . . . . . 14
3.1.4 Confidence Intervals vs. Prediction Intervals . . . . . . . . . . . . . . . . . . . . . . . . . 15
3.2 Multiple Linear Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
3.2.1 Interpretation of Coefficients in Multiple Regression . . . . . . . . . . . . . . . . . . . . 16
3.2.2 Key Questions in Multiple Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
3.3 Extensions of the Linear Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
3.3.1 Interaction Terms . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
3.3.2 Polynomial Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
3.3.3 Qualitative (Categorical) Predictors . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
3.4 Potential Problems in Linear Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
3.4.1 1. Non-linearity of the Relationship . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
3.4.2 2. Heteroscedasticity (Non-constant Error Variance) . . . . . . . . . . . . . . . . . . . . 17
3.4.3 3. Correlated Errors (Autocorrelation) . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
3.4.4 4. Outliers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
3.4.5 5. High-Leverage Points . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
3.4.6 6. Multicollinearity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
3.5 KNN Regression vs. Linear Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
4 Chapter 4 – Classification 20
4.1 Why Not Linear Regression for Classification? . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
4.2 Logistic Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
4.2.1 The Logistic Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
4.2.2 Estimating Coefficients — Maximum Likelihood . . . . . . . . . . . . . . . . . . . . . . 21
1

## Page 2

4.2.3 Multiple Logistic Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
4.2.4 Multinomial Logistic Regression (K >2Classes) . . . . . . . . . . . . . . . . . . . . . . 21
4.3 Evaluating Classifiers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
4.3.1 The Confusion Matrix . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
4.3.2 The ROC Curve and AUC . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
4.3.3 Cost-Optimal Classification Threshold . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
4.4 Generative Models for Classification . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
4.5 Linear Discriminant Analysis (LDA) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
4.5.1 LDA forp= 1(Single Predictor) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
4.5.2 LDA forp>1(Multiple Predictors) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
4.6 Quadratic Discriminant Analysis (QDA) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
4.6.1 LDA vs. QDA: When to Use Which . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
4.7 Naive Bayes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
4.8 K-Nearest Neighbours Classifier (Revisited) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
4.9 Comparison of Classification Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
5 Chapter 5 – Resampling Methods 27
5.1 Cross-Validation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
5.1.1 The Validation Set Approach . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
5.1.2 Leave-One-Out Cross-Validation (LOOCV) . . . . . . . . . . . . . . . . . . . . . . . . . 27
5.1.3k-Fold Cross-Validation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
5.1.4 Bias-Variance Trade-off fork-Fold CV . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
5.1.5 CV for Model Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
5.1.6 Cross-Validation for Classification . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
5.2 The Bootstrap . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
5.2.1 Core Idea and Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
5.2.2 What Observations End Up in a Bootstrap Sample? . . . . . . . . . . . . . . . . . . . . 29
5.2.3 The Portfolio Example: When Bootstrap is Essential . . . . . . . . . . . . . . . . . . . . 30
5.3 Cross-Validation vs. Bootstrap: A Critical Distinction . . . . . . . . . . . . . . . . . . . . . . . 30
6 Chapter 6 – Linear Model Selection and Regularization 32
6.1 Subset Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
6.1.1 Best Subset Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
6.1.2 Forward Stepwise Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
6.1.3 Backward Stepwise Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
6.2 Choosing the Optimal Model Size . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
6.2.1 Information Criteria . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
6.2.2 Cross-Validation for Model Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
6.3 Shrinkage Methods: Ridge and Lasso . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
6.3.1 Ridge Regression (L2 Regularisation) . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
6.3.2 The Lasso (L1 Regularisation) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
6.3.3 Ridge vs. Lasso: A Detailed Comparison . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
6.3.4 Selectingλby Cross-Validation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
6.4 Dimension Reduction: Principal Components Regression (PCR) . . . . . . . . . . . . . . . . . . 35
7 Chapter 7 – Moving Beyond Linearity 37
7.1 Polynomial Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
7.2 Step Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
7.3 Basis Functions: A Unifying Framework . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
7.4 Regression Splines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
7.4.1 Piecewise Polynomials . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
7.4.2 Cubic Splines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
7.4.3 Natural Splines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
7.4.4 Choosing the Number and Location of Knots . . . . . . . . . . . . . . . . . . . . . . . . 39
7.5 Smoothing Splines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
7.6 Local Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
7.7 Generalised Additive Models (GAMs) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
7.7.1 Fitting GAMs: Backfitting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
7.7.2 GAMs for Classification (Logistic GAMs) . . . . . . . . . . . . . . . . . . . . . . . . . . 41
7.7.3 Advantages and Disadvantages of GAMs . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
2

## Page 3

8 Chapter 8 – Tree-Based Methods 43
8.1 Regression Trees . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
8.1.1 Building a Regression Tree: Recursive Binary Splitting . . . . . . . . . . . . . . . . . . . 43
8.1.2 Tree Pruning: Cost-Complexity Pruning . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
8.2 Classification Trees . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
8.3 Advantages and Disadvantages of Trees . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
8.4 Bagging (Bootstrap Aggregating) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
8.4.1 Out-of-Bag (OOB) Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
8.4.2 Variable Importance from Bagged Trees . . . . . . . . . . . . . . . . . . . . . . . . . . . 46
8.5 Random Forests . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46
8.6 Gradient Boosting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
8.6.1 The Boosting Idea . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
8.6.2 The Three Hyperparameters of Boosting . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
8.6.3 Why Does Boosting Work? The Bias Perspective . . . . . . . . . . . . . . . . . . . . . . 48
8.7 Summary of Ensemble Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
9 Chapter 9 – Support Vector Machines 50
9.1 What is a Hyperplane? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
9.2 The Maximal Margin Classifier (Hard Margin SVM) . . . . . . . . . . . . . . . . . . . . . . . . 50
9.2.1 Support Vectors . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
9.2.2 Limitations of the Maximal Margin Classifier . . . . . . . . . . . . . . . . . . . . . . . . 51
9.3 The Support Vector Classifier (Soft Margin SVM) . . . . . . . . . . . . . . . . . . . . . . . . . 51
9.3.1 Interpreting the Slack Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
9.3.2 The Role ofC(Budget Parameter) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
9.4 The Support Vector Machine (Kernel Trick) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
9.4.1 The Dual Formulation and Inner Products . . . . . . . . . . . . . . . . . . . . . . . . . . 52
9.4.2 Common Kernel Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 52
9.4.3 ChoosingCandγby Cross-Validation . . . . . . . . . . . . . . . . . . . . . . . . . . . . 52
9.5 Multi-Class SVM . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
9.5.1 One-vs-One (OvO) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
9.5.2 One-vs-All (OvA) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
9.6 SVM vs. Other Classifiers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
10 Chapter 10 – Deep Learning and Neural Networks 55
10.1 From Linear to Non-Linear: The Need for Activation Functions . . . . . . . . . . . . . . . . . . 55
10.2 Architecture of a Multilayer Perceptron (MLP) . . . . . . . . . . . . . . . . . . . . . . . . . . . 55
10.3 Activation Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56
10.3.1 Why ReLU is Preferred in Hidden Layers . . . . . . . . . . . . . . . . . . . . . . . . . . 56
10.4 Loss Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56
10.5 Training: Backpropagation and Gradient Descent . . . . . . . . . . . . . . . . . . . . . . . . . . 57
10.5.1 The Gradient Descent Update Rule . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 57
10.5.2 Backpropagation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 57
10.5.3 Mini-Batch Stochastic Gradient Descent . . . . . . . . . . . . . . . . . . . . . . . . . . . 57
10.5.4 Adaptive Optimisers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 57
10.6 Hyperparameters of Neural Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58
10.7 Regularisation of Neural Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58
10.7.1 L2 Regularisation (Weight Decay) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58
10.7.2 Dropout . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58
10.7.3 Early Stopping . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59
10.8 The Bias-Variance Trade-off in Neural Networks . . . . . . . . . . . . . . . . . . . . . . . . . . 59
10.9 Neural Networks vs. Other Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 60
11 Chapter 11 – Unsupervised Learning 62
11.1 Principal Component Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 62
11.1.1 Motivation and Intuition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 62
11.1.2 Loading Vectors and Scores . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 62
11.1.3 Proportion of Variance Explained . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 62
11.1.4 Choosing the Number of Components . . . . . . . . . . . . . . . . . . . . . . . . . . . . 63
11.1.5 The Biplot . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 63
11.1.6 PCA for Regression: Principal Components Regression . . . . . . . . . . . . . . . . . . . 63
11.2K-Means Clustering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 63
3

## Page 4

11.2.1 Problem Formulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 63
11.2.2 TheK-Means Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
11.2.3 Choosing the Number of ClustersK. . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
11.3 Hierarchical Clustering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 65
11.3.1 Motivation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 65
11.3.2 The Agglomerative Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 65
11.3.3 Linkage Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 65
11.3.4 Dissimilarity Measures . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 65
11.3.5 Dendrogram and Cluster Stability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 66
11.4 Comparing Clustering Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 66
11.5 Practical Considerations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 66
4

## Page 5

1 Chapter 1 – Introduction to Statistical Learning
1.1 What is Statistical Learning?
Statistical learning refers to a vast collection of tools and methods for understanding and modelling data. At
its core, statistical learning is about estimating a functionf that relates a set of input variables (predictors,
features)X= (X 1,X 2,...,Xp)to an output variable (response)Y. The general relationship is written as:
Y=f(X) +ε
Here,f is a fixed but unknown function capturing the systematic relationship between the inputs and the output.
The termεis a random error term, independent ofX, with E[ε] = 0. Statistical learning is fundamentally
concerned with estimatingffrom data.
1.2 Notation
The following notation is used throughout the compendium and corresponds directly to ISLP:
Symbol Meaning
nNumber of observations (rows) in the dataset
pNumber of predictors (features/columns)
xij Value of thej-th predictor for thei-th observation
Xn×pdata matrix of all predictor values
yi Value of the response variable for observationi
YThe response variable (what we want to predict or
explain)
ˆfOur estimated model or function
ˆyi Fitted/predicted value for observationi:ˆyi = ˆf(xi)
εRandom error term:E[ε] = 0, independent ofX
βj Regression coefficient for predictorj
ˆβj Estimated regression coefficient for predictorj
1.3 Why Do We Estimatef?
There are two main reasons for estimatingf:predictionandinference. These represent fundamentally
different goals that affect how we choose and interpret models.
1.3.1 Prediction
In a prediction setting, we have access to a set of inputsX, but the responseY is not directly available or is
costly to obtain. We use our estimateˆfto predict:
ˆY= ˆf(X)
In this setting, we typically treatˆf as ablack box: we care only about how accurateˆY is as a prediction for
Y, not about the internal workings ofˆf. The accuracy of ˆYdepends on two sources of error:
• Reducible error: The gap between ˆf and the truef. This error can, in principle, be reduced by
choosing a better estimation method.
• Irreducible error: The variability inY due toε. Even if we knewf perfectly, this error would remain
because Y = f(X) +εand εcontains unknown influences not captured byX. No model, however
sophisticated, can eliminate this error.
The expected mean squared prediction error for a new observation atx0 can be decomposed as:
E
[
(Y−ˆf(x 0))2]
=
[
f(x 0)−ˆf(x 0)
]2
  
reducible
+Var(ε)  
irreducible
5

## Page 6

1.3.2 Inference
In an inference setting, we are interested in understanding the relationship betweenX and Y — specifically
howYchanges as individual predictors change. Key inferential questions include:
•Which predictors are associated with the response?
•What is the direction and magnitude of each predictor’s effect?
•Is the relationship linear, or is it more complex?
In inference, interpretability matters greatly. A linear model where each coefficient has a clear meaning is often
preferred over a highly flexible black-box model, even if the latter predicts slightly better.
Prediction vs. Inference: A Key Distinction
Prediction: We wantˆY to be close toY. The form ofˆf does not matter — only its accuracy. Black-box
methods (random forests, neural networks) are often preferred.
Inference: We want to understand howY changes withX. The form of ˆf must be interpretable.
Simpler models (OLS, logistic regression) are preferred.
Many real problems involve both goals simultaneously. The choice of method should reflect the primary
objective.
1.4 Parametric vs. Non-Parametric Methods
Statistical learning methods fall into two broad families based on how they make assumptions about the shape
off:
1.4.1 Parametric Methods
Parametric methods assume a specific functional form forf before looking at the data. The estimation process
then reduces to estimating a finite set of parameters.
Example: Linear regression assumesf(X) =β0 +β1X1 +···+βpXp. We estimate the(p + 1)parameters
β0,β1,...,βp using the training data.
Advantages: - Computationally simple and fast. - Interpretable coefficients and statistical tests. - Works well
with limited data if the assumed form is approximately correct.
Disadvantages: - If the assumed form does not match the truef, the model will be misspecified and produce
poor estimates. - Adding many parameters to increase flexibility risksoverfitting— fitting the noise in the
training data rather than the true signal.
1.4.2 Non-Parametric Methods
Non-parametric methods make no explicit assumption about the functional form off. They allowf to take
any shape that closely follows the observed data.
Examples: K-nearest neighbours, smoothing splines, local regression, kernel methods.
Advantages: - Much more flexible; can capture complex non-linear relationships. - Do not suffer from model
misspecification.
Disadvantages: - Require far more data to produce accurate estimates, because no structural assumptions
reduce the estimation problem. - Often harder to interpret — there are no coefficients to read off. - Risk of
overfitting if not carefully regularised.
Parametric Non-Parametric
Assumes
form of
f?
Yes (e.g. linear) No
Estimation Fit parameters Fitfdirectly to data
Data
require-
ment
Low High
InterpretabilityHigh Low
6

## Page 7

Parametric Non-Parametric
Risk of
misspeci-
fication
Yes No
Risk of
overfit-
ting
With many parameters Without regularisation
1.5 Supervised vs. Unsupervised Learning
Supervised learningis the setting where both predictor variablesX and a response variableY are observed.
The goal is to learn the mapping fromXtoY:
•IfYisquantitative(numerical): this is aregressionproblem.
•IfYisqualitative(categorical): this is aclassificationproblem.
Unsupervised learningis the setting where we observe onlyX, with no response variableY. The goal is to
discover interesting structure or patterns inXitself. Common tasks include:
•Clustering: group observations into clusters of similar observations.
• Dimensionality reduction: find a low-dimensional representation ofX that preserves most of the
information (e.g. PCA).
Unsupervised learning is more exploratory and harder to evaluate (there is no “correct answer” to compare
predictions against).
The Prediction–Interpretability Trade-off
Flexible methods (e.g. neural networks, random forests) can capture very complex relationships and
tend to predict well, but are difficult to interpret.
Restrictive methods (e.g. linear regression) are highly interpretable but may miss complex structure.
The right choice depends on whether the goal isprediction(favour flexibility) orinference(favour
interpretability). There is no universally best method — this is sometimes called theNo Free Lunch
theorem.
Theory Questions – Chapter 1
Q1: What is the difference between reducible and irreducible error? Can we, in principle,
eliminate both by choosing a better model?
A:Reducible error arises because our estimateˆf is not a perfect approximation of the true functionf.
In principle, this error can be reduced by choosing a more appropriate statistical learning method or by
collecting more data. Irreducible error, on the other hand, arises fromε— the random variation inY
that cannot be explained byX, no matter how well we knowf. This includes unmeasured variables,
inherent randomness in the process, and measurement noise. No model, however complex, can eliminate
irreducible error, because it is a fundamental property of the data-generating process, not of our
modelling choices.
Q2: When would you prefer a more restrictive, less flexible model over a highly flexible
one?
A:A more restrictive model is preferred in two main situations. First, when the goal isinference: if we
want to understand how individual predictors relate to the response — for example, "does advertising
budget affect sales?" — a linear model with interpretable coefficients is far easier to reason about
than a black-box neural network. Second, when data arescarce: flexible models have many effective
parameters and require large samples to estimate them well. With smalln, a flexible model will overfit
to noise. A restrictive model that approximately captures the truef will generalise better in this situation.
Q3: Is it possible to have a problem that is simultaneously supervised and unsupervised?
A:In practice, some problems have elements of both. For example, we might use unsupervised methods
7

## Page 8

(like PCA) to reduce the dimensionality ofX before applying a supervised model. However, supervised
and unsupervised learning are conceptually distinct: supervised learning requires a labelled responseY,
while unsupervised learning operates withoutY. Semi-supervised learning, a related field, explicitly
combines both by using a small labelled dataset alongside a large unlabelled one.
8

## Page 9

2 Chapter 2 – Assessing Model Accuracy
2.1 Measuring the Quality of Fit
After fitting a statistical model, we need to evaluate how well it performs. In the regression setting, the most
commonly used measure isMean Squared Error (MSE):
MSE= 1
n
n∑
i=1
(
yi−ˆf(xi)
)2
MSE measures the average squared difference between the predicted valuesˆf(xi)and the true valuesyi. Smaller
MSE means better fit.
Critically, we must distinguish between two types of MSE:
• Training MSE: computed using the same observations that were used to fitˆf. A flexible model can
always achieve very low training MSE by fitting the training data closely, including its noise — this is
overfitting.
• Test MSE: computed using new observations not used in training. This is what we truly care about —
how well does the model generalise to unseen data?
Training MSE is not enough
Training MSE always decreases (or stays flat) as the model becomes more flexible. This makes it apoor
criterion for model selection. A model that memorises the training data will have very low training MSE
but very high test MSE.
Always evaluate model performance on held-out test data.If test data are not available, use
cross-validation (Chapter 5).
2.1.1 The Pattern of Training and Test MSE as Flexibility Increases
As model flexibility increases (e.g., increasing polynomial degree, decreasing regularisation penalty, growing a
deeper tree):
• Training MSEmonotonically decreases. The more flexible the model, the better it can fit the training
data.
• Test MSEfollows a characteristicU-shape: it initially decreases as the model captures true structure
in the data, reaches a minimum at the optimal flexibility, then increases again as the model begins to
overfit.
The minimum of the test MSE curve defines theoptimal level of flexibility— the “sweet spot” between
underfitting and overfitting.
2.2 The Bias-Variance Trade-off
The U-shape of test MSE is explained by thebias-variance trade-off, one of the most fundamental concepts
in statistical learning. For any test pointx0, the expected test MSE decomposes into exactly three components:
E
[
(y0−ˆf(x 0))2]
=Var
(ˆf(x 0)
)
  
variance
+
[
Bias
(ˆf(x 0)
)]2
  
bias2
+Var(ε)  
irreducible error
Each component has a precise meaning:
VarianceVar( ˆf(x0))refers to how much the estimated functionˆf would change if we fitted it using a different
training dataset. A high-variance model is very sensitive to the specific training data used — small changes in
the training set lead to large changes inˆf. Flexible models (deep trees, high-degree polynomials) tend to have
high variance.
Bias Bias( ˆf(x0)) =f(x0)−E[ ˆf(x0)]refers to the error introduced by approximating the true, possibly complex
function f with a simpler modelˆf. If the model is too rigid (e.g. fitting a straight line to clearly non-linear
data), it will systematically miss the true pattern regardless of how much data we have. Restrictive models
tend to have high bias.
9

## Page 10

Irreducible errorVar(ε)is the noise inherent in the data, unrelated to the model. It sets a lower bound on
the achievable test MSE.
Component High flexibility Low flexibility
Variance High — model changes a lot with data Low — stable estimates
Bias Low — can approximate complexfHigh — too simple to capturef
Irreducible Unchanged Unchanged
The trade-off: Increasing model flexibility reduces bias but increases variance. Decreasing flexibility reduces
variance but increases bias. The minimum test MSE occurs where the sum of squared bias and variance is
minimised — the sweet spot. Importantly,we can never reduce irreducible error; it is always present
regardless of our model.
Why does variance increase with flexibility?
Consider fitting a polynomial of degree 10 to 15 training observations. With so many parameters relative
to observations, a small change in even one training point can lead to a dramatically different fitted
curve. The model "memorises" specific data points rather than learning general patterns. This extreme
sensitivity to the training data is what we mean by high variance.
Contrast this with a linear fit: regardless of which specific training observations we use, the fitted line
will be broadly similar (similar slope, similar intercept), because the model is constrained to a simple
form. It may be systematically wrong (bias), but it is consistently wrong.
2.3 The Classification Setting
So far we have discussed regression (quantitativeY). Forclassification(qualitative Y), the natural error
measure is themisclassification rate:
Error rate= 1
n
n∑
i=1
1(yi̸= ˆyi)
where1(·)is the indicator function (1 if the condition holds, 0 otherwise). As with MSE, we distinguish training
error rate from test error rate, and the test error rate is what we care about.
The bias-variance trade-off applies in classification too: flexible models may overfit (low training error, high
test error) and inflexible models may underfit (high error on both).
2.3.1 The Bayes Classifier
The theoretically optimal classifier — the one that minimises the expected test error rate — is theBayes
classifier. It classifies each observation to the classj with the highest posterior probability given the observed
inputs:
ˆy(x0) = arg max
j
Pr(Y=j|X=x 0)
TheBayes error rateis the lowest possible test error rate, achieved by the Bayes classifier:
Bayes error rate= 1−E
[
max
j
Pr(Y=j|X)
]
This is analogous to the irreducible error in regression. Even the Bayes classifier makes mistakes when the true
class probabilities are not extreme (i.e., whenPr(Y =j|X)is not close to 0 or 1 for all classes). The Bayes
error rate is the irreducible lower bound for any classifier.
The Bayes classifier is a theoretical benchmark only
In practice, we never knowPr(Y =j|X=x0), so we cannot actually implement the Bayes classifier. It
serves purely as a gold standard against which real classifiers are measured. Methods like LDA, logistic
regression, and KNN can be viewed as different ways of estimating these posterior probabilities.
10

## Page 11

2.3.2 K-Nearest Neighbours Classifier
Since we cannot computePr(Y =j|X=x0)directly, a simple approximation is to estimate it from theK
nearest training observations. TheK-Nearest Neighbours (KNN) classifierdefines the neighbourhood
N0 of x0 as the set ofK training points closest tox0 (using Euclidean distance), and estimates the class
probabilities as the proportion of neighbours from each class:
Pr(Y=j|X=x 0) = 1
K
∑
i∈N0
1(yi =j)
The observation is then assigned to the class with the highest estimated probability.
The role ofK: The choice ofKdirectly controls the bias-variance trade-off in KNN:
KDecision boundary Bias Variance Behaviour
K= 1Very jagged, irregular Very low Very high Memorises training
data; training error =
0
K
moderate
Smooth, curved Moderate Moderate Best test performance
in practice
K=nFlat (predicts majority class
everywhere)
Very high Zero Ignores all local
structure
Note that in KNN, increasingK reducesflexibility (increases bias, decreases variance), which is the opposite
direction from, e.g., polynomial degree where increasing the degree increases flexibility. The optimalK is
chosen by cross-validation.
Key insight: KNN approximates the Bayes classifier
The Bayes classifier assigns each observation to the class with highest true posterior probabilityPr(Y =
j|X=x0).
KNN estimates this probability by:ˆP(Y=j|X=x 0) = 1
K
∑
i∈N0 1(yi =j)
As n→∞and K→∞such thatK/n→0, KNN converges to the Bayes classifier. With finite data
and optimalK, KNN can come surprisingly close to the Bayes error rate even when the true decision
boundary is highly non-linear.
2.4 Summary: Key Principles of Model Assessment
The following principles underpin all model evaluation in statistical learning and will resurface throughout the
compendium:
1.Always evaluate on test data(or use cross-validation). Training error is a misleading guide.
2. The bias-variance trade-offmeans there is no universally best level of model flexibility. The optimal
model minimises expected test error.
3. Irreducible errorsets a lower bound on achievable test error. A model that achieves the Bayes error
rate (in classification) or Var(ε)(in regression) is doing as well as theoretically possible.
4. Model selectionshould be based on estimated test error, not training error. Methods like cross-validation
(Chapter 5) and information criteria (Chapter 6) are designed for this purpose.
Theory Questions – Chapter 2
Q1: Explain the bias-variance trade-off. Why does test MSE follow a U-shape as model
flexibility increases?
A:The expected test MSE at any pointx0 decomposes into three terms:Var( ˆf(x0)) +Bias2( ˆf(x0)) +
Var(ε). As model flexibility increases, bias decreases (the model can approximate more complex
functions) but variance increases (the model becomes more sensitive to the specific training data).
When flexibility is low, high bias dominates and test MSE is large. As flexibility increases, bias falls
faster than variance rises, so test MSE decreases. Eventually, variance grows rapidly and begins to
dominate, causing test MSE to rise again. This produces the characteristic U-shape. The minimum
of the curve identifies the optimal flexibility. The irreducible errorVar(ε)is constant throughout and
11

## Page 12

represents the floor below which test MSE cannot fall.
Q2: What is the Bayes error rate and why is it important? Why can we not achieve zero
error even with a perfect classifier?
A:The Bayes error rate is1−E[maxj Pr(Y =j|X)]— the expected error of the theoretically optimal
Bayes classifier, which always predicts the most probable class givenX. It is the irreducible lower
bound on the test error rate for any classifier: no method, regardless of complexity, can do better. The
reason we cannot achieve zero error is that the classes may genuinely overlap in the feature space — i.e.,
two observations with identicalX may belong to different classes. This overlap is due to unmeasured
factors and inherent randomness (captured byε), which no model can account for. The Bayes error rate
quantifies exactly this irreducible confusion.
Q3: A model achieves training MSE = 0.5 and test MSE = 8.2. What does this tell you
about the model, and what would you do?
A:The large gap between training MSE (0.5) and test MSE (8.2) is a classic sign ofoverfitting(high
variance). The model has fitted the noise in the training data very closely — memorising it rather than
learning the true underlying pattern — and therefore fails to generalise to new observations. To address
this, one would reduce model complexity: for example, reduce polynomial degree, increase regularisation
penalty (Ridge/Lasso), reduce tree depth, or use fewer features. Cross-validation (Chapter 5) should be
used to select the appropriate level of complexity based on estimated test error.
12

## Page 13

3 Chapter 3 – Linear Regression
Linear regression is the most fundamental and widely used statistical learning method. Despite its simplicity, it
forms the conceptual backbone for many more advanced methods (Ridge, Lasso, GLMs, GAMs). Understanding
linear regression deeply — including its assumptions, diagnostics, and limitations — is essential for the exam.
3.1 Simple Linear Regression
Simple linear regression (SLR) models the relationship between asingle predictorX and a quantitative
responseYas a linear function:
Y=β0 +β1X+ε
Here, β0 is theintercept(the expected value ofY when X = 0) andβ1 is theslope(the expected change in
Yfor a one-unit increase inX). The fitted model is:
ˆy=ˆβ0 + ˆβ1x
3.1.1 Estimating Coefficients — Ordinary Least Squares (OLS)
The principle ofOrdinary Least Squaresis to choose ˆβ0 and ˆβ1 so as to minimise theResidual Sum of
Squares (RSS)— the total squared deviation between the observed and fitted values:
RSS=
n∑
i=1
(yi−ˆyi)2 =
n∑
i=1
(yi−ˆβ0−ˆβ1xi)2
Taking partial derivatives of RSS with respect toˆβ0 and ˆβ1 and setting them to zero yields the closed-form
OLS estimators:
ˆβ1 =
∑n
i=1(xi−¯x)(yi−¯y)∑n
i=1(xi−¯x)2 = Cov(X,Y)
Var(X) , ˆβ0 = ¯y−ˆβ1¯x
where¯x= 1
n
∑xi and¯y= 1
n
∑yi are the sample means.
Key properties of OLS estimators:
•They areunbiased:E[ ˆβ0] =β0 andE[ ˆβ1] =β1.
• By theGauss-Markov theorem, OLS estimators are BLUE —BestLinearUnbiasedEstimators —
meaning that among all linear unbiased estimators, OLS has the smallest variance. This holds when
errors are homoscedastic and uncorrelated.
•The fitted line always passes through the point(¯x,¯y).
3.1.2 Assessing the Accuracy of Coefficient Estimates
Because ˆβ0 and ˆβ1 are estimated from a sample, they are random variables with associated uncertainty. The
standard errorof ˆβ1 measures this uncertainty:
SE( ˆβ1) =
√
ˆσ2
∑n
i=1(xi−¯x)2,SE( ˆβ0) = ˆσ
√
1
n + ¯x2
∑n
i=1(xi−¯x)2
where ˆσ2 = RSS/(n−2)is the estimated error variance (we divide byn−2because two parameters were
estimated).
Interpretation of SE: A smaller SE means we have estimated the coefficient more precisely. SE decreases when:
- n is larger (more data→more precise estimates). -Var(X)is larger (more spread inX→better-identified
slope). - The true error varianceσ2 is smaller (less noise in the data).
Hypothesis test forβ1: We typically testH0 :β1 = 0(no linear relationship betweenX and Y) against
H1 :β1̸= 0. The test statistic is:
13

## Page 14

t=
ˆβ1
SE( ˆβ1)
Under H0, t follows at-distribution withn−2degrees of freedom. A large|t|(equivalently, a smallp-value)
gives evidence againstH0.
A 95% confidence interval forβ1 is:
ˆβ1±tα/2,n−2·SE(ˆβ1)
This interval contains the trueβ1 in approximately 95% of repeated samples. If the interval does not include
zero, we rejectH0 :β1 = 0at the 5% level.
3.1.3 Assessing Model Accuracy: RSE,R 2, and theF-statistic
Beyond the coefficients, we need measures of how well the overall model fits the data.
Residual Standard Error (RSE)estimates the standard deviation ofε— the average size of a residual:
RSE=
√
RSS
n−p−1
where p is the number of predictors (so for SLR,p = 1and RSE =
√
RSS/(n−2)). RSE is on the same scale
asY— it tells us how far, on average, the model’s predictions are from the true values.
R2 (coefficient of determination)measures the proportion of total variance inYexplained by the model:
R2 = 1−RSS
TSS,TSS=
n∑
i=1
(yi−¯y)2
where TSS is theTotal Sum of Squares(total variability inY before fitting).R2∈[0, 1]: -R2 = 1: perfect
fit (all variation explained). -R2 = 0: the model explains nothing beyond the mean.
Important: R2 always increases (or stays flat) when we add more predictors, even if they are irrelevant. For
multiple regression,adjustedR 2 corrects for this (see Chapter 6).
In simple linear regression,R2 =Cor(X,Y) 2 =r 2 — the squared correlation coefficient.
TheF-statistictests whether at least one predictor in the model is useful:
H0 :β1 =β2 =···=βp = 0vs.H 1 :at least oneβj̸= 0
F= (TSS−RSS)/p
RSS/(n−p−1)
Under H0, F follows anF-distribution withp and n−p−1degrees of freedom. A largeF (small p-value)
gives evidence that the model explains a meaningful amount of variation.
14

## Page 15

Key output statistics — summary table
Statistic Formula What it tells you
ˆβj OLS estimate Expected∆Yper 1-unit↑X j, others fixed
SE( ˆβj)
√
ˆσ2/ ∑(xij−¯xj)2 Uncertainty in ˆβj
t-statistic ˆβj/SE( ˆβj)TestsH 0 :βj = 0
p-valuePr(|T|≥|t||H0)<0.05→predictor significant
RSE
√
RSS/(n−p−1)Avg. residual size, in units ofY
R2 1−RSS/TSS Proportion ofY-variance explained
F-statistic (TSS−RSS)/p
RSS/(n−p−1) Tests if any predictor is useful
3.1.4 Confidence Intervals vs. Prediction Intervals
After fitting a model, we often want to make statements aboutY at a new input valuex0. Two types of
intervals serve different purposes:
Confidence interval (CI)for themean responseatx 0, i.e., forE[Y|X=x0] =β0 +β1x0:
ˆy(x0)±tα/2,n−2·SE(ˆy(x0))
This expresses our uncertainty about where the true regression line lies — uncertainty that shrinks asn grows.
Prediction interval (PI)for asingle new observationy 0 atx 0:
ˆy(x0)±tα/2,n−2·
√
SE(ˆy(x0))2 + ˆσ2
The PI is alwayswiderthan the CI because it must account for two sources of uncertainty: 1. Uncertainty
about the true meanE[Y|X=x0](same as CI). 2. The irreducible varianceˆσ2 — even if we knew the true
line exactly, individual observations scatter around it.
CI vs. PI: A practical distinction
CI: "We are 95% confident that theaveragesales for stores with 1000 customers lies between 40 and 60
units."
PI: "We are 95% confident thatthis specific storewith 1000 customers will sell between 25 and 75 units."
The PI is wider because even after perfectly estimating the mean response, individual observations still
vary around the mean. The PI uncertainty never vanishes even with infinite data (irreducible error),
whereas the CI width→0asn→∞.
3.2 Multiple Linear Regression
When there arep>1predictors, themultiple linear regressionmodel is:
Y=β0 +β1X1 +β2X2 +···+βpXp +ε
In matrix form:Y=X β+ε, whereXis the n×(p + 1)design matrix (with a column of ones for the intercept)
andβ= (β0,β1,...,βp)⊤.
The OLS solution minimises∥Y−Xβ∥2 and has a closed-form solution:
ˆβ= (X⊤X)−1X⊤Y
This requiresX⊤Xto be invertible, which fails when predictors are perfectly collinear or whenp≥n.
15

## Page 16

3.2.1 Interpretation of Coefficients in Multiple Regression
ˆβj represents the expected change inY for a one-unit increase inXj,holding all other predictors constant.
This “partial effect” interpretation is fundamentally different from simple regression.
Why coefficients change from simple to multiple regression: In simple regression,ˆβ1 captures the total
association betweenX1 and Y, including any association that runs through other variables correlated withX1.
In multiple regression,ˆβ1 isolates the direct effect ofX1 on Y, controlling for the other predictors. This means:
• A predictor that appears significant in simple regression may become insignificant when others are
included (confounding removed).
• A predictor that appears insignificant alone may become significant when confounders are controlled for
(suppression removed).
3.2.2 Key Questions in Multiple Regression
Question Answer
Is at least one predictor useful?F-test: largeF, smallp-value
Which specific predictors are useful? Individualt-tests; variable selection (Chapter 6)
How well does the model fit? RSE (in units ofY) andR 2∈[0,1]
How accurate are predictions? CI (mean response) or PI (new observation)
3.3 Extensions of the Linear Model
3.3.1 Interaction Terms
The standard linear model assumes that the effect of each predictor onY does not depend on the values of
other predictors — anadditivityassumption. When this is violated, we includeinteraction terms:
Y=β0 +β1X1 +β2X2 +β3X1X2 +ε
The effect ofX1 on Y is nowβ1 +β3X2, which depends onX2. This is aninteraction: the effect of one
predictor is moderated by the level of another.
The Hierarchy Principle
If an interaction termX1X2 is included in the model, the main effectsX1 and X2 mustalwaysbe
included too, even if their individualp-values are not significant.
Reason: The interpretation of the interaction coefficientˆβ3 depends on the main effects being present.
Removing a main effect while keeping its interaction changes the meaning of the other coefficients
fundamentally.
3.3.2 Polynomial Regression
By including powers of a predictor as additional columns in the design matrix, we can capture non-linear
relationships while still using OLS:
Y=β0 +β1X+β2X 2 +···+βdXd +ε
This is still a linear model (linear in the parametersβj), so all OLS theory applies. The polynomial degreed
controls flexibility and is chosen via cross-validation. In practice,d> 4is rarely used due to instability at the
boundaries (Runge’s phenomenon).
3.3.3 Qualitative (Categorical) Predictors
A categorical variable withk levels cannot be entered into a regression as a single column of integers (that
would impose an artificial ordering). Instead, it is encoded usingk−1dummy variables(one level is omitted
as the baseline/reference category):
Example:X={A,B,C}⇒createDB =1(X=B), D C =1(X=C)
16

## Page 17

The omitted level (hereA) is thereference category. The coefficients ˆβB and ˆβC measure the expected
difference inYrelative to levelA, holding other predictors fixed.
If allk dummies are included simultaneously, perfect multicollinearity arises (the dummy columns sum to the
intercept column) — this is thedummy variable trap, and it makesX⊤Xsingular and OLS undefined.
3.4 Potential Problems in Linear Regression
Fitting a linear model is only the beginning. It is essential to check whether the underlying assumptions are
satisfied. Below are the most important diagnostic checks:
3.4.1 1. Non-linearity of the Relationship
OLS assumes a linear relationship between eachXj and Y.Detection: plot residuals ei =yi−ˆyi against
fitted valuesˆyi. A curved pattern (U-shape or inverted U) indicates non-linearity.
Consequence: Systematic bias in predictions — the model is consistently wrong in certain regions ofX.
Remedy: Add polynomial terms, splines, or use a non-linear method (Chapter 7).
3.4.2 2. Heteroscedasticity (Non-constant Error Variance)
OLS assumes Var(εi) =σ2 for alli. When this fails, errors have different variances for different observations.
Detection: a “funnel” or “fan” shape in the residuals-vs-fitted plot (residuals spread wider for largerˆy).
Consequence: Standard errors for coefficients are wrong, makingt-tests and CIs invalid.
Remedy: TransformY(e.g.,logY), use weighted least squares, or use robust standard errors.
3.4.3 3. Correlated Errors (Autocorrelation)
OLS assumes errorsεi are independent. In time series data, consecutive observations often have correlated
errors.
Detection: Plot residuals against time/observation index; look for systematic patterns (runs of positive or
negative residuals).
Consequence: Standard errors are underestimated — the model appears more precise than it is.
Remedy: Include time lags as predictors, use time-series models.
3.4.4 4. Outliers
An outlier is an observation with an unusually large residual — the observedyi is far fromˆyi.
Detection: Studentised residualsri =e i/ˆσ(−i); values|ri|>3are flagged as potential outliers.
Consequence: Outliers inflate RSE, reducingR2 and inflating standard errors. They may or may not strongly
affect coefficient estimates (depends on their leverage).
3.4.5 5. High-Leverage Points
A high-leverage point has an unusual value ofX relative to the other observations — it is far from¯xin the
predictor space. Unlike outliers (which concerny), leverage concerns onlyx.
Detection: The hat matrix diagonalhii (leverage): valueshii >2(p+ 1)/nindicate high leverage.
Consequence: High-leverage points can have a disproportionate influence on the fitted regression line. A
single high-leverage outlier can drastically changeˆβ.
Distinction: An outlier has a large residual but may have low leverage (it lies at typicalx-values but unusual
y). A high-leverage point has unusualx-values but may not necessarily have a large residual (if it happens to
fall on the regression line). The most dangerous combination is a high-leverage outlier.
17

## Page 18

3.4.6 6. Multicollinearity
Multicollinearity occurs when two or more predictors are highly correlated with each other.
Detection: - Pairwise correlation matrix (simple collinearity). -Variance Inflation Factor(VIF):VIF( ˆβj) =
1
1−R2
Xj|X−j
, whereR2
Xj|X−j
is theR2 from regressingXj on all other predictors. VIF> 5–10signals a problem.
Consequence: Individual coefficient estimates become very imprecise (large SE), even if the model as a whole
fits well. TheF-statistic may be significant (some combination of predictors is useful) while individualt-tests
are not (we cannot isolate individual effects).
Remedy: Drop one of the correlated predictors, combine them (e.g., PCA), or use Ridge regression (Chapter
6), which is specifically designed to handle collinearity.
Regression Diagnostics Checklist
1. Residuals vs. Fitted: check for non-linearity (curved pattern) and heteroscedasticity (funnel shape)
2.Q-Q plot of residuals: check for normality of errors (needed for valid inference with smalln)
3.Residuals vs. Order: check for autocorrelation (patterns over time)
4.Studentised residuals: identify outliers (|ri|>3)
5.Leverage / hat valueshii: identify high-leverage points
6.VIF: detect multicollinearity (>5–10is problematic)
Problem Detection Main consequence Remedy
Non-linearity Curved residual plot Biased predictions Polynomials,
splines, GAMs
Heteroscedasticity Funnel residual plot Invalid SEs and CIs Log transform,
WLS
Autocorrelation Residuals vs. time SEs too small Lagged predictors,
time-series models
Outliers|r i|>3Inflated RSE Investigate,
possibly remove
High leverageh ii >2(p+ 1)/nDistorted ˆβInvestigate,
possibly remove
Multicollinearity VIF>10Large SEs for affected ˆβj Drop predictor,
Ridge, PCA
3.5 KNN Regression vs. Linear Regression
An important comparison is between parametric linear regression and the non-parametricK-Nearest Neigh-
bours (KNN) regression. KNN regression predicts the response at a new pointx0 as the average of theK
nearest training observations:
ˆf(x 0) = 1
K
∑
i∈N0
yi
whereN 0 is the set ofKtraining points closest tox 0.
Aspect Linear Regression KNN Regression
Functional form Linear:ˆy= ˆβ0 + ˆβ1x1 +···None — adapts to local data
Interpretability High: coefficients have direct meaning Low: no global formula
Works best when True relationship is approximately linear True relationship is highly non-linear
With smallnGood (few parameters to estimate) Poor (neighbours may not be truly
“near”)
With largepHandles well (coefficients for each predictor) Degrades:curse of dimensionality
Extrapolation Extrapolates beyond training range (linearly) Cannot extrapolate: no nearby
neighbours
18

## Page 19

The curse of dimensionality: In high dimensions (p large), all observations become approximately equidistant
from any test point, so theK “nearest” neighbours are not actually close in feature space. KNN loses its local
averaging advantage and may perform worse than a simple linear model. Linear regression does not suffer from
this problem because it estimates global parameters rather than relying on local neighbourhoods.
Theory Questions – Chapter 3
Q1: What does the OLS estimator minimise, and what properties does it have? Under
what conditions is OLS BLUE?
A:OLS minimises the Residual Sum of Squares:RSS = ∑n
i=1(yi−ˆβ0−ˆβ1xi)2. The resulting estimators
ˆβ0 and ˆβ1 are unbiased (E[ ˆβj] =βj) and have a closed-form solution. By the Gauss-Markov theorem,
OLS estimators are BLUE — Best Linear Unbiased Estimators — meaning that among all estimators
that are both linear inyand unbiased, OLS achieves the smallest variance. The theorem requires
four conditions: (1) the model is correctly specified (no omitted non-linearity), (2) errors have zero
mean (E[ε] = 0), (3) errors are homoscedastic (Var(εi) =σ2 for alli), and (4) errors are uncorrelated
(Cov(εi,εj) = 0for i̸=j). Note that normality of errors is NOT required for Gauss-Markov; it is needed
for exact inference (hypothesis tests and CIs) in small samples.
Q2: What is the difference between a confidence interval and a prediction interval for a
new observation? Why is the prediction interval always wider?
A:Both intervals are centred at the same fitted valueˆy(x0), but they capture different quantities and
therefore have different widths. Aconfidence intervalexpresses uncertainty about the true mean
response E[Y|X=x0] =β0 +β1x0 — it quantifies how precisely we have estimated the regression line
at x0. Aprediction intervalexpresses uncertainty about a specific new observationy0 at x0. The PI
is wider because it accounts for two independent sources of uncertainty: (1) the same uncertainty in the
estimated regression line (as in the CI), and (2) the irreducible errorˆσ2 from the fact that even if we
knew the true line, a new observation would still randomly deviate from it. Asn→∞, the CI width
shrinks to zero (we estimate the line perfectly), but the PI width converges to±tα/2·σ— it never
vanishes because irreducible error remains.
Q3: In a multiple regression, a predictor has a larget-statistic and smallp-value, yet the
F-statistic for the overall model is not significant. How can this happen? What is the
reverse situation?
A:This situation can arise when there are many predictors (p is large relative ton). With many
predictors, the overallF-test has low power because the model has many degrees of freedom to estimate.
However, if one predictor happens to be strongly correlated withY by chance, its individualt-statistic
may be large. The reverse situation — significantF but no significant individualt-statistics — is
more common and arises frommulticollinearity: when predictors are highly correlated, no individual
predictor can be identified as uniquely responsible for explainingY, so all individualt-statistics are
small. Yet, collectively the predictors do explainY well, giving a largeF. This is why one should check
both theF-statistic (overall model utility) and individualt-statistics (individual predictor utility), and
always check VIFs for multicollinearity.
19

## Page 20

4 Chapter 4 – Classification
Classification is the task of predicting a qualitative (categorical) responseY. Common examples include
predicting whether an email is spam or not, whether a patient has a disease, or whether a loan will default.
The responseYtakes values in a finite set of classes{1,2,...,K}.
4.1 Why Not Linear Regression for Classification?
It might seem natural to encode the classes as numbers (e.g.Y∈{0, 1}for binary classification) and fit a linear
regression. This approach has two fundamental problems:
Problem 1 — Ordering: IfY has three classes encoded as{0, 1, 2}, linear regression implies class 2 is “twice
as far” from class 0 as class 1 — an artificial ordering that may be meaningless for nominal classes like {stroke,
drug overdose, epilepsy}.
Problem 2 — Out-of-range predictions: Linear regression produces predicted valuesˆy∈(−∞,+∞), while
class probabilities must lie in[0, 1]. Predictions outside[0, 1]are not interpretable as probabilities and can
cause downstream problems.
For binaryY∈{0, 1}, linear regression can be used as a rough approximation (it estimatesE[Y|X] = Pr(Y =
1|X)), but the two problems above motivate a proper classification framework.
4.2 Logistic Regression
Logistic regression directly models the probability thatY belongs to a particular class. For binary classification
withY∈{0,1}, we modelp(X) = Pr(Y= 1|X).
4.2.1 The Logistic Model
To ensurep(X)∈[0, 1]for all values ofX, we apply thelogistic (sigmoid) functionto a linear combination
of the predictors:
p(X) = eβ0+β1X
1 +eβ0+β1X = 1
1 +e−(β0+β1X)
The sigmoid maps any real number to(0,1): asβ0 +β1X→+∞,p(X)→1; asβ0 +β1X→−∞,p(X)→0.
Rearranging, we obtain thelog-odds(orlogit) representation, which reveals that logistic regression is a linear
model on the log-odds scale:
p(X)
1−p(X)  
odds
=e β0+β1X
log
( p(X)
1−p(X)
)
  
log-odds (logit)
=β0 +β1X
Interpretingβ1:
•A one-unit increase inXmultiplies theoddsbye β1.
• The effect on theprobabilityp(X)is non-linear and depends on the current value ofp: the change is
largest whenp≈0.5and smallest near 0 or 1 (the sigmoid is flattest at the extremes).
• Thesignof β1 determines the direction:β1 > 0means p(X)increases with X; β1 < 0means it decreases.
Odds vs. probability
If the probability of success isp = 0.75, then the odds arep/(1−p) = 0.75/0.25 = 3— meaning the
event is three times as likely to happen as not to happen.
If β1 = 0.5, then a one-unit increase inX multiplies the odds bye0.5≈1.65— a 65% increase in the
odds. This multiplicative interpretation holds regardless of the current value ofX, whereas the effect on
the probability itself is non-linear and context-dependent.
20

## Page 21

4.2.2 Estimating Coefficients — Maximum Likelihood
Unlike OLS for linear regression, there is no closed-form solution for the logistic regression coefficients. Instead,
they are estimated byMaximum Likelihood Estimation (MLE): we find the values ofβthat maximise
the probability of observing the data we actually observed.
The log-likelihood function is:
ℓ(β) =
∑
i:y i=1
logp(xi) +
∑
i:y i=0
log
(
1−p(xi)
)
This is also called thebinary cross-entropy loss(negated). Maximisingℓ(β)— equivalently, minimising the
cross-entropy — is done numerically using iterative algorithms such asNewton-Raphson(or IRLS: Iteratively
Reweighted Least Squares). The algorithm is guaranteed to converge to the global maximum because the
log-likelihood is concave.
The resulting MLE estimatorsˆβare: -Consistent: converge to the trueβas n→∞. -Asymptotically
normal: for largen, ˆβj∼N(βj,SE ( ˆβj)2), enablingz-tests and CIs. -Not unbiasedin finite samples (unlike
OLS), but the bias is small for moderaten.
4.2.3 Multiple Logistic Regression
Withppredictors, the model extends naturally:
log
( p(X)
1−p(X)
)
=β0 +β1X1 +β2X2 +···+βpXp
p(X) = eβ0+β1X1+···+βpXp
1 +eβ0+β1X1+···+βpXp
Each ˆβj is interpreted as the change in log-odds ofY = 1for a one-unit increase inXj,holding all other
predictors fixed. All the warnings about multiple regression (confounding, collinearity, etc.) apply here too.
4.2.4 Multinomial Logistic Regression (K >2Classes)
ForK >2classes, we choose one class as thebaseline(say classK) and modelK−1log-odds ratios relative
to it. For each classk= 1,...,K−1:
log
( Pr(Y=k|X)
Pr(Y=K|X)
)
=βk0 +βk1X1 +···+βkpXp
Solving for the probabilities:
Pr(Y=k|X) = eβk0+β⊤
k X
1 + ∑K−1
l=1 eβl0+β⊤
l X,Pr(Y=K|X) = 1
1 + ∑K−1
l=1 eβl0+β⊤
l X
AllK probabilities sum to 1. The interpretation ofˆβkj is: a one-unit increase inXj is associated with a change
in the log-odds of classkvs. baseline classKof ˆβkj, holding other predictors fixed.
4.3 Evaluating Classifiers
4.3.1 The Confusion Matrix
A confusion matrix tabulates predicted classes against actual classes. For binary classification:
Predicted: Negative (ˆY= 0) Predicted: Positive ( ˆY= 1)
Actual: Negative
(Y= 0)
True Negative (TN) False Positive (FP)
Actual: Positive
(Y= 1)
False Negative (FN) True Positive (TP)
21

## Page 22

From this table, a rich set of performance metrics can be derived:
Metric Formula Interpretation
Sensitivity/ Recall /
TPR
TP/(TP+FN) Proportion of actual positives correctly identified
Specificity/ TNR TN/(TN+FP) Proportion of actual negatives correctly identified
Precision/ PPV TP/(TP+FP)Proportion of predicted positives that are truly
positive
Overall Accuracy(TP+TN)/n Proportion of all observations correctly classified
False Positive Rate
(FPR)
FP/(FP+TN) = 1−Specificity; rate of false alarms
F1-score2· Precision·Recall
Precision+Recall Harmonic mean of precision and recall; useful
when classes are imbalanced
Accuracy is misleading for imbalanced classes
If 99% of observations belong to class 0, a classifier that always predicts 0 achieves 99% accuracy —
yet it is completely useless. In such settings, sensitivity and precision are far more informative. Always
report the full confusion matrix and multiple metrics when class imbalance is present.
4.3.2 The ROC Curve and AUC
Most classifiers produce a continuous score (e.g.ˆp(X)from logistic regression) that is then thresholded to
produce a class label. The default threshold is 0.5, but this is not always optimal.
TheReceiver Operating Characteristic (ROC) curvevisualises classifier performance across all possible
thresholds by plotting: -y-axis:True Positive Rate(Sensitivity) = TP/(TP +FN)- x-axis:False Positive
Rate= FP/(FP+TN) = 1−Specificity
As the threshold decreases from 1 to 0, more observations are classified as positive, so both TPR and FPR
increase. The ROC curve traces this trade-off.
Key reference points: - Aperfect classifierpasses through the top-left corner(0, 1): TPR = 1, FPR = 0.
- Arandom classifierlies on the diagonal from(0,0)to(1,1): it has no discriminative ability. - A classifier
below the diagonalis worse than random (flip all predictions to improve it).
Area Under the Curve (AUC): the AUC summarises classifier performance across all thresholds in a single
number:
AUC= Pr(ˆp(random positive)>ˆp(random negative))
That is, AUC is the probability that the model ranks a randomly chosen positive observation higher than a
randomly chosen negative one. AUC∈[0.5, 1], withAUC = 1indicating a perfect classifier andAUC = 0.5
indicating random guessing.
4.3.3 Cost-Optimal Classification Threshold
The default threshold of 0.5 is optimal only when the costs of a false positive and a false negative are equal. In
practice, these costs are often asymmetric — for example, missing a cancer diagnosis (FN) is typically far more
costly than an unnecessary follow-up test (FP).
Thecost-minimising thresholdis derived by comparing the expected cost of predicting each class. Predict
positive if:
ˆp(X)·cFN >(1−ˆp(X))·cFP
Solving forˆp(X)gives the optimal threshold:
t∗= cFP
cFP +c FN
wherec FP is the cost of a false positive andcFN is the cost of a false negative.
22

## Page 23

Cost-threshold intuition
•c FN≫cFP (missing a positive is very costly):t∗≪0.5— lower the threshold to catch more positives,
accepting more false alarms.
•c FP≫cFN (false alarms are very costly):t∗≫0.5— raise the threshold, only predict positive when
very confident.
•c FN =c FP:t ∗= 0.5— the symmetric default is optimal.
Total expected cost=FN×cFN +FP×cFP. To find the optimal threshold in practice, evaluate total
cost across a grid of thresholds on validation data.
4.4 Generative Models for Classification
An alternative to directly modellingPr(Y =k|X)(thediscriminativeapproach of logistic regression) is
thegenerativeapproach: model the distribution of X within each classPr(X|Y=k), then apply Bayes’
theorem to obtain the posterior:
Pr(Y=k|X=x) = πk·fk(x)
K∑
l=1
πl·fl(x)
where: -πk = Pr(Y =k)is thepriorprobability of class k (estimated as ˆπk =nk/n). -fk(x) = Pr(X =x|
Y=k)is theclass-conditional density(whatXlooks like within classk).
The different generative classifiers (LDA, QDA, Naive Bayes) differ in how they modelfk(x).
4.5 Linear Discriminant Analysis (LDA)
LDA assumes that within each class,X follows amultivariate Gaussian distributionwith class-specific
mean vectorsµk but ashared covariance matrixΣ(the same for all classes):
X|Y=k∼ N(µk,Σ)
4.5.1 LDA forp= 1(Single Predictor)
With one predictor, the class-conditional density is:
fk(x) = 1√
2πσexp
(
−(x−µk)2
2σ2
)
Plugging into Bayes’ theorem and taking logarithms, one can show that the posterior is maximised by assigning
xto the class with the largestlinear discriminant function:
δk(x) =x·µk
σ2−µ2
k
2σ2 + logπk
This islinear in x (hence “Linear” DA), meaning the decision boundary between any two classes is a single
point on the real line. For two classes with equal priors, the boundary is simply the midpoint(µ1 +µ2)/2.
Parameter estimatesfrom training data:
ˆπk = nk
n ,ˆµ k = 1
nk
∑
i:y i=k
xi,ˆσ 2 = 1
n−K
∑
k
∑
i:y i=k
(xi−ˆµk)2
4.5.2 LDA forp>1(Multiple Predictors)
Withppredictors, the multivariate discriminant function is:
δk(x) =x⊤Σ−1µk−1
2µ⊤
k Σ−1µk + logπk
23

## Page 24

This is again linear inx, so the decision boundaries between classes arehyperplanes(linear in the feature
space). The estimated parameters are:
ˆµk = 1
nk
∑
i:y i=k
xi, ˆΣ= 1
n−K
K∑
k=1
∑
i:y i=k
(xi−ˆµk)(xi−ˆµk)⊤
The key requirement is thatˆΣis invertible, which requiresn>p+K.
Why does a sharedΣlead to linear boundaries?
The log-posterior ratio between classkand classlis:
log Pr(Y=k|x)
Pr(Y=l|x) = logπk
πl
+x⊤Σ−1(µk−µl)−1
2(µ⊤
k Σ−1µk−µ⊤
l Σ−1µl)
Because Σ is the same for both classes, the quadratic terms inxcancel, leaving a linear function ofx.
Setting this equal to zero gives the linear decision boundary. IfΣk̸= Σl (as in QDA), the quadratic
terms do not cancel and the boundary becomes quadratic.
4.6 Quadratic Discriminant Analysis (QDA)
QDA relaxes the shared-covariance assumption and allows each class to have itsown covariance matrixΣk:
X|Y=k∼ N(µk,Σ k)
The discriminant function becomes:
δk(x) =−1
2 log|Σk|−1
2(x−µk)⊤Σ−1
k (x−µk) + logπk
This isquadratic inx, giving curved (quadratic) decision boundaries. QDA must estimate K separate
covariance matricesΣk, each of sizep×p— a total ofKp(p + 1)/2parameters. This makes QDA much more
flexible but also more data-hungry than LDA.
4.6.1 LDA vs. QDA: When to Use Which
Criterion LDA QDA
Covariance assumption Shared Σ across all classes Class-specificΣ k
Decision boundaryLinear (hyperplane) Quadratic (curved
surface)
Number of parametersKp+p(p+ 1)/2Kp+Kp(p+ 1)/2
Bias Higher (more constrained) Lower
VarianceLower Higher
Preferred whennis small relative top;
classes have similar spread
nis large; classes have
clearly different
covariances
True boundary is linearLDA is better or equal QDA may overfit
True boundary is curvedLDA may underfit QDA is better
The LDA–QDA choice is another instance of the bias-variance trade-off: LDA makes a stronger assumption
(sameΣ) that reduces variance but increases bias if wrong; QDA is more flexible but requires more data.
4.7 Naive Bayes
Naive Bayes is a generative classifier that makes an even stronger simplifying assumption:within each class,
the predictors are mutually independent:
24

## Page 25

fk(x) =
p∏
j=1
fkj(xj)
where fkj is the marginal density of thej-th predictor within classk. This drastically reduces the number of
parameters to estimate: instead of a fullp×pcovariance matrix, we need onlyp univariate densities per class.
How eachfkj is modelled: - IfXj is quantitative: assume Gaussian, estimateˆµkj and ˆσ2
kj from class-k
observations. - IfXj is categorical: estimate as the proportion of each category within classk.
The conditional independence assumption is almost certainly wrong in practice, but Naive Bayes
often works surprisingly well because: 1. Even a misspecified model can produce good class rankings. 2. When
p is large andn is small, the variance reduction from the independence assumption outweighs the bias it
introduces — the model generalises better despite being “wrong.”
Naive Bayes is particularly popular for text classification (wherep can be tens of thousands of words) precisely
because the independence assumption keeps the model tractable.
4.8 K-Nearest Neighbours Classifier (Revisited)
KNN classification (introduced in Chapter 2) estimates the posterior class probabilities non-parametrically
using theKclosest training observations:
Pr(Y=j|X=x 0) = 1
K
∑
i∈N0
1(yi =j),ˆy= arg max
j
Pr(Y=j|X=x 0)
KNN makes no distributional assumptions and can produce arbitrarily complex, non-linear decision boundaries.
However: -Standardiseall predictors before applying KNN (Euclidean distance is scale-dependent). - KNN
degrades in high dimensions (curse of dimensionality). - There is no interpretable model — just a look-up of
neighbours. - ChooseKby cross-validation.
4.9 Comparison of Classification Methods
Method Decision boundary Key assumption Best when
Logistic
regression
Linear Linear log-odds Binary/multinomial
Y; interpretability
needed; no strong
distributional
assumption
LDA Linear GaussianX|Y=k; shared
Σ
Smalln;K >2
classes; Gaussian
predictors with
similar spread
QDA Quadratic GaussianX|Y=k;
class-specificΣ k
Largen; classes
have clearly
different covariance
structures
Naive Bayes Depends onf kj Conditional independence
within each class
Very largep, small
n; text/document
classification
KNN Arbitrary non-linear Proximity inX-space implies
similarY
Highly non-linear
boundary; largen;
smallp
LDA vs. Logistic Regression: a subtle but important distinction
Both LDA and logistic regression producelineardecision boundaries, and their performance is often
similar in practice. The key difference is in how they estimate the boundary:
• Logistic regressionisdiscriminative: it directly models Pr(Y = 1|X)without any assumption on
Pr(X).
25

## Page 26

• LDAisgenerative: it models Pr(X|Y= k)as Gaussian and derives Pr(Y = k|X)via Bayes’
theorem.
When the Gaussian assumption holds, LDA uses the distributional information aboutX more efficiently
and can outperform logistic regression (especially whenn is small). When the assumption fails badly,
logistic regression is more robust. In practice, both should be tried.
Theory Questions – Chapter 4
Q1: Why can we not use linear regression for a classification problem with two classes,
even if we encode them as 0 and 1? What does logistic regression model instead, and what
function ensures predictions stay in[0,1]?
A:Linear regression is inappropriate for binary classification for two reasons. First, it can produce fitted
values outside[0, 1], which are not interpretable as class probabilities. Second, with more than two
classes, numerically encoding the classes (e.g.{0, 1, 2}for three classes) imposes an artificial ordering
and equal spacing that is meaningless for nominal categories. Logistic regression instead directly models
the probabilityp(X) = Pr(Y = 1 |X)by applying thesigmoid (logistic) functionto a linear
combination of the predictors:p(X) =eβ0+β1X/(1 +eβ0+β1X). The sigmoid has range(0, 1)for all real
inputs, guaranteeing that predicted probabilities are always valid. The model is linear on thelog-odds
(logit) scale:log(p/(1−p)) =β0 +β1X.
Q2: What is the AUC of the ROC curve, and what does an AUC of 0.5 vs. 0.9 tell you
about a classifier?
A:The AUC (Area Under the ROC Curve) is the probability that the model assigns a higher
predicted probability to a randomly chosen positive observation than to a randomly chosen negative
one: AUC = Pr(ˆp(positive)>ˆp(negative)). The ROC curve plots TPR (sensitivity) against FPR (1−
specificity) as the classification threshold varies from 1 to 0. An AUC of 0.5 means the classifier has
no discriminative ability whatsoever — it performs no better than randomly assigning classes (this
corresponds to the diagonal of the ROC plot). An AUC of 0.9 means that in 90% of cases where we draw
one random positive and one random negative observation, the model correctly ranks the positive higher.
This is a strong classifier. The AUC is threshold-independent and particularly useful for comparing
classifiers without committing to a specific decision threshold.
Q3: What is the fundamental difference between LDA and logistic regression, even though
both produce linear decision boundaries? When would you prefer each?
A:Both methods produce linear decision boundaries and are often competitive in performance, but
they differ fundamentally in their approach. Logistic regression isdiscriminative: it directly models
Pr(Y =k|X)without making any assumptions about the distribution ofX itself. LDA isgenerative:
it models the class-conditional distributionsPr(X|Y= k)as multivariate Gaussians with a shared
covariance matrix, then uses Bayes’ theorem to derivePr(Y = k|X). LDA is preferred when the
Gaussian assumption approximately holds andn is small relative top, because the additional structural
information allows more efficient estimation. LDA also naturally extends toK >2classes without any
modification. Logistic regression is preferred when the Gaussian assumption is clearly violated (e.g.
binary or skewed predictors), when interpretability of log-odds coefficients is important, or when the
primary goal is to model the class boundary directly without distributional assumptions.
26

## Page 27

5 Chapter 5 – Resampling Methods
Resampling methods are techniques that involve repeatedly drawing samples from training data — either
to estimate how well a model will perform on new data (cross-validation) or to quantify uncertainty in an
estimator (the bootstrap). They are among the most practically important tools in statistical learning, since
analytical formulas for test error or standard errors often do not exist for complex models.
5.1 Cross-Validation
The central challenge in model assessment is that we want to know thetest error— how well a model fitted
on the training data performs on new, unseen observations. Cross-validation provides a principled way to
estimate test error from training data alone, without needing a separate held-out test set.
5.1.1 The Validation Set Approach
The simplest approach: randomly split the available data into two halves — atraining setand avalidation
set(also called a hold-out set). Fit the model on the training set and evaluate its error on the validation set.
Validation MSE= 1
|V|
∑
i∈V
(yi−ˆf(xi))2
whereVis the set of validation observations and ˆfwas fitted on the complementary training observations.
Advantages: Simple to implement; computationally cheap (fit the model only once).
Disadvantages: -High variability: the estimated test error depends strongly on which observations happen
to fall in the training vs. validation set. Different random splits can give very different error estimates. -
Overestimates test error: the model is trained on only≈50%of the data. A model trained on more data
would perform better, so the validation error overestimates how poorly the model would perform if trained on
the full dataset.
5.1.2 Leave-One-Out Cross-Validation (LOOCV)
LOOCV addresses both limitations of the validation set approach by usingn−1observations for training and
holding out exactly one observation at a time, repeating this for every observation:
LOOCV Algorithm
1. For eachi= 1,2,...,n:
(a) Fit the model on all observationsexcepti: ˆf (−i)
(b) Predict observationi:ˆyi = ˆf (−i)(xi)
(c) Compute the squared error: MSEi = (yi−ˆyi)2
2. Average over allnhold-outs:
CV(n) = 1
n
n∑
i=1
MSEi
Advantages: -Low bias: the model is trained onn−1observations — almost the full dataset. The training
set closely approximates what would be available in practice. -Deterministic: no random splitting, so the
result is always the same (no variability from different random seeds).
Disadvantages: -Expensive: requires fitting the modeln times. For largen or complex models (e.g. neural
networks), this can be prohibitively slow. -High variance: then training sets overlap almost completely
(each differs from the others by only one observation), making then error estimates highly correlated. The
average of highly correlated estimates has higher variance than the average of independent estimates.
Computational shortcut for linear models: For OLS regression (and polynomial regression), LOOCV can
be computed at the cost of a single model fit using the hat matrix shortcut:
CV(n) = 1
n
n∑
i=1
(yi−ˆyi
1−hii
)2
where hii is thei-th diagonal entry of the hat matrixH=X(X⊤X)−1X⊤(theleverageof observation i).
This shortcut does not generalise to non-linear models.
27

## Page 28

5.1.3k-Fold Cross-Validation
k-fold CV is the practical workhorse of model selection. The data are randomly divided intok roughly
equal-sizedfolds(groups). For each foldj= 1,...,k:
k-Fold CV Algorithm
1. Randomly partition thenobservations intokfolds of approximately equal size
2. Forj= 1,2,...,k:
(a) Hold out foldjas the test set
(b) Train the model on all remainingk−1folds
(c) Compute the test error on foldj: MSEj
3. Average thektest errors:
CV(k) = 1
k
k∑
j=1
MSEj
LOOCV is the special case wherek=n.
Typical choices:k= 5ork= 10are the standard recommendations in ISLP and the broader literature.
5.1.4 Bias-Variance Trade-off fork-Fold CV
The choice ofkdirectly controls the bias-variance trade-off of the CV error estimate itself:
Method Training set size Bias of CV estimate Variance of CV estimate Computation
Validation
set
≈n/2High (trained on half
data)
High (sensitive to split)1fit
LOOCV
(k=n)
n−1Very low High (highly correlated
folds)
nfits
k= 10
fold
≈9n/10Low Low–moderate10fits
k= 5
fold
≈4n/5Low–moderate Low5fits
The advantage ofk = 5or k = 10over LOOCV is not just computational. Because thek training sets ink-fold
CV are more different from each other (each fold containsn/k unique observations), the fold-level errorsMSEj
are less correlated. The average of less correlated quantities has lower variance. Empirically,k= 10has been
shown to give test error estimates with low bias and low variance — a good balance.
Why does LOOCV have higher variance thank-fold despite training on more data?
It seems paradoxical that training on more data (LOOCV usesn−1, k-fold uses≈(k−1)n/k) leads to
higher variance in the error estimate. The key is that LOOCV producesn estimates MSE1,...,MSEn
that are almost perfectly correlated — each pair of training sets sharesn−2identical observations.
Highly correlated random variables have high variance when averaged. Withk-fold CV, each pair of
training sets shares only(k−2)n/k observations, so the fold estimates are less correlated, giving a
lower-variance average.
5.1.5 CV for Model Selection
Cross-validation is most commonly used toselect among competing models— choosing the polynomial
degree, the regularisation parameterλ, or the number of neighboursK:
1. For each candidate model (or hyperparameter value), compute CV(k).
2. Select the model with thelowest CV error(λmin).
3. Optionally apply theone-standard-error rule: select the simplest model whose CV error is within one
standard error of the minimum — often giving a sparser, more interpretable model with similar predictive
performance.
28

## Page 29

Do not use the same data for both model selection and error reporting
If you select the best model by CV and then report that CV error as the test error, you are using the
same data twice, which leads to optimistic (too low) error estimates. The correct approach is:
(1)Use CV on the training set for model selection.
(2)Report performance on a completely separate held-out test set (or use nested CV).
5.1.6 Cross-Validation for Classification
In classification problems,k-fold CV is applied identically, but MSE is replaced by themisclassification rate
on each fold:
CV(k) = 1
k
k∑
j=1
Errj,where Err j = 1
|Vj|
∑
i∈Vj
1(yi̸= ˆyi)
Other metrics (AUC, F1) can also be averaged across folds.
5.2 The Bootstrap
The bootstrap is a general-purpose resampling tool for estimating theuncertainty(standard error, confidence
interval) of virtually any statistic — including those for which no analytical formula exists.
5.2.1 Core Idea and Algorithm
The key insight is that the training dataset itself is our best approximation of the population. We can simulate
“drawing new samples from the population” by repeatedly drawingwith replacementfrom the training data.
Bootstrap Algorithm
1. ChooseB(typicallyB= 1000or more)
2. Forb= 1,2,...,B:
(a) Draw a bootstrap sampleZ∗bof sizenwith replacementfrom the original dataset
(b) Compute the statistic of interest onZ∗b: call it ˆθ∗b
3. Estimate the standard error:
ˆSEB(ˆθ) =
√ 1
B−1
B∑
b=1
(
ˆθ∗b−¯ˆθ∗
)2
,where ¯ˆθ∗= 1
B
B∑
b=1
ˆθ∗b
4. Compute a 95% confidence interval using thepercentile method:
95% CI=
[
2.5th percentile ofˆθ∗b,97.5th percentile of ˆθ∗b]
The bootstrap distribution ofˆθ∗baround ¯ˆθ∗approximates the true sampling distribution ofˆθaround θ. This
approximation improves asBincreases and asnincreases.
5.2.2 What Observations End Up in a Bootstrap Sample?
Since sampling is donewith replacement, some observations will appear multiple times in a bootstrap sample
and others will not appear at all. The probability that observationi isnotselected in a single draw is(1−1/n).
Overndraws:
Pr(obsi /∈Z∗b) =
(
1−1
n
)n
−−−−→
n→∞
e−1≈0.368
So approximately36.8% of observations are absentfrom each bootstrap sample, and63.2% are present
(some more than once). This fact will become important when we discuss Out-of-Bag error estimation for
Random Forests (Chapter 8).
29

## Page 30

A key implication of the≈36.8%absent rate
Because each bootstrap sample leaves out about one third of the data, we could in principle evaluate
model fit on the left-out observations — analogous to a test set. This is exactly the idea behind
Out-of-Bag (OOB) errorin bagging and random forests (Chapter 8). The OOB observations act as
a built-in validation set for each bootstrap-fitted model, at no additional computational cost.
5.2.3 The Portfolio Example: When Bootstrap is Essential
Consider investing a fractionαin assetX and(1 −α)in asset Y, with the goal of minimising the variance of
the total return. The variance-minimising allocation is:
ˆα= ˆσ2
Y−ˆσXY
ˆσ2
X + ˆσ2
Y−2ˆσXY
whereˆσ2
X,ˆσ2
Y andˆσXY are the sample variances and covariance.
The problem: there is no simple closed-form formula forSE(ˆα)— the formula involves complicated functions
of sample moments. The bootstrap solves this effortlessly:
1. DrawB= 1000bootstrap samples from the(X,Y)data.
2. Computeˆα∗bfor each bootstrap sample.
3. Report ˆSEB(ˆα)as the bootstrap standard error.
This works foranystatistic — the median, a correlation, the AUC of a classifier, a regularised regression
coefficient, or any quantity for which you want to quantify uncertainty.
5.3 Cross-Validation vs. Bootstrap: A Critical Distinction
These two methods are complementary tools that answer different questions:
CV vs. Bootstrap — what each estimates
Cross-Validationestimates how well amodelgeneralises to new data — it provides an estimate oftest
error. Use it to select among models or hyperparameters, or to report expected prediction performance.
Bootstrapestimates theuncertaintyof a statistic or estimator — it providesstandard errors and
confidence intervals. Use it whenever you want to quantify how precisely you have estimated a
quantity, especially when analytical formulas are unavailable.
They are not interchangeable: CV tells you aboutmodel performance; bootstrap tells you about
estimation uncertainty.
Feature Cross-Validation Bootstrap
Primary purposeEstimate test error / model selection Estimate SE and CI of a
statistic
Sampling schemeWithout replacement (disjoint folds) With replacement
What varies across
iterations
Which observations are held out Which observations are
included
OutputEstimated test MSE / error rate ˆSE(ˆθ), confidence interval
TypicalBork k= 5or10B= 1000+
Requires model
refitting
Yes,ktimes Yes,Btimes
Theory Questions – Chapter 5
Q1: Explain the bias-variance trade-off in the choice ofk in k-fold cross-validation. Why is
k= 5ork= 10typically recommended over LOOCV?
A:The choice ofk controls a bias-variance trade-off in the CV error estimate itself. With smallk (e.g.
k = 2), each model is trained on only half the data, so the training set is much smaller than the full
dataset. This means each fold’s model is weaker than the final model, and the CV error overestimates
30

## Page 31

the true test error (high bias). Withk = n (LOOCV), each model is trained onn−1observations,
giving nearly unbiased estimates of test error (low bias). However, then training sets in LOOCV
are nearly identical (each differs by one observation), making then fold-level error estimates highly
correlated. The average of highly correlated estimates has higher variance than the average of weakly
correlated estimates.k = 5or k = 10strikes a better balance: the training sets are different enough that
fold errors are only moderately correlated (lower variance than LOOCV), yet large enough that bias is
low. Empirically, these choices provide reliable test error estimates and are far cheaper than LOOCV for
non-linear models.
Q2: In the bootstrap, approximately what fraction of observations are absent from each
bootstrap sample? Why does this matter?
A:The probability that a specific observation is not selected in a bootstrap sample of sizen drawn
with replacement is(1−1/n)n, which converges toe−1≈0.368as n→∞. So approximately 36.8%
of observations are absent from each bootstrap sample and 63.2% are present (some appearing more
than once). This matters for two reasons. First, it means that roughly one third of the data is always
available as a "free" validation set for each bootstrap-fitted model — this is the basis of Out-of-Bag
(OOB) error estimation in random forests, which gives a nearly unbiased estimate of test error
without requiring cross-validation. Second, the duplicated observations in the bootstrap sample inflate
apparent agreement (since the same observation may appear in both training and test sets in naive
implementations), which is why bootstrap is not used as a direct substitute for CV in test error estimation.
Q3: A colleague proposes estimating the standard error of a regression coefficient by
running cross-validation and computing the standard deviation of the coefficient estimates
across folds. Is this a valid approach? What is the correct tool?
A:This is not a valid approach. Cross-validation is designed to estimatetest error(model prediction
performance), not to quantify the sampling variability of an estimator. The variation in a coefficient
estimate across CV folds reflects the sensitivity of the model to which observations are included, but this
is not the same as the standard error — the variation we would see if we repeatedly drew new datasets of
sizen from the true population. The correct tool is thebootstrap: drawB bootstrap samples from the
training data, fit the model on each, and compute the coefficient estimateˆβ∗b
j for each. The bootstrap
standard error ˆSEB( ˆβj)directly approximates the true sampling variability. For OLS specifically, the
analytical formulaSE( ˆβj) = ˆσ
√
[(X⊤X)−1]jj is available and more efficient, but bootstrap remains
useful for regularised estimators (Ridge, Lasso) where analytical SEs are not standard.
31

## Page 32

6 Chapter 6 – Linear Model Selection and Regularization
When the number of predictorsp is large relative to the number of observationsn, OLS suffers from two related
problems. First, OLS has high variance: with many parameters to estimate from limited data, the coefficients
are noisy and generalise poorly. Second, whenp≥n, OLS is entirely undefined becauseX⊤Xis singular.
Three broad strategies address these problems: (1)subset selection— choose a subset of thep predictors;
(2)shrinkage/regularisation— fit all predictors but shrink coefficients toward zero; and (3)dimension
reduction— project the predictors onto a lower-dimensional space.
6.1 Subset Selection
6.1.1 Best Subset Selection
Fit a separate OLS model for every possible subset of thep predictors — a total of2p models. For each subset
size k = 0, 1,...,p, identify the best modelMk (the one with the lowest RSS among all
(p
k
)
models of that
size). Then choose the final model fromM0,M 1,...,Mp using an appropriate criterion (CV, AIC, BIC, or
adjustedR 2).
Limitation: Computationally infeasible forp >40. With p = 30, there are2 30≈109 models to fit. This
motivates stepwise alternatives.
6.1.2 Forward Stepwise Selection
Forward stepwise builds up the model from nothing, adding one predictor at a time:
Forward Stepwise Algorithm
1. Begin withM 0: the null model containing only an intercept
2. Fork= 0,1,...,p−1:
(a) Consider allp−kmodels that augmentMk with exactly one additional predictor
(b) Choose the one that gives the greatest improvement in RSS (or equivalently,R2): call itMk+1
3. Select the best overall model fromM0,M 1,...,Mp using CV or an information criterion
Total models considered:1 +p(p+ 1)/2(vs.2 p for best subset). Can be used even whenp>n.
Key limitation: Forward stepwise is greedy — it adds the best predictor at each step but does not revisit
earlier choices. A predictor added early may become redundant once others are included, but it cannot be
removed. Therefore, forward stepwise is not guaranteed to find the globally optimal model of any given size.
6.1.3 Backward Stepwise Selection
Backward stepwise starts with the full model and removes predictors one at a time:
Backward Stepwise Algorithm
1. Begin withMp: the full model containing allppredictors
2. Fork=p,p−1,...,1:
(a) Consider allkmodels that drop exactly one predictor fromMk
(b) Choose the one with the smallest decrease in RSS (i.e. drop the least useful predictor): call it
Mk−1
3. Select the best overall model fromM0,M 1,...,Mp using CV or an information criterion
Requirement:n>p(the full model must be fittable). SameO(p 2)cost as forward stepwise.
Forward and backward stepwise produce different sequences of models and may select different final models.
Neither is guaranteed to match best subset selection, but both are computationally tractable for largep.
6.2 Choosing the Optimal Model Size
Once we have a sequence of modelsM0,M 1,...,Mp, we must choose which one to use. We cannot use
training RSS orR2 directly, because these always improve as we add predictors — even irrelevant ones. Two
approaches exist:information criteria(closed-form adjustments to training error) andcross-validation
(direct estimation of test error).
32

## Page 33

6.2.1 Information Criteria
Information criteria adjust training RSS upward by a penalty that grows with model complexity, approximating
the expected test error:
Criterion Formula Penalty Choose model with
Cp (Mallows) 1
n(RSS+ 2dˆσ2) 2dˆσ 2 SmallestC p
AIC−2ℓ+ 2d2dSmallest AIC
BIC 1
n(RSS+ log(n)·d·
ˆσ2)
log(n)·d·ˆσ2 Smallest BIC
AdjustedR 2 1−
RSS/(n−d−1)
TSS/(n−1)
Implicit via
n−d−1
Largest adj.R 2
wheredis the number of predictors andˆσ2 is the estimated error variance from the full model.
AIC vs. BIC: Both penalise complexity, but BIC useslog(n)instead of2as the penalty multiplier. For
n≥8, log(n)> 2, so BIC penalises additional parameters more heavily than AIC and tends to selectsparser
models. For largen, BIC is consistent (selects the true model if it is among the candidates) while AIC tends
to overfit slightly. In practice, BIC often gives more parsimonious models.
Cp and AIC are equivalent for OLS with Gaussian errors
Under Gaussian errors, maximising the log-likelihoodℓis equivalent to minimising RSS (sinceℓ=
−n/2·log(RSS/n) +const). Therefore AIC=−2ℓ+ 2d∝RSS/ˆσ2 + 2d∝Cp. The two criteria select
the same model when used with OLS. They diverge for non-Gaussian models (logistic regression, etc.),
where AIC is defined via the likelihood butCp is not.
6.2.2 Cross-Validation for Model Selection
An alternative to information criteria is to directly estimate the test error of each modelMk using k-fold CV,
then select the model with the lowest estimated test error. This approach:
•Makes no assumptions about the error distribution or the form of the model.
•Works for any type of model (not just OLS).
• Is slightly more expensive but often more reliable than information criteria when their assumptions are
not met.
Theone-standard-error ruleis commonly applied: rather than picking the model with the absolute minimum
CV error, choose thesimplest modelwhose CV error is within one standard error of the minimum. This
guards against overfitting to the variability in the CV estimate itself.
6.3 Shrinkage Methods: Ridge and Lasso
Shrinkage methods fit a model containing allp predictors but regularise the coefficient estimates by adding a
penalty termto the loss function. The penalty shrinks coefficients toward zero, trading a small increase in
bias for a potentially large reduction in variance.
6.3.1 Ridge Regression (L2 Regularisation)
Ridge regression minimises:
n∑
i=1

yi−β0−
p∑
j=1
βjxij


2
+λ
p∑
j=1
β2
j =RSS+λ∥β∥2
2
The second term is theL2 penalty— the sum of squared coefficients. The tuning parameterλ≥0controls
the strength of regularisation:
•λ= 0: no penalty⇒OLS solution.
•λ→∞: allˆβj→0(exceptˆβ0 = ¯y).
•Intermediateλ: coefficients are shrunk toward zero but not set exactly to zero.
33

## Page 34

Closed-form solution: Ridge has a closed-form solution (unlike Lasso):
ˆβRidge = (X⊤X+λI)−1X⊤y
The addition ofλImakes(X ⊤X+ λI)invertible even whenX⊤Xis not — this is why Ridge works even when
p≥n. It also directly addresses multicollinearity: when two predictors are nearly collinear, OLS assigns very
large and unstable coefficients of opposite sign; Ridge shrinks both toward zero, stabilising the estimates.
Note: The interceptβ0 is never penalised (it does not appear in the penalty sum). This is because penalising
β0 would make the predictions depend on the arbitrary choice of origin forY. The intercept is estimated as
ˆβ0 = ¯yafter centering.
Always standardise predictors before Ridge or Lasso
The L2 and L1 penalties are not scale-invariant: a predictor measured in kilometres has smaller coefficients
than the same predictor measured in metres, and Ridge would shrink the latter more. Before applying
Ridge or Lasso,standardise all predictorsto have mean zero and unit variance:
˜xij = xij−¯xj
ˆσxj
This ensures the penalty treats all predictors equally regardless of their measurement scale. The intercept
is estimated separately and should not be standardised.
The bias-variance trade-off in Ridge: Asλincreases, Ridge coefficients shrink, which: -Reduces variance:
the model is less sensitive to the specific training data. -Increases bias: the model is constrained away from
the true OLS solution.
The optimalλminimises test MSE, which has the usual U-shape: too smallλgives high-variance OLS; too
largeλgives an overly shrunk model with high bias. Chooseλby cross-validation.
6.3.2 The Lasso (L1 Regularisation)
The Lasso (Least Absolute Shrinkage and Selection Operator) replaces the L2 penalty with anL1 penalty—
the sum of absolute values of coefficients:
RSS+λ
p∑
j=1
|βj|=RSS+λ∥β∥1
The critical difference from Ridge: The L1 penalty createscorner solutions— the Lasso can set some
coefficientsexactly to zero. This performsautomatic variable selection: at sufficiently largeλ, only a
subset of predictors remains in the model. Ridge can only shrink coefficients toward zero, never to zero exactly.
Why corners arise: In the equivalent constrained formulation, Ridge constrains the solution to a sphere
(∑β2
j≤s) and Lasso constrains it to a diamond (∑|βj|≤s). The diamond has sharp corners on the coordinate
axes. The OLS contours (ellipses) are likely to touch the diamond at a corner — where one or moreβj = 0
exactly — but much less likely to touch the sphere at a point on an axis.
No closed form: Unlike Ridge, Lasso has no closed-form solution because the L1 penalty is not differentiable
at zero. It is solved numerically usingcoordinate descentor the LARS algorithm.
6.3.3 Ridge vs. Lasso: A Detailed Comparison
Property Ridge (L2) Lasso (L1)
Penalty termλ ∑
jβ2
j λ∑
j|βj|
Sets coefficients to exactly
zero?
No — only shrinks toward zero Yes — produces sparse solutions
Variable selection?No — all predictors remain Yes — built-in selection
Closed-form solution?Yes:(X ⊤X+λI)−1X⊤yNo — coordinate descent
With correlated predictorsShrinks both together (similar
coefficients)
Tends to pick one and zero out the
rest
34

## Page 35

Property Ridge (L2) Lasso (L1)
Best whenMany predictors all have moderate
effects
Few predictors have large effects;
sparse truth
Works whenp>n?Yes Yes (selects at mostnpredictors)
InterpretabilityAll predictors present; less
interpretable
Sparse model; easier to interpret
Geometric intuition for why Lasso produces zeros but Ridge does not
Both Ridge and Lasso can be viewed as constrained minimisation problems:
•Ridge: minimise RSS subject to ∑
jβ2
j≤s(a sphere / ball inRp)
•Lasso: minimise RSS subject to ∑
j|βj|≤s(a diamond / polytope inRp)
The unconstrained OLS minimum lies outside the constraint region (otherwise no shrinkage would occur).
The constrained solution is where the boundary of the constraint region first touches an elliptical contour
of the RSS. The sphere has no corners — the touching point is almost surely an interior boundary point
where noβj = 0. The diamondhascorners on the coordinate axes (whereβj = 0for one or morej).
The elliptical RSS contours are likely to first touch the diamond at a corner, giving a sparse solution.
6.3.4 Selectingλby Cross-Validation
λis atuning parameterand must not be estimated on the training data (that would re-introduce the
overfitting problem we are trying to solve). The standard procedure:
Selectingλby CV
1. Define a grid ofλvalues, e.g.λ∈{10−4,10−3,...,10 4}(logarithmically spaced)
2. For eachλin the grid, compute thek-fold CV error (using only training data)
3. Selectλmin: the value with the lowest CV error
4. Optionally apply theone-standard-error rule: selectλ1se, the largestλwhose CV error is within
one SE of the minimum. This gives a sparser (simpler) model with comparable performance
5. Refit the final model on thefull training setusing the selectedλ
Thecoefficient path(plot of ˆβj(λ)vs. logλ) is a useful visualisation. For Ridge, all coefficients shrink
gradually to zero asλincreases. For Lasso, some coefficients reach zero at certainλvalues (kinks in the path)
and remain zero for largerλ.
6.4 Dimension Reduction: Principal Components Regression (PCR)
Rather than selecting or shrinking predictors,dimension reductionmethods transform thep predictors into
M <pnew variables (linear combinations) and regressYon theseMderived features.
Principal Components Regression (PCR)proceeds in two steps:
1. Compute the firstM principal componentsZ1,...,ZM ofX(see Chapter 10 for full details on PCA).
EachZm = ∑p
j=1ϕjmXj is a linear combination of the original predictors, chosen to explain the maximum
variance inX.
2. RegressYonZ 1,...,ZM using OLS.
Key assumption: the directions of largest variation inXare also the directions most associated withY.
When this holds, usingM principal components instead of allp predictors reduces variance (fewer effective
parameters) at little cost in bias.
ChoosingM: via cross-validation.M=precovers OLS;M= 1gives the most reduced model.
PCR vs. Ridge: Ridge shrinks the coefficients of allp directions inXbut more aggressively shrinks the
directions of small variance. PCR discards the directions of small variance entirely and keeps the others
unchanged. Both reduce effective dimensionality, but in different ways.
PCR is not the same as PCA
PCA(Principal Component Analysis) is anunsupervisedmethod: it finds the directions of maximum
variance inXwithout any reference toY.
35

## Page 36

PCRuses these unsupervised components as predictors in a supervised regression. The components are
chosen to explainX, not to predictY — this is a weakness. A direction that accounts for little variance
inXmight still be strongly associated withY, and PCR would discard it.
Partial Least Squares (PLS)addresses this by finding components that maximise both the variance
inXand the correlation withY, but it is not covered in detail in this course.
Theory Questions – Chapter 6
Q1: Why does RSS always decrease (or stay flat) as we add more predictors to a linear
model? Why can we not use RSS orR2 directly to select model size, and what do we use
instead?
A:When we add a predictor to a model, OLS can always set its coefficient to zero if it provides
no value — in which case RSS stays exactly the same. If the predictor has any correlation with
the response (even by chance in the training data), OLS will find a non-zero coefficient that
reduces RSS. Therefore, RSS is non-increasing as predictors are added andR2 = 1−RSS/TSSis
non-decreasing. Both measures always favour the largest model, even if the added predictors are
pure noise. To select model size, we need criteria that balance fit against complexity. The four
standard approaches are: (1) Cp, which adds a penalty of2dˆσ2 to RSS; (2) AIC, proportional to
Cp for Gaussian OLS; (3) BIC, with a heavier log(n)·dˆσ2 penalty that tends to select sparser
models; and (4) adjustedR2, which penalises additional predictors through the degrees-of-freedom
adjustment. Alternatively,k-fold CV directly estimates test error without any distributional assumptions.
Q2: Explain why Ridge regression can never set a coefficient exactly to zero, while Lasso
can. What practical implication does this have?
A:The key is the geometry of the constraint region. Ridge constrains the solution to lie within an L2
ball (sphere): ∑
jβ2
j≤s. A sphere has no corners — its surface is perfectly smooth everywhere. The
OLS contours (ellipses in 2D) will almost always touch the sphere at an interior boundary point, where
no βj equals zero. The Lasso constrains the solution to an L1 ball (diamond):∑
j|βj|≤s. A diamond
has sharp corners located exactly on the coordinate axes, where one or moreβj = 0. The OLS contours
are much more likely to first contact the diamond at one of these corners than at an interior boundary
point, producing sparse solutions with exactly-zero coefficients. The practical implication is that Lasso
performs automatic variable selection: at a givenλ, only a subset of predictors has non-zero coefficients,
giving a model that is easier to interpret and potentially more parsimonious. Ridge keeps all predictors
in the model at anyλ>0.
Q3: You fit a Ridge regression and a Lasso regression to the same dataset withλchosen
by 10-fold CV. The Ridge model uses all 50 predictors; the Lasso selects 8. The test MSEs
are similar. Which model do you prefer, and why?
A:If the test MSEs are comparable, the Lasso model with 8 predictors is strongly preferred in most
contexts. First,interpretability: a model with 8 predictors is far easier to understand and communicate
— stakeholders can reason about which variables drive the outcome. Second,parsimony: the Lasso
result suggests that the true signal may be concentrated in a small subset of predictors and the remaining
42 are largely noise. A simpler model is also less likely to overfit on future datasets with slightly different
characteristics. Third,variable selection: in many applications (biology, economics, policy), identifying
which predictors are relevant is itself valuable. The only scenario where Ridge might be preferred is if
we believe all 50 predictors genuinely contribute (e.g. genomic data where many genes have small but
real effects), in which case Lasso’s tendency to arbitrarily pick one correlated predictor and drop others
becomes a disadvantage.
36

## Page 37

7 Chapter 7 – Moving Beyond Linearity
The linear model is a powerful baseline, but it rests on the assumption that the true relationship between
each predictor and the response is linear. When this assumption fails badly, predictions can be systematically
wrong across entire regions of the input space. Chapter 7 introduces a family of methods that relax linearity
while preserving the OLS fitting framework: polynomial regression, step functions, splines, local regression, and
generalised additive models. A unifying idea connects almost all of them: thebasis functionframework.
7.1 Polynomial Regression
The simplest extension of linear regression replaces the single linear termβ1Xwith a polynomial of degreed:
yi =β0 +β1xi +β2x2
i +···+βdxd
i +εi
This is still alinear model— linear in the parametersβ0,...,βd — so the OLS estimatorˆβ= (X⊤X)−1X⊤y
applies directly, with a design matrix whose columns are[1,x,x2,...,xd].
Degree selection: choosed via cross-validation. In practice,d≤4is almost always sufficient: higher-degree
polynomials become oscillatory and numerically unstable, especially near the boundaries of the data range
(Runge’s phenomenon).
Limitations of polynomial regression:
• A single polynomial is aglobalfunction — a change in the fit at one point forces changes everywhere.
There is no mechanism to fit data flexibly in one region while remaining smooth in another.
• Polynomials are especially unreliable at theboundaries(below the smallestx and above the largestx),
where they can diverge wildly.
•For these reasons, regression splines are generally preferred over high-degree polynomials.
7.2 Step Functions
Step functions partition the range ofX intoK + 1disjoint intervals usingK cutpoints c1 <c 2 <···<cK
and fit a separate constant within each interval. We createKindicator variables:
C0(X) =1(X <c 1), C 1(X) =1(c 1≤X <c2), ..., C K(X) =1(X≥cK)
SinceC 0 +C 1 +···+CK = 1always, we dropC 0 (to avoid the dummy variable trap) and fit:
yi =β0 +β1C1(xi) +β2C2(xi) +···+βKCK(xi) +εi
Here ˆβ0 is the mean response in the baseline binC0, and ˆβk is the mean difference relative to the baseline in
binC k.
Strength: the fit is constant within each bin — very interpretable (e.g. “average wage for age 20–30 vs. age
30–40”).
Weakness: step functions produce discontinuous fits — artificial jumps at the cutpoints. They also require
the analyst to choose cutpoint locations, which can have a large influence on results. If the true relationship is
smooth, step functions force a poor approximation.
7.3 Basis Functions: A Unifying Framework
Both polynomial regression and step functions are special cases of thebasis functionframework. We define
Kknown functionsb 1(x),b 2(x),...,bK(x)(the “basis”) and fit:
yi =β0 +
K∑
k=1
βkbk(xi) +εi
Thisisstillalinearmodelintheparameters β0,...,βK, soOLSapplieswithdesignmatrix[1 ,b 1(x),b 2(x),...,bK(x)].
The choice of basis functions determines the shape of the fit:
•b k(x) =x k: polynomial regression of degreeK
37

## Page 38

•b k(x) =1(c k−1≤x<ck): step function
•b k(x) = (x−ξk)3
+: truncated power basis for cubic splines (see below)
The power of this framework is that it extends immediately to multiple predictors, classification (via GLMs),
and other response types — always by choosing appropriate basis functions and fitting by OLS or MLE.
7.4 Regression Splines
Regression splines combine the local flexibility of step functions with the smoothness of polynomials by fitting
piecewise polynomials subject to smoothness constraints at the join points (knots).
7.4.1 Piecewise Polynomials
Apiecewise cubic polynomialwith one knot atcfits two separate cubic polynomials:
ˆf(x) =
{
β01 +β11x+β21x2 +β31x3 ifx<c
β02 +β12x+β22x2 +β32x3 ifx≥c
Without constraints, this has 8 free parameters. Each additional knot adds 4 parameters (one new cubic on the
right side of the knot).
Problem: unconstrained piecewise polynomials can be discontinuous and have sharp bends at the knots.
Adding continuity constraints makes the fit smoother.
7.4.2 Cubic Splines
Acubic splinewith K knots atξ1 <ξ2 <···<ξK is a piecewise cubic polynomial that is constrained to be
continuous and have continuous first and second derivatives at every knot. These3K constraints reduce the
effective degrees of freedom from4(K+ 1)(unconstrained) toK+ 4:
Degrees of freedom=K+ 4
This means a cubic spline withK knots usesK + 4basis functions. A convenient basis is thetruncated
power basis: start with the cubic polynomial basis{1,x,x 2,x 3}(4 functions) and add one truncated power
function per knot:
(x−ξk)3
+ =
{
(x−ξk)3 ifx>ξk
0otherwise
Adding(x−ξk)3
+ to the model introduces a cubic term that “turns on” to the right of knotξk and is zero to
the left. This exactly enforces continuity off, f′, andf′′at ξk while allowing the cubic polynomial to change
shape there.
WhyK+ 4degrees of freedom for a cubic spline?
Start withK+ 1separate cubic polynomials:(K+ 1)×4 = 4K+ 4parameters.
At each of theKknots, enforce 3 constraints (continuity off,f′,f ′′): subtract3Kconstraints.
Net degrees of freedom:(4K+ 4)−3K=K+ 4.
Each additional knot adds exactly 1 effective degree of freedom (not 4), because 3 of the 4 new parameters
are consumed by the smoothness constraints.
7.4.3 Natural Splines
Anatural cubic spline(also called a natural spline) is a cubic spline with the additional constraint that the
function islinear(not cubic) in the two boundary regions:x<ξ1 and x>ξK. This removes 2 parameters
from each end (thex2 andx 3 terms), so:
Degrees of freedom for natural spline=K
38

## Page 39

The boundary linearity constraint reduces the degrees of freedom fromK + 4to K and provides much more
stable, less erratic estimatesnear the edges of the data — the region where polynomials tend to blow up.
In Python’spatsy, natural splines are created withcr(x, df=K)(cubic regression splines).
Natural splines vs. cubic splines: Natural splines trade a slight increase in bias (forced linearity at the
boundaries) for a substantial reduction in variance at the tails. Since we typically have less data near the
boundaries, this trade-off is almost always beneficial.
7.4.4 Choosing the Number and Location of Knots
Two practical strategies:
1. Place knots at quantiles ofX: if we wantK knots, place them at the100/(K + 1)th,200/(K + 1)th,
...,100 K/(K + 1)th percentiles of the observedx values. This puts more knots where data are dense
and fewer where data are sparse, adapting the spline’s flexibility to the data distribution.
2. Choose K by cross-validation: compute the CV error for a range of knot counts (K = 1, 2,...) and
select the value that minimises test error. The optimalKbalances fit and flexibility.
Splines vs. high-degree polynomials: A cubic spline withK knots (df= K + 4) typically outperforms
a polynomial of degreeK + 4because splines arelocally flexible(the fit can change shape near each knot
without affecting distant regions), while polynomials impose a single global shape. Splines are also more
numerically stable.
7.5 Smoothing Splines
Smoothing splines take a different approach: rather than specifying knots in advance, find the functiong(x)
that minimises thepenalised RSS:
Minimise over all smoothg:
n∑
i=1
(
yi−g(xi)
)2
+λ
∫ ∞
−∞
[
g′′(t)
]2
dt
•The first term is the usual RSS: penalises lack of fit.
• The second term is theroughness penalty:g′′(t)is the second derivative (curvature) ofg att. Integrating
the squared curvature over the entire domain penalises functions that wiggle a lot.λ≥0controls the
trade-off.
Behaviour asλvaries:
•λ= 0: no penalty —g can be any function that interpolates alln data points exactly (g(xi) =yi for all
i). Training RSS = 0, but the function may be extremely wiggly and overfit badly.
•λ→∞: the roughness penalty dominates —g′′(t) = 0everywhere, so g must be linear (a straight line,
i.e. OLS).
•Intermediateλ: a smooth curve that trades off fit and roughness.
Key theoretical result: The functiong that minimises the penalised criterion above is always anatural
cubic splinewith knots at every uniquexvalue. The penaltyλeffectively reduces the number of degrees of
freedom the spline can use — even though there aren knots, the effective degrees of freedomdfλ∈[2,n ](2 for
a straight line,nfor full interpolation).
Choosing λ: byLOOCV. A computational shortcut (analogous to the LOOCV shortcut for OLS from
Chapter 5) makes this nearly as cheap as fitting the model once. Instatsmodels, smoothing splines are
accessed vialowess(approximate) orUnivariateSpline.
The effective degrees of freedom dfλ
The smoothing spline produces fitted valuesˆg =S λy, whereSλis then×nsmoother matrix (analogous
to the hat matrixHin OLS). The effective degrees of freedom are defined as:
dfλ=tr(S λ)
Whenλ→0:Sλ→Iand dfλ→n(interpolation, full flexibility).
Whenλ→∞:Sλ→HOLS for a linear fit and dfλ→2(a straight line has 2 parameters).
Practitioners often specify dfλdirectly (e.g. dfλ= 10) and let the software find the correspondingλ.
39

## Page 40

7.6 Local Regression
Local regression (also known asLOESSorLOWESS— Locally Weighted Scatterplot Smoothing) is a
non-parametric smoother that fits a separate weighted regression at each prediction pointx0, using only the
observations nearest tox0.
Local Regression Algorithm atx0
1. Identify thek training observations closest tox0 (by distance|xi−x0|). The fractions = k/n is
called thespan.
2. Assign observation-specific weightswi =K(xi,x 0)based on distance: observations close tox0 receive
weight near 1; distant ones receive near 0. Thetricube kernelis a common choice:
wi =
(
1−
( |xi−x0|
maxj|xj−x0|
)3)3
3. Fit aweighted least squaresregression ofy on x using only thek local observations with weights
wi. This is equivalent to minimising∑
i∈N0wi(yi−β0−β1xi)2.
4. ˆf(x 0) = ˆβ0 + ˆβ1x0 is the fitted value atx0 from this local regression.
5. Repeat for everyx0 in the grid of prediction points.
The spanscontrols the bias-variance trade-off:
• Small s (e.g. s = 0.2): very local fit — only 20% of observations used at each point. Low bias (the local
model closely follows the data), high variance (sensitive to individual observations), wiggly curve.
• Large s (e.g. s = 0.8): more global fit — 80% of observations used. High bias (forced to smooth over
local features), low variance, smooth curve.
Choosesby cross-validation (LOOCV is common for local regression).
Local regression vs. KNN regression(Chapter 3): KNN regression also uses neighbours, but fits a simple
mean. Local regression fits a weighted polynomial in each neighbourhood, which reducesboundary bias—
the tendency of KNN to under- or over-shoot near the edges of the data range because neighbours are only
available on one side.
Local regression in multiple dimensions (p> 1): LOWESS generalises top> 1predictors by computing
multivariate distances. However, it suffers from the curse of dimensionality forp> 3or4: neighbourhoods
must be large to contain enough observations, destroying the “local” advantage. GAMs are preferred in high
dimensions.
7.7 Generalised Additive Models (GAMs)
GAMs extend multiple linear regression by replacing each linear termβjXj with asmooth function fj(Xj),
while maintainingadditivity:
Y=β0 +f 1(X1) +f 2(X2) +···+fp(Xp) +ε
Eachfj can be independently specified as a spline, smoothing spline, polynomial, local regression, or even a
plain linear termβjXj. The model is additive — the total effect of the predictors is the sum of individual
effects — but each effect can be arbitrarily non-linear.
7.7.1 Fitting GAMs: Backfitting
GAMs for regression are fitted using thebackfitting algorithm, which iterates over predictors one at a time:
Backfitting Algorithm
1. Initialise: ˆfj≡0for allj;ˆβ0 = ¯y
2. Cycle throughj= 1,2,...,p(repeat until convergence):
(a) Compute thepartial residuals: ri =yi−ˆβ0−∑
k̸=j
ˆfk(xik)(the residuals after removing the
contributions of all other functions)
(b) Fit a smoother (e.g. smoothing spline, LOWESS) ofri onx ij to update ˆfj
40

## Page 41

3. Repeat until the functionsˆfj stop changing (convergence is guaranteed for additive models)
7.7.2 GAMs for Classification (Logistic GAMs)
For a binary response, replace the linear predictor in logistic regression with additive smooth functions:
log
( p(X)
1−p(X)
)
=β0 +f 1(X1) +f 2(X2) +···+fp(Xp)
Fitted using local scoring (iterated IRLS), available in Python viapygam.LogisticGAM.
7.7.3 Advantages and Disadvantages of GAMs
Aspect Details
Flexibility Eachfj can capture non-linear effects; more flexible than
linear regression
Interpretability Additive structure: plot eachfj separately to understand
its effect onY, holding others fixed
Mixed termsCan freely mix smooth termsf j(Xj)and linear terms
βjXj
Automatic smoothnessλ j controls smoothness of eachfj, chosen by CV or GCV
Limitation — interactionsThe additivity assumption means GAMs cannot model
interactions between predictors. The effect ofX1 onY
cannot depend onX2. For interactions, add explicit
interaction terms or use tree-based methods
Limitation — high-order interactionsWith many predictors and complex interactions, trees
and boosting often outperform GAMs
When to prefer each non-linear method
• Polynomial regression: simple, interpretable, works well when the non-linearity is mild and global.
Avoid degree>4.
• Regression splines / Natural splines: flexible and smooth; more stable than high-degree
polynomials; good when the non-linearity varies across the range ofX. Preferred over polynomial
regression in most cases.
• Smoothing splines: fully data-driven; no knot placement decision;λcontrols smoothness. Use
when you want maximum flexibility with automatic regularisation.
•Local regression (LOWESS): powerful for exploratory analysis and visualising trends; robust to
outliers with appropriate kernel. Best forp= 1orp= 2.
• GAMs: the natural extension to multiple predictors; interpretable; ideal when each predictor has an
independent non-linear effect but interactions are not the primary concern.
• Tree-based methods (Chapter 8): preferred when strong interactions among predictors are
expected, since trees automatically capture interactions through splits on multiple variables.
Theory Questions – Chapter 7
Q1: What is a natural cubic spline, and why is it preferred over a standard cubic spline at
the boundaries of the data?
A:A natural cubic spline is a cubic spline with the additional constraint that the function is linear
(i.e. the second and third derivative terms are zero) in the two boundary regions: below the smallest
knot and above the largest knot. A standard cubic spline withK knots hasK + 4degrees of freedom;
imposing linearity at each boundary removes 2 parameters at each end, reducing the degrees of freedom
to K. The motivation for this constraint is that polynomials behave erratically near the boundaries of
the data — there are few observations to anchor the fit, so cubic terms tend to diverge. By forcing
linearity in the tails, natural splines produce more stable, better-behaved estimates in the boundary
regions, at the cost of slightly increased bias there. Since the boundaries are precisely where data are
sparse, this trade-off almost always benefits generalisation.
41

## Page 42

Q2: A smoothing spline is fitted withλ= 0and separately with λ→∞. Describe what
each fitted function looks like and what this implies about the role ofλ.
A:The smoothing spline minimises ∑
i(yi−g(xi))2 +λ
∫
[g′′(t)]2dt. Whenλ= 0there is no penalty
on roughness, so the minimiser simply interpolates all data points:g(xi) =yi for alli. The resulting
function passes through every observation exactly (training RSS = 0) and may be extremely wiggly and
overfit. Whenλ→∞the roughness penalty dominates, forcingg′′(t) = 0everywhere — which meansg
must be a linear function (a straight line). This is equivalent to OLS withp = 1. For intermediateλ,
the smoothing spline produces a smooth curve that balances fit (small RSS) against roughness (small∫
[g′′]2). The parameter λthus controls the effective degrees of freedom: fromn (full interpolation,
λ= 0) down to 2 (straight line,λ→∞). In practice,λis chosen by LOOCV.
Q3: Explain how GAMs extend multiple linear regression to non-linear relationships.
What is the key limitation of GAMs compared to tree-based methods?
A:A GAM replaces the linear terms βjXj in multiple regression with arbitrary smooth functions
fj(Xj), givingY =β0 + ∑
jfj(Xj) +ε. Eachfj can be a spline, LOWESS, polynomial, or linear term,
independently chosen for each predictor. The model is fitted by backfitting: iterating over predictors
and fitting a univariate smoother to the partial residuals (residuals after removing all otherfk). This
preserves additive structure while allowing each predictor to have a non-linear, data-driven relationship
with Y. The key limitation is additivity: in a GAM, the effect ofXj on Y is the same regardless of
the values ofXk for k̸= j — interactions between predictors cannot be captured. For example, if
the effect of age on income depends strongly on education level, a GAM would miss this interaction.
Tree-based methods (random forests, gradient boosting) naturally capture such interactions through
splits on multiple variables in sequence, at the cost of lower interpretability than the additive structure
of GAMs.
42

## Page 43

8 Chapter 8 – Tree-Based Methods
Tree-based methods are among the most widely used methods in applied machine learning. They partition
the predictor space into rectangular regions and make predictions using the mean (regression) or majority
class (classification) within each region. A single decision tree is simple and interpretable, but tends to
have high variance and moderate accuracy. Ensemble methods — bagging, random forests, and gradient
boosting — address this by combining many trees, dramatically improving predictive performance at the cost
of interpretability.
8.1 Regression Trees
8.1.1 Building a Regression Tree: Recursive Binary Splitting
A regression tree partitions thep-dimensional predictor space intoJ non-overlapping rectangular regions
R1,R 2,...,RJ. Within each regionRm, the prediction is the mean of the training responses that fall in that
region:
ˆyRm = 1
|Rm|
∑
i:x i∈Rm
yi
The goal is to find regions that minimise the total RSS:
J∑
m=1
∑
i:x i∈Rm
(yi−ˆyRm)2
Finding the globally optimal partition is computationally infeasible, so we userecursive binary splitting—
a top-down, greedy algorithm:
Recursive Binary Splitting
1. Start with allnobservations in a single region (the root node)
2. At each step, consider all predictorsj = 1,...,p and all possible split pointss for each predictor. For
predictorjand split points, define two half-planes:
R1(j,s) ={x|xj <s}andR 2(j,s) ={x|xj≥s}
Choose the(j,s)pair that minimises:
∑
i:x i∈R1(j,s)
(yi−ˆyR1)2 +
∑
i:x i∈R2(j,s)
(yi−ˆyR2)2
3. Split the current region using the chosen(j,s), creating two child nodes
4. Repeat steps 2–3 within each child node until a stopping criterion is met (e.g. fewer than 5 observations
per leaf)
Key properties: greedy (optimal at each step, not globally); top-down (splits are not reconsidered);
produces axis-aligned rectangular regions.
Why is the algorithm greedy?At each step, recursive binary splitting chooses the single best split without
considering how that split affects future splits. A split that is locally optimal may not be part of the globally
optimal tree. This is the fundamental weakness of a single tree.
8.1.2 Tree Pruning: Cost-Complexity Pruning
Growing a large tree until every leaf contains only one observation gives a training error of zero — but wildly
overfits. The solution ispruning: grow a large treeT0 first, then remove branches that add little predictive
power.
Pruning based solely on training RSS leads to overly large trees (the smallest terminal node always reduces
training RSS). Instead, we usecost-complexity pruning(also called weakest-link pruning), which penalises
tree size:
43

## Page 44

RSSα(T) =
|T|∑
m=1
∑
i:x i∈Rm
(yi−ˆyRm)2 +α|T|
where|T|is the number ofleaf nodes(terminal nodes) in the subtreeT, andα≥0is the complexity parameter
(analogous toλin Ridge/Lasso).
•α= 0: no penalty on size⇒the full treeT0 is optimal.
•αlarge: each additional leaf must reduce RSS by more thanαto justify its inclusion⇒smaller trees.
Full Tree-Building and Pruning Procedure
1. Use recursive binary splitting to grow a large treeT0 (stopping only when nodes are very small)
2. Apply cost-complexity pruning: for each value ofα, find the subtreeT(α)that minimises RSSα(T).
Asαincreases, the optimal subtree shrinks monotonically
3. Use k-fold CV to chooseα: for eachαin a grid, compute the CV error. Selectˆαwith the lowest CV
error (or apply the one-SE rule)
4. Return the subtreeT(ˆα)fitted on the full training set
In Python (sklearn): the pruning parameter isccp_alpha. Use cost_complexity_pruning_path()
to obtain the sequence ofαvalues, then select viaGridSearchCV.
8.2 Classification Trees
Classification trees are built the same way as regression trees, but sinceY is categorical, the split criterion and
prediction rule change:
• Prediction: within each leaf, predict themajority class(the most common class among training
observations in that region).
• Split criterion: instead of RSS, use animpurity measurethat quantifies how mixed the classes are
within a node.
Let ˆpmk denote the proportion of training observations in regionRm belonging to classk. Three impurity
measures:
Measure Formula Properties
Classification error1−max k ˆpmk Simple; equals 0 only when a node is
pure; not differentiable — not
preferred for growing trees
Gini indexG= ∑K
k=1 ˆpmk(1−ˆpmk)Measures total variance across
classes; equals 0 when a node is pure
(oneˆpmk = 1);preferred for
growing trees
Cross-entropy (deviance)D=− ∑K
k=1 ˆpmk log(ˆpmk)Numerically similar to Gini; also
equals 0 for pure nodes; slightly more
computation
Gini index in depth: For a binary outcome (K= 2) withˆpm1 =p:
G=p(1−p) + (1−p)·p= 2p(1−p)
This is maximised atp = 0.5(maximum impurity / perfect mixing) and minimised atp = 0or p = 1(pure
node). A split is chosen to maximise the reduction in weighted Gini impurity across the two child nodes.
Classification error vs. Gini/entropy: Classification error is not sufficiently sensitive to changes in node
purity during tree growing (small moves inˆpmk that improve purity may not change the majority class). Gini
and entropy are more sensitive to purity and therefore preferred for determining splits. Classification error is
used for final tree evaluation (it is the most interpretable performance metric).
44

## Page 45

Why Gini index is preferred over classification error for growing trees
Consider a node with 400 observations: 200 from class A, 200 from class B (50%/50%). Two possible
splits:
Split 1: Left child: 300 A, 100 B. Right child: 100 A, 100 B.
Split 2: Left child: 200 A, 400 B (from another region). Right child: 0 A, 0 B (pure).
Both splits have the same classification error (one misclassification per node), but Split 2 creates one
pure node while Split 1 creates none. The Gini index correctly prefers Split 2 because it is more sensitive
to the purity improvement. Classification error would be indifferent between the two.
8.3 Advantages and Disadvantages of Trees
Advantages Disadvantages
Highly interpretable; easy to visualise and
explain to non-experts
High variance: small changes in training data can produce
very different trees
Naturally handles non-linearity and
interactions between predictors
Generally lower predictive accuracy than linear models when
the true relationship is approximately linear
Works with both quantitative and qualitative
predictors without dummy coding
Greedy splitting is not globally optimal
Mirrors human decision-making Unstable: sensitive to outliers and near-threshold
observations
Robust to irrelevant predictors (they simply do
not get selected for splits)
Requires pruning to avoid overfitting
No need to standardise predictors (distance is
not used)
Cannot extrapolate beyond training data ranges
The high variance of single trees is their main practical limitation — it motivates all the ensemble methods
below.
8.4 Bagging (Bootstrap Aggregating)
Baggingreduces variance by averaging many trees, each trained on a bootstrap sample of the data:
ˆfbag(x) = 1
B
B∑
b=1
ˆf∗b(x)
where ˆf∗bis a tree grown on theb-th bootstrap sample. For classification, the final prediction is themajority
voteacross theBtrees.
Why does averaging reduce variance?IfB i.i.d. estimators each have varianceσ2, their average has
varianceσ2/B. Bootstrap trees are not truly independent (they are trained on overlapping datasets), so the
variance reduction is less dramatic, but still substantial. Crucially, because each tree is grown deep (unpruned),
individual bias remains low — bagging reduces variance without increasing bias.
Key hyperparameter:B (number of trees). More trees always improve or maintain performance — there
is no overfitting asB increases. In practice,B = 100–500is sufficient; returns diminish rapidly after a few
hundred trees.
8.4.1 Out-of-Bag (OOB) Error
Recall from Chapter 5 that each bootstrap sample leaves out approximately 36.8% of observations. These
out-of-bag (OOB)observations can be used to estimate test error without any additional CV:
OOB Error Estimation
1. For each observationi, identify all treesb for whichi wasnotin the bootstrap sample (approximately
B/3trees)
2. Predict observationiusing only these OOB trees:ˆyOOB
i = 1
|{b:i/∈Z∗b}|
∑
b:i/∈Z∗bˆf∗b(xi)
3. The OOB error is the MSE (regression) or misclassification rate (classification) of these OOB
45

## Page 46

predictions across allnobservations
OOB error is a valid, nearly unbiased estimate of test error and is approximately equivalent to LOOCV.
It comes for free — no additional model fitting is needed.
8.4.2 Variable Importance from Bagged Trees
One downside of bagging is that averaging many trees destroys interpretability — we can no longer visualise
a single decision tree. However, we can measurevariable importance: for each predictorj, sum the total
reduction in RSS (regression) or Gini impurity (classification) from all splits on predictorj, averaged over all
Btrees. A large total reduction indicates an important predictor.
Importance(Xj) = 1
B
B∑
b=1
∑
splits onX j in treeb
∆RSS
Variable importance plots (bar charts sorted by importance) are the primary tool for understanding which
predictors drive the ensemble’s predictions.
8.5 Random Forests
Random forests improve on bagging bydecorrelatingthe trees. In bagging, allB trees use the same set of
p predictors — if there is one dominant predictor, virtually every tree will split on it near the root, making
all trees highly correlated. Averaging many correlated estimates reduces variance far less than averaging
independent ones.
The solution: at each split in each tree, consider only arandom subset ofmpredictors(not allp):
Default:m= √p(classification), m=p/3(regression)
By restricting the predictors at each split, trees are forced to use different predictors in different regions, making
them less correlated. Averaging decorrelated trees reduces variance more effectively than averaging correlated
bagged trees.
Bagging is a special case of Random Forests
When m =p (all predictors available at each split), Random Forests reduces to exactly Bagging. Asm
decreases:
•Trees become more diverse (less correlated)→lower variance when averaged
•Individual trees become weaker (they cannot always use the best predictor)→higher bias per tree
The bias increase from usingm < pis typically more than offset by the variance reduction from
decorrelation, so Random Forests withm =√p almost always outperforms Bagging (m = p). The
optimalmcan be tuned by CV.
Key hyperparameters:
Parameter Typical value Effect
B(n_estimators) 100–500 More trees always helps
(or neutral); no overfitting
m(max_features) √p(classification),p/3(regression) Lowerm→more
decorrelated trees, lower
variance, but weaker
individual trees
Max depth / min samples leaf Unrestricted (grow deep) Deep trees have low bias;
variance controlled by
averaging
Variable importanceis computed identically to bagging — averaged over allBtrees in the forest.
46

## Page 47

8.6 Gradient Boosting
Boosting takes a fundamentally different approach from bagging. Instead of fitting trees independently on
bootstrap samples and averaging, boosting fits treessequentially, each new tree correcting the errors of the
current ensemble.
8.6.1 The Boosting Idea
The core intuition: fit a tree, look at where it was wrong (the residuals), then fit the next tree to those residuals.
By repeatedly targeting the current mistakes, the ensemble gradually improves its fit to the training data.
Gradient Boosting Algorithm (Regression)
1. Initialise: ˆf(x) = 0and residualsr i =y i for alli
2. Forb= 1,2,...,B:
(a) Fit a regression treeˆfb withdsplits (i.e.d+ 1leaves) to the data(x i,ri)
(b) Update the ensemble by adding a shrunken version of the new tree:
ˆf(x)←ˆf(x) +νˆfb(x)
(c) Update residuals:r i←yi−ˆf(xi)(i.e.r i←ri−νˆfb(xi))
3. Output: ˆf(x) = ∑B
b=1νˆfb(x)
8.6.2 The Three Hyperparameters of Boosting
Boosting has three tuning parameters, all of which affect the bias-variance trade-off:
Hyperparameter Symbol Typical value Role
Number of treesB100–5000 (chosen by CV) More trees→
lower bias, but
can overfit if
Bis too large
Learning rate (shrinkage)ν0.001–0.1 Scales the
contribution
of each tree.
Smallerν
requires larger
Bbut
generally gives
better
generalisation
Tree depthd1–6 Controls the
complexity of
each tree and
the interaction
depth.d= 1
(stumps) gives
an additive
model;d= 2
allows
pairwise
interactions;
etc.
Interaction betweenνand B: Smaller learning rateνmeans each tree contributes less, so more treesB are
needed to achieve the same fit. The combination of smallνand largeB (e.g.ν= 0.01, B = 5000) tends to give
better generalisation than largeνand smallB. In practice, setνsmall and chooseB by CV or early stopping
on a validation set.
47

## Page 48

Tree depthd and interactions: Each split in a tree captures the effect of one variable conditionally on
preceding splits. A tree of depthd can capture interactions among up tod variables. Stumps (d = 1) make
boosting equivalent to an additive model (similar to a GAM). For most tabular data,d= 2–4is sufficient.
Boosting can overfit — unlike bagging and random forests
In bagging and random forests, increasingB never hurts (the variance of the average can only decrease).
In boosting, increasingB always reduces training error, but beyond a point it starts to overfit (memorise
noise). Always chooseB by cross-validation or monitor a held-out validation error during training and
stop early when it stops improving.
8.6.3 Why Does Boosting Work? The Bias Perspective
Bagging and random forests primarily reducevariance— individual deep trees have low bias, and averaging
reduces variance. Boosting primarily reducesbias— individual shallow trees (stumps) have high bias, and the
sequential fitting of residuals progressively reduces it. This is why boosting uses shallow trees: the ensemble
corrects for the bias of each tree by adding more trees. The result is a powerful method that can reduce both
bias and variance simultaneously, which is why gradient boosting (XGBoost, LightGBM, CatBoost) consistently
ranks among the top performers on tabular data benchmarks.
8.7 Summary of Ensemble Methods
Method Sampling Trees Depth Key advantage Key disadvantage
Single tree None 1 Pruned Interpretable High variance, moderate
accuracy
BaggingBootstrap
rows
Bindepen-
dent
Deep (un-
pruned)
Reduces variance Trees correlated if strong
predictor exists
Random
Forest
Bootstrap
rows +
random
features
Bindepen-
dent
Deep (un-
pruned)
Decorrelates trees,
further reduces
variance
Less interpretable;mto
tune
Gradient
Boosting
No bootstrap
(full data)
B
sequential
Shallow
(d= 1–6)
Reduces bias AND
variance; very high
accuracy
3 hyperparameters to tune;
can overfit
Choosing between Random Forests and Gradient Boosting
Random Forestsare preferred when:
•Training time is limited (trees are fully independent; easy to parallelise)
•Robustness is important (gradient boosting is more sensitive to noisy labels and outliers)
•OOB error is needed (free test error estimate without CV)
Gradient Boostingis preferred when:
• Maximum predictive accuracy is the goal (boosting typically outperforms random forests when
well-tuned)
•The data have a strong signal-to-noise ratio (less sensitive to overfitting when labels are clean)
In practice: try both, select by CV. For tabular data on Kaggle-style competitions, gradient boosting
(XGBoost, LightGBM) dominates.
Theory Questions – Chapter 8
Q1: Explain the difference between the Gini index and classification error as split criteria
for classification trees. Why is the Gini index preferred when growing trees?
A:Both measures quantify the impurity of a node — how mixed the class proportions are. Classification
error is1−maxk ˆpmk: it equals 0 when a node is pure and is maximised at1−1/K when classes are
equally distributed. The Gini index isG = ∑
k ˆpmk(1−ˆpmk): it also equals 0 for pure nodes and is
maximised at equal class proportions. The key difference is sensitivity: classification error only changes
when the majority class changes, so small improvements in purity (e.g. a split that movesˆpfrom 0.51
to 0.70 for the majority class) are not reflected in the split criterion. The Gini index is sensitive to
all changes in ˆpmk, not just those that affect the majority class — it changes continuously as class
48

## Page 49

proportions change. This makes the Gini index a better guide for choosing splits during tree growing,
since it consistently favours splits that create purer nodes. Classification error is still used for reporting fi-
nal test performance because it has a direct, interpretable meaning (fraction of observations misclassified).
Q2: Why do random forests outperform bagging in most settings? What is the role of the
hyperparameterm?
A:Bagging reduces variance by averagingB independent trees trained on bootstrap samples. However,
if one predictor is strongly associated withY, nearly every bagged tree will split on it near the root,
making the trees highly correlated. Averaging correlated random variables reduces variance by a factor
of roughly1/B·(1−ρ) +ρ(where ρis the correlation), which is much less than1/B when ρis large.
Random forests break this correlation by considering only a random subset ofm predictors at each split.
This forces different trees to use different predictors, producing more diverse (less correlated) trees. The
trade-off is that individual trees are slightly weaker (they cannot always use the best predictor), but
the variance reduction from decorrelation more than compensates, giving better generalisation. The
hyperparameter m directly controls this trade-off: smallerm produces more diverse, less correlated trees
(lower variance) but weaker individual trees (higher bias). The defaultsm =√p (classification) and
m=p/3(regression) work well in practice and can be tuned by CV.
Q3: Explain how gradient boosting differs from bagging in both its mechanism and the
type of error it primarily reduces. Why can gradient boosting overfit while bagging cannot?
A:Bagging fits B treesindependentlyon bootstrap samples of the training data and averages their
predictions. Each tree is grown deep (low bias, high variance) and the averaging operation reduces
variance. The key insight is that averaging does not change expected bias — it only reduces variance.
Therefore, bagging primarily reduces variance. Because each additional tree is independent and its
contribution is averaged into a larger ensemble, adding more trees can only help or be neutral — bagging
cannot overfit. Gradient boosting fits treessequentially: each tree is fitted to the current residuals (the
errors of the existing ensemble). The learning rateνscales each tree’s contribution. Because each tree
targets the mistakes of the previous ensemble, boosting progressively reduces bias — individual trees are
shallow (high bias, low variance) and the sequential process reduces their collective bias. However, as
more trees are added, the ensemble increasingly memorises the training data’s noise (especially with
large ν), which increases variance and causes overfitting. This is whyB must be chosen carefully by
cross-validation or early stopping, unlike in bagging where more trees are always at least as good.
49

## Page 50

9 Chapter 9 – Support Vector Machines
Support Vector Machines (SVMs) are a family of powerful classifiers built around the geometric concept of a
separating hyperplane. The SVM idea develops progressively through three stages: (1) the maximal margin
classifier for perfectly separable data, (2) the support vector classifier that tolerates some misclassification, and
(3) the full SVM that uses the kernel trick to handle non-linear decision boundaries. SVMs are particularly
effective in high-dimensional settings and remain competitive with ensemble methods when well-tuned.
9.1 What is a Hyperplane?
Inp-dimensional space, ahyperplaneis a flat affine subspace of dimensionp−1. It is defined by:
β0 +β1X1 +β2X2 +···+βpXp = 0
•Inp= 2dimensions, a hyperplane is aline.
•Inp= 3dimensions, a hyperplane is aplane.
•Inp>3dimensions, it is a(p−1)-dimensional flat surface.
The hyperplane dividesRp into two half-spaces:
β0 +β⊤x>0⇔xis on the positive side
β0 +β⊤x<0⇔xis on the negative side
The magnitude|β0 +β⊤x|(with∥β∥= 1) is theperpendicular distancefrom pointxto the hyperplane.
This signed distance forms the basis for all SVM classifiers.
We encode class labels asyi∈{−1, +1}(rather than{0, 1}), which simplifies the mathematical notation: a
classifier based on the hyperplane predictsˆyi =sign(β0 +β⊤xi).
9.2 The Maximal Margin Classifier (Hard Margin SVM)
When two classes arelinearly separable(a hyperplane exists that perfectly separates all training observations),
there are infinitely many separating hyperplanes. Themaximal margin classifierselects the one that
maximises themargin— the total perpendicular distance from the hyperplane to the nearest training
observations on each side.
Formally, let the hyperplane be{x: β0 +β⊤x= 0}with∥β∥= 1. The margin is2M where M is the distance
from the hyperplane to the nearest point. We solve:
max
β0,β,M
Msubject toy i(β0 +β⊤xi)≥M∀i,∥β∥= 1
The constraintyi(β0 +β⊤xi)≥Mensures every observation is correctly classified and lies at leastM units
from the hyperplane (sinceyi∈{−1,+1}, the product is positive when correctly classified).
9.2.1 Support Vectors
The observations that lie exactly on the margin boundaries — i.e. those for whichyi(β0 +β⊤xi) =M — are
calledsupport vectors. They are the only observations that determine the hyperplane. All other observations
could be moved without changing the solution, as long as they remain on the correct side of their margin
boundary.
Why "support vectors"?
These observations literally "support" the hyperplane: they are the points closest to the decision boundary
and define (support) where the margin lies. The hyperplane is entirely determined by the support
vectors; observations far from the boundary have no influence on it. This is the reason SVMs can be
effective even in high-dimensional spaces — only a small number of observations matter.
50

## Page 51

9.2.2 Limitations of the Maximal Margin Classifier
•Fails when data are not linearly separable: no feasible solution exists when classes overlap.
• Extremely sensitive to individual observations: because the hyperplane is determined only by the
support vectors, a single new observation near the boundary can completely change the hyperplane —
very high variance.
• Overfitting risk: the hardest constraint (all observations must be correctly classified) may produce a
very narrow margin that generalises poorly.
9.3 The Support Vector Classifier (Soft Margin SVM)
The support vector classifier (SVC) relaxes the hard margin constraint by allowing some observations to be on
the wrong side of the margin or even the wrong side of the hyperplane. This is controlled viaslack variables
ξi≥0, one per observation:
max
β0,β,M,ξi
Msubject to:y i(β0 +β⊤xi)≥M(1−ξi), ξi≥0,
n∑
i=1
ξi≤C,∥β∥= 1
9.3.1 Interpreting the Slack Variables
Value ofξi Location of observationi
ξi = 0Correctly classified and on or beyond the margin boundary — no
violation
0<ξi≤1On the correct side of the hyperplane but inside the margin
(between margin and boundary)
ξi >1On the wrong side of the hyperplane — misclassified
The total slack∑
iξi is bounded byC, which serves as a budget for violations.
9.3.2 The Role ofC(Budget Parameter)
Cis the most important tuning parameter of the SVC and directly controls the bias-variance trade-off:
•C = 0: no violations allowed⇒reduces to the hard-margin maximal margin classifier (only valid if classes
are separable).
• Large C: a large violation budget — the classifier tolerates many observations inside the margin or even
misclassified. This produces awide marginwith many support vectors. The classifier is less sensitive to
individual observations (lower variance, higher bias). In sklearn, this corresponds to asmallC (sklearn
parameterises by1/C, so largeCin sklearn means strict margin).
• Small C: a tight violation budget — very few violations allowed. The margin isnarrowwith fewer
support vectors. The classifier fits the training data more closely (higher variance, lower bias). In sklearn,
largeCmeans strict margin and smallCmeans soft margin.
sklearn’sCparameter is the inverse of the "budget"
In ISLP (and most textbooks), alargerC budget meansmoreviolations are allowed→widermargin
→lowervariance. In sklearn’sSVC, the parameterC is theregularisation parameter(penalty on
margin violations):large C→fewviolations tolerated→narrowmargin →highervariance. The two
conventions are opposite. On the exam, always clarify which convention you are using.
Observations that influence the classifier: only observations that either lie on the margin boundary or
violate the margin (i.e. those withξi > 0— inside or on the wrong side) become support vectors and influence
ˆf. Correctly classified observations far from the boundary haveξi = 0and play no role.
ChooseCby cross-validation.
9.4 The Support Vector Machine (Kernel Trick)
The SVC finds a linear decision boundary in the original feature space. For non-linearly separable data, we could
manually add polynomial or interaction features — but this is computationally expensive in high dimensions.
Thekernel trickachieves the same effect implicitly and efficiently.
51

## Page 52

9.4.1 The Dual Formulation and Inner Products
A key mathematical insight: the SVC solution can be written entirely in terms ofinner productsbetween
observations. The decision function takes the form:
f(x) =β0 +
n∑
i=1
αiyi⟨xi,x⟩
where αi≥0are Lagrange multipliers. Crucially,αi > 0only for the support vectors — all other observations
haveαi = 0and do not contribute. The inner product⟨xi, x⟩=x⊤
i xmeasures the similarity between training
observationiand test pointx.
The kernel trick: replace every inner product⟨xi,x⟩with akernel functionK(xi,x):
f(x) =β0 +
∑
i∈S
αiyiK(xi,x)
whereS is the set of support vectors. The kernel function computes a generalised similarity measure that
implicitly corresponds to computing the inner product in a high-dimensional (possibly infinite-dimensional)
feature space, without ever explicitly constructing that space. This makes it computationally tractable even
when the expanded feature space would be enormous.
9.4.2 Common Kernel Functions
Kernel Formula Decision boundary Key parameter
LinearK(x i,xi′) =
x⊤
i xi′
Linear (standard SVC) None
Polynomial
(degreed)
K(xi,xi′) =
(1 +x⊤
i xi′)d
Polynomial of degreed d
RBF /
Gaussian
K(xi,xi′) =
exp
(
−γ∥xi−xi′∥2) Arbitrarily non-linearγ
TheRBF (Radial Basis Function) kernelis by far the most commonly used in practice. It gives the
highest weight to training observations close to the test point (small∥xi−xi′∥2) and near-zero weight to distant
observations. It implicitly corresponds to an infinite-dimensional feature space and can represent arbitrarily
complex decision boundaries.
Intuition for the RBF kernel
The RBF kernelK(xi, xi′) = exp(−γ∥xi−xi′∥2)is essentially a Gaussian function of the squared
Euclidean distance between two observations. Whenxi′=x i (the same point),K = 1(maximum
similarity). Asxi′moves away fromxi, K decreases toward 0. The parameterγcontrols the rate of this
decay:
• Small γ: the kernel decays slowly — distant observations still have meaningful similarity. The
decision boundary is smooth and global (high bias, low variance).
• Large γ: the kernel decays rapidly — only very close observations are considered similar. The
decision boundary becomes highly local and complex (low bias, high variance, risk of overfitting).
9.4.3 ChoosingCandγby Cross-Validation
Both C and γ(for the RBF kernel) must be chosen by cross-validation. A common approach isgrid search:
try all combinations ofCandγfrom a candidate grid and select the combination with the lowest CV error.
Low value High value
C(sklearn
convention)
Wide margin; more violations; lower variance,
higher bias
Narrow margin; fewer violations; higher
variance, lower bias
γ(RBF
kernel)
Smooth, global boundary; high bias, low
variance
Complex, local boundary; low bias, high
variance; risk of overfit
52

## Page 53

The optimal combination balances the flexibility of the boundary (controlled byγ) against the tolerance for mis-
classification (controlled byC). A common starting grid:C∈{0.01, 0.1, 1, 10, 100}, γ∈{0.001, 0.01, 0.1, 1, 10}
— using a logarithmic scale in both directions.
The key advantage of kernels: efficient high-dimensional computation
Suppose we want a polynomial boundary of degreed = 3in p = 100dimensions. Explicitly constructing
all polynomial features would involve
(100+3
3
)
≈176,851features — storing and computing with these
is expensive. The polynomial kernelK(xi, xi′) = (1 +x⊤
i xi′)3 produces the same decision boundary
as if we had expanded to the full degree-3 feature space, but requires only a single dot product plus a
few arithmetic operations per pair of observations. The kernel trick is what makes SVMs tractable in
high-dimensional settings.
9.5 Multi-Class SVM
The basic SVM is a binary classifier (K= 2classes). ForK >2classes, two strategies extend it:
9.5.1 One-vs-One (OvO)
Train
(K
2
)
= K(K−1)/2binary SVMs, one for each pair of classes. For each test observation, apply all
K(K−1)/2classifiers and assign the observation to the class that wins the most pairwise contests (majority
vote). This is thedefault in sklearn.
Example: withK = 4classes, we train
(4
2
)
= 6binary SVMs: (1 vs 2), (1 vs 3), (1 vs 4), (2 vs 3), (2 vs 4), (3
vs 4). Each binary classifier votes for one class; the class with 3 or more votes (out of 6) wins.
9.5.2 One-vs-All (OvA)
TrainK binary SVMs, each comparing one class against all others combined. For test observationx, compute
the decision functionfk(x)for each of theKSVMs and assign to the class with the highest score:
ˆy= arg max
k
fk(x)
OvA requires training onlyK classifiers (vs.K(K−1)/2for OvO) and is therefore faster for largeK, but
class imbalance can be more severe (each classifier hasnk positives vs.n−nk negatives).
Strategy Number of SVMs Vote mechanism Default in sklearn
One-vs-One
(OvO)
K(K−1)/2Majority vote across
pairwise contests
Yes
One-vs-All
(OvA)
KAssign to class with
highestf k(x)
No
(decision_function_shape='ovr')
9.6 SVM vs. Other Classifiers
Aspect SVM (RBF kernel) Logistic Regression Random Forest
Decision
boundary
Arbitrary non-linear Linear Piecewise axis-aligned
Works well in
highp
Yes (kernel trick; few support
vectors)
Yes Less so (curse of
dimensionality for trees)
Interpretability Low (no coefficients; only
support vectors)
High (log-odds coefficients) Moderate (variable
importance)
Probabilistic
output
Requires extra calibration
(Platt scaling)
Direct probabilities Direct probabilities
Key hyperpa-
rameters
C,γ(RBF) RegularisationCorλ B,m, depth
Best when Clear margin of separation;
moderaten, largep
Linear boundary; need
interpretability
Largen; complex
interactions
53

## Page 54

Theory Questions – Chapter 9
Q1: What is the margin in the maximal margin classifier, and why do we want to maximise
it? What are the support vectors?
A:The margin is the perpendicular distance between the decision hyperplane and the nearest training
observations on each side. Maximising the margin produces the hyperplane that is furthest from all
training observations — intuitively, a larger margin means the classifier is more "confident" about
its predictions and more robust to small perturbations in the data or new observations close to the
boundary. Observations that lie exactly on the margin boundaries — i.e. those withyi(β0 +β⊤xi) =M
— are the support vectors. They are the only observations that determine the position and orien-
tation of the hyperplane; all other observations could be moved (as long as they remain on the
correct side of the margin) without changing the solution. This makes the maximal margin classifier
computationally efficient (only support vectors matter) but sensitive to individual boundary observations.
Q2: Explain the role of the slack variablesξi in the support vector classifier. What does it
mean whenξi = 0,0<ξi≤1, andξi >1?
A:The slack variablesξi≥0allow the support vector classifier to tolerate observations that violate
the margin or are misclassified. The constraintyi(β0 +β⊤xi)≥M(1−ξi)with ∑
iξi≤Cpermits
up to a total budgetC of violation. Whenξi = 0, observationi is correctly classified and lies on or
beyond the margin boundary — no violation. When0<ξi≤1, observationi is on the correct side of
the hyperplane but has crossed into the margin: it is correctly classified but within the margin zone
(between the margin boundary and the decision hyperplane). Whenξi > 1, observationi is on the
wrong side of the hyperplane and is misclassified. The budget parameterC controls the total amount of
violation tolerated: largeC allows many violations, producing a wider margin and a lower-variance (but
higher-bias) classifier; smallC allows few violations, producing a narrower margin and a higher-variance
(but lower-bias) classifier.
Q3: What is the kernel trick, and why is it essential for SVMs to handle non-linear
boundaries? Explain intuitively what the RBF kernel computes.
A:The SVC finds a linear boundary in the original feature space. To handle non-linear boundaries,
we could expand the feature space by adding polynomial or interaction terms, but this becomes
computationally prohibitive in high dimensions. The kernel trick exploits the fact that the SVC’s
decision function depends on the data only through pairwise inner products⟨xi, xi′⟩. By replacing these
inner products with a kernel functionK(xi,xi′), we implicitly map the data into a higher-dimensional
(possibly infinite-dimensional) feature space and compute inner products there, without ever explicitly
constructing that space. This is computationally efficient because the kernel evaluation requires only a
single function call per pair of observations. The RBF (Gaussian) kernelK(xi, xi′) = exp(−γ∥xi−xi′∥2)
measures similarity as a decreasing function of the Euclidean distance between two observations. It
assigns high similarity (close to 1) to observations that are nearby and near-zero similarity to distant
observations. With smallγ, the similarity decays slowly and the boundary is smooth; with largeγ, only
very close observations are considered similar, producing a highly local and potentially very complex
boundary. The SVM with RBF kernel corresponds to an infinite-dimensional feature space, making it
capable of representing arbitrarily complex decision boundaries.
54

## Page 55

10 Chapter 10 – Deep Learning and Neural Networks
Neural networks are a family of flexible, non-linear models that can approximate virtually any function given
enough hidden units and data. They underpin modern deep learning and achieve state-of-the-art performance
on image recognition, language modelling, and many structured prediction tasks. For tabular data — the
typical setting in BAN404 exams — neural networks compete with gradient boosting, and which is better
depends on the dataset. Understanding the architecture, training procedure, and regularisation of neural
networks is essential.
10.1 From Linear to Non-Linear: The Need for Activation Functions
A linear model computesˆY=Xβ. A neural network with one hidden layer computes:
Ak =h

β0k +
p∑
j=1
βjkXj

, k= 1,...,K(hidden layer)
ˆY=g 0 +
K∑
k=1
gkAk (output layer)
If the activation functionh were linear (i.e.h(z) =z), then the hidden layer would be a linear transformation
of X and the output would be a linear function of a linear function — which is still a linear function ofX.
Non-linear activation functions are essential: without them, stacking multiple layers provides no benefit
over a single linear model.
10.2 Architecture of a Multilayer Perceptron (MLP)
An MLP consists of an input layer, one or more hidden layers, and an output layer. Information flows forward
from inputs to output (forward pass).
Layer Description Size
Input layerOne node per predictor; no computation — simply
passesX j to the first hidden layer
pnodes
Hidden layerlEach node computes a weighted sum of the previous
layer’s outputs, then applies an activation functionh
Kl nodes
Output layerProduces the final prediction; activation function
depends on the task
1 node
(regression/binary);K
nodes (K-class)
For a network withLhidden layers of sizesK1,K 2,...,KL:
h(l) =h
(
W(l)h(l−1)+b (l)
)
, l= 1,...,L
ˆY=g
(
W(L+1)h(L) +b (L+1)
)
whereW (l) is the weight matrix andb(l) is the bias vector for layerl, andh and g are the hidden and output
activation functions respectively.
Total parameters: For a single hidden layer withKnodes, the total number of trainable parameters is:
p·K  
input-to-hidden weights
+K 
hidden biases
+K 
hidden-to-output weights
+ 1 
output bias
=pK+ 2K+ 1
Each additional hidden layer multiplies the parameter count further. Neural networks areoverparameterised
models — they typically have far more parameters than observations, making regularisation essential.
55

## Page 56

Universal approximation theorem
A single hidden layer MLP with enough hidden units (K large) and a non-linear activation function can
approximate any continuous function on a compact domain to arbitrary precision. This is the theoretical
justification for using neural networks as general-purpose function approximators.
However, "enough hidden units" may be astronomically large for complex functions. In practice,deep
networks(many layers with moderate width) learn complex functions more efficiently than a single
wide hidden layer, because successive layers learn hierarchical representations (e.g. edges→shapes→
objects in image recognition).
10.3 Activation Functions
The activation functionh introduces non-linearity into the network. Different functions are used in hidden
layers vs. output layers:
Function Formula Range Typical use
ReLUh(z) = max(0,z) [0,∞)Hidden layers — default choice;
fast; avoids vanishing gradients
Sigmoidh(z) = 1
1 +e−z (0,1)Output layer for binary
classification
Softmaxh(z k) = ezk
∑K
l=1ezl
(0,1), sums to 1 Output layer forK-class
classification
Tanhh(z) = ez−e−z
ez +e−z (−1,1) Hidden layers; zero-centred version
of sigmoid
Linearh(z) =z(−∞,∞)Output layer for regression
10.3.1 Why ReLU is Preferred in Hidden Layers
•Computationally efficient:max(0,z)is trivially fast to evaluate and differentiate.
• Avoids the vanishing gradient problem: sigmoid and tanh saturate (flat derivatives) for large|z|,
causing gradients to vanish during backpropagation and making deep networks very slow to train. ReLU
has a gradient of exactly 1 forz >0, preserving gradient flow.
• Sparsity: approximately half the ReLU units are zero at any given time, creating sparse activations that
act as implicit regularisation.
Why softmax for multi-class output?
For aK-class classification problem, the output layer needs to produceK probabilities that sum to
1. The softmax functionh(zk) =ezk/ ∑
lezl achieves this: it exponentiates allK logits z1,...,zK and
normalises so they sum to 1. The class with the highest logit gets the highest probability. This is the
multi-class generalisation of the sigmoid: forK = 2, softmax reduces to sigmoid applied to the difference
z1−z2.
10.4 Loss Functions
The network is trained by minimising aloss functionthat measures how far predictions are from the truth.
The appropriate loss depends on the prediction task:
Task Output activation Loss function Formula
Regression Linear Mean Squared Error (MSE) 1
n
∑
i(yi−ˆf(xi))2
Binary
classifica-
tion
Sigmoid Binary cross-entropy− 1
n
∑
i [yi log ˆpi + (1−yi) log(1−ˆpi)]
Multi-
class
classifica-
tion
Softmax Categorical cross-entropy− 1
n
∑
i
∑K
k=1yik log ˆpik
56

## Page 57

Binary cross-entropyis the negative log-likelihood of the Bernoulli distribution. Minimising it is equivalent
to maximum likelihood estimation for logistic regression — so a single sigmoid-output unit with binary
cross-entropy loss is mathematically equivalent to logistic regression with a highly non-linear feature space
constructed by the hidden layers.
Categorical cross-entropywith softmax output is the multi-class generalisation of MLE for multinomial
logistic regression.
10.5 Training: Backpropagation and Gradient Descent
Neural network training finds weightsW(l) and biasesb (l) that minimise the lossL. Unlike OLS regression,
there isno closed-form solution— the loss surface is non-convex with many local minima and saddle points.
Iterative gradient-based optimisation is used.
10.5.1 The Gradient Descent Update Rule
The fundamental update rule: move the weights in the direction of steepest descent of the loss:
W(l)←W(l)−η∂L
∂W(l)
where η >0is thelearning rate— the step size per gradient update. Computing∂L/∂W(l) requires the
backpropagation algorithm.
10.5.2 Backpropagation
Backpropagation is an efficient application of thechain ruleof calculus to compute gradients of the loss with
respect to all weights in the network. It propagates the gradient backward from the output layer through the
hidden layers:
Forward Pass and Backward Pass
Forward pass(compute predictions and loss):
1. Pass inputxi through each layer, computing activationsh(l) and finallyˆyi
2. Compute the lossL(ˆyi,yi)
Backward pass(compute gradients via chain rule):
1. Compute∂L/∂ˆy(gradient of loss w.r.t. output)
2. For each layerl=L,L−1,...,1(going backward):
(a) Compute∂L/∂z(l) using the chain rule and the gradient from layerl+ 1
(b) Compute∂L/∂W(l) =∂L/∂z(l)·(h(l−1))⊤
3. Update all weights:W (l)←W(l)−η∂L/∂W(l)
10.5.3 Mini-Batch Stochastic Gradient Descent
Computing the loss and gradient over alln observations at each step (full batch gradient descent) is expensive
for largen.Mini-batch SGDapproximates the gradient using a random subset (mini-batch) ofBmb≈32–256
observations per update:
•Epoch: one complete pass through the entire training dataset.
•Iteration: one gradient update using one mini-batch.
•One epoch =⌈n/Bmb⌉iterations.
Mini-batch SGD introduces noise into the gradient estimates, which helps escape shallow local minima and
saddle points. The noise also acts as an implicit regulariser.
10.5.4 Adaptive Optimisers
Plain SGD with a fixed learning rate often converges slowly or oscillates.Adaptive optimisersautomatically
adjust the learning rate for each parameter:
• Adam(Adaptive Moment Estimation): maintains a moving average of both the gradient and its squared
magnitude, giving effective per-parameter learning rates. Strongly recommended as the default for neural
networks.
57

## Page 58

• RMSProp: divides the learning rate by a running average of recent squared gradients, preventing the
learning rate from becoming too small in directions with consistent gradients.
In practice: useAdamwith a moderate initial learning rate (e.g.η= 0.001) unless there is a specific reason to
prefer plain SGD.
10.6 Hyperparameters of Neural Networks
Neural networks have many hyperparameters. Unlike model parameters (W(l), b(l)), hyperparameters are not
learnt from data — they must be set before training:
Hyperparameter Typical range Effect
Number of hidden layersL1–5 (tabular); up to 100+ (images,
text)
More layers→more
complex feature
hierarchies; harder to
train; more
regularisation needed
Nodes per layerKl 16–1024 Wider layers→more
parameters; higher
capacity; higher
variance
Learning rateη10 −4–10−2 Too high→divergence
or oscillation; too low
→very slow
convergence
Mini-batch sizeB mb 32–256 Larger batches→more
stable gradients; less
noise; may need
adjustedη
Number of epochs 10–1000 Train until validation
loss stops improving
(early stopping)
Dropout ratep drop 0.2–0.5 See regularisation
section
L2 weight decayλ10 −4–10−2 See regularisation
section
Tuning strategy: for tabular data, start with a moderate architecture (2–3 hidden layers, 64–256 nodes,
ReLU, Adam, early stopping), then tune by grid search or random search with CV.
10.7 Regularisation of Neural Networks
Neural networks are highly overparameterised and prone to overfitting. Three regularisation techniques are
standard:
10.7.1 L2 Regularisation (Weight Decay)
Add a penalty on the sum of squared weights to the loss:
Lreg =L+ λ
2
∑
l
∑
j,k
(
W (l)
jk
)2
Equivalently, the weight update becomesW(l)←(1−ηλ)W(l)−η∂L/∂W(l): weights are “decayed” toward
zero at each step. This is exactly the neural network analogue of Ridge regression and serves the same purpose
— shrinking weights reduces overfitting by penalising large, complex models.
10.7.2 Dropout
Dropout is a network-specific regularisation technique that has no direct analogue in classical statistics:
58

## Page 59

Duringtraining: at each forward pass, randomly set each hidden unit to zero with probabilitypdrop (typically
0.2–0.5). This is done independently for each mini-batch. Units that are set to zero do not contribute to the
forward pass and receive no gradient update in the backward pass.
Duringinference(prediction): all units are active, but their outputs are scaled by(1−pdrop)to keep expected
magnitudes consistent.
Why does dropout regularise?It prevents any single hidden unit from becoming excessively important.
The network cannot rely on any particular unit being present, so it must learn redundant representations
spread across many units. Equivalently, training with dropout is approximately equivalent to training and
averaging an exponential number of different network architectures (each dropout mask defines a different
sub-network), providing an ensemble-like variance reduction.
10.7.3 Early Stopping
Monitor thevalidation lossthroughout training and stop when it stops decreasing:
Early Stopping Procedure
1. Split training data into train (≈90%) and validation (≈10%) sets
2. After each epoch, compute the validation loss
3. If the validation loss has not improved forp consecutive epochs (patiencep, e.g.p = 10–20), stop
training
4. Restore the weights from the epoch with the best validation loss
Early stopping is simple, highly effective, and requires no additional hyperparameter tuning beyond the
patience value.
Early stopping works because overparameterised neural networks initially fit the true signal (validation loss
decreases), then begin fitting noise (validation loss increases while training loss continues to decrease). Stopping
at the right point prevents the transition into overfitting.
The three regularisers compared
L2 weight decay: penalises large weights globally; analogous to Ridge; works by biasing the model
toward simpler functions.
Dropout: randomly disables units during training; prevents co-adaptation of units; effectively trains an
ensemble of sub-networks.
Early stopping: prevents overtraining by monitoring validation performance; effectively limits the
number of gradient updates, controlling effective model complexity.
In practice, all three are often used simultaneously. Weight decay and dropout are set as hyperparameters;
early stopping is applied during all training runs.
10.8 The Bias-Variance Trade-off in Neural Networks
Neural networks are subject to the same bias-variance trade-off as all statistical models:
Network configuration Bias Variance Remedy
Too small / too few layers High (cannot
capture
complex
patterns)
Low Add more layers
or wider layers
Too large, no regularisation Low High (overfits training
data)
Add dropout, L2,
early stopping
Too large + regularisation Low–
moderate
Low–moderateOptimal
generalisation
The key insight from modern deep learning:larger networks with strong regularisation often outperform
smaller networks without regularisation, even if the smaller network is already overparameterised. This
runs counter to classical intuition and is still an active research area.
59

## Page 60

Standardisation is mandatory: unlike tree-based methods, neural networks are highly sensitive to the scale
of input features. Features on large scales dominate the gradient updates and prevent other features from being
learnt. Always standardise all inputs to zero mean and unit variancebeforetraining, fitting the scaler on
training data only.
10.9 Neural Networks vs. Other Methods
The ISLP textbook includes a comparison on the Ames Housing dataset (Chapter 13), where several methods
are compared on test MAE:
Method Relative test performance
OLS Weakest
Ridge / Lasso Better than OLS
Single tree Similar to OLS
Bagging Good
Random Forest Very good
Gradient Boosting Among the best
Neural Network (deep) Among the best
SVM Good
Neural networks and gradient boosting tend to be the top performers on tabular data with sufficient observations.
The choice between them depends on: interpretability requirements, data size, whether features require
engineering, and computational budget.
Theory Questions – Chapter 10
Q1: Why are activation functions necessary in a neural network? What happens if all
activation functions are linear?
A:Activation functions are necessary because they introduce non-linearity into the network. Without
non-linearity, stacking multiple layers provides no additional modelling power: a composition of linear
functions is itself a linear function. Formally, if layer 1 computesh(1) =W (1)xand layer 2 computes
h(2) =W (2)h(1), thenh (2) =W (2)W(1)x= ˜Wx, which is just a single linear transformation. Any
L-layer network with linear activations is equivalent to a single-layer linear model, regardless of width
or depth. Non-linear activations (ReLU, sigmoid, tanh) break this collapse, allowing the network to
represent complex, non-linear functions of the inputs. ReLU is preferred in hidden layers because it
is computationally simple, avoids the vanishing gradient problem that plagues sigmoid and tanh at
saturation, and produces sparse activations.
Q2: Explain the backpropagation algorithm. What problem does it solve, and what
mathematical principle does it use?
A:Backpropagation solves the problem of efficiently computing the gradient of the lossL with respect to
all weights in the network — a computation that is necessary for gradient descent but involves millions of
parameters in deep networks. The mathematical principle is thechain rule of calculus: ifL depends
onz (L) which depends onW (L), then∂L/∂W(L) = (∂L/∂z(L))·(∂z(L)/∂W(L)). Backpropagation
exploits the layered structure of the network to propagate gradients backward from the output layer
to the input layer, reusing intermediate computations. A forward pass first computes and stores all
activationsh (l) and pre-activationsz (l). The backward pass then computes the gradient at the output
layer and propagates it back through the layers using the chain rule, accumulating gradients for each
weight matrix. The entire gradient computation costs approximately twice a forward pass — the
algorithm is remarkably efficient given the number of parameters involved.
Q3: Explain how dropout regularises a neural network. How does it differ during training
versus inference, and why does it work?
A:During training, dropout randomly sets each hidden unit to zero with probabilitypdrop at each
forward pass, independently across mini-batches. This means no unit can be relied upon to always be
present, so the network cannot develop strong co-adaptations between units (e.g. unitA learning to
correct the specific errors of unitB). Instead, each unit must learn a robust representation that is useful
60

## Page 61

independently. During inference, all units are active, but their outputs are multiplied by(1−pdrop)to
keep the expected total input to each subsequent layer the same as during training. Dropout works
for three related reasons: (1) it prevents co-adaptation of hidden units, forcing the network to learn
more redundant and robust features; (2) it is approximately equivalent to training and averaging an
exponential number of sub-networks (one per dropout mask), providing ensemble-like variance reduction;
and (3) it acts similarly to L2 regularisation in practice by preventing individual weights from growing
too large. Dropout is most effective in the middle hidden layers and is typically not applied to the input
or output layers.
61

## Page 62

11 Chapter 11 – Unsupervised Learning
All methods studied so far aresupervised: every observation carries a labelY that guides learning. Unsuper-
vised learning removes this supervision — we have only a matrix of featuresXand seek to discover structure,
patterns, or compact representations within the data. There is no definitive “right answer” to compare against,
which makes unsupervised learning simultaneously more exploratory and harder to evaluate. ISLP Chapter 12
covers three pillars:Principal Component Analysis (PCA)for dimension reduction,K-means clustering
for partition-based grouping, andhierarchical clusteringfor tree-structured grouping.
11.1 Principal Component Analysis
11.1.1 Motivation and Intuition
When p is large, visualisation and modelling become difficult. PCA seeks a low-dimensional representation of
the data that retains as much variation as possible. The key idea: find a sequence of orthogonal directions in
Rp such that the first direction captures the greatest variance in the data, the second (orthogonal to the first)
the next greatest, and so on.
These directions are calledprincipal components(PCs). Projecting the data onto the firstM≪pPCs
yields ann×Mmatrix that compresses the original data while discarding only the least variable — and
presumably least informative — directions.
11.1.2 Loading Vectors and Scores
Thefirst principal component loading vectorϕ1 = (ϕ11,ϕ21,...,ϕp1)⊤solves:
ϕ1 = arg max
ϕ:∥ϕ∥=1
Var


p∑
j=1
ϕj1xij

 = arg max
ϕ:∥ϕ∥=1
1
n
n∑
i=1


p∑
j=1
ϕj1xij


2
The unit-norm constraint∥ϕ1∥= 1prevents the trivial solution of making the loadings arbitrarily large. The
first principal component scorefor observationiis the projection:
zi1 =
p∑
j=1
ϕj1xij =ϕ⊤
1 xi
Thesecond loading vector ϕ2 maximises variance subject to the same unit-norm constraintandthe
additional orthogonality constraintϕ2⊥ϕ1. This continues for all subsequent components. The scores
zim =ϕ⊤
mxi form the columns of thescore matrixZ(n×M).
PCA via the singular value decomposition
In practice, PCA is computed via thesingular value decomposition(SVD) of the (mean-centred)
data matrix ˜X =UDV⊤, whereUis n×porthogonal,Dis diagonal with singular valuesd1≥d2≥
···≥dp≥0, andVis p×porthogonal. The columns ofVare the loading vectors; the columns ofUD
are the score vectors. The singular values relate to variance: the variance explained by them-th PC is
d2
m/n.
Mean-centring is mandatory.PCA measures variance, and variance is relative to the mean. If the
data are not centred, the first PC will primarily capture the overall level of the variables rather than
genuine variation around the mean.
11.1.3 Proportion of Variance Explained
Thetotal varianceof the data is ∑p
j=1 Var(Xj). After mean-centring, this equals ∑p
j=1
1
n
∑n
i=1x2
ij =
1
n
∑p
m=1d2
m.
Theproportion of variance explained (PVE)by them-th PC is:
PVEm = d2
m∑p
l=1d2
l
=
1
n
∑n
i=1z2
im∑p
j=1
1
n
∑n
i=1x2
ij
The PVEs sum to 1 across allpcomponents. Plotting PVEm againstmgives thescree plot.
62

## Page 63

11.1.4 Choosing the Number of Components
There is no single correct answer; the choice is typically guided by:
• Scree plot / elbow rule: plot PVE (or cumulative PVE) vs.m and look for an “elbow” — the point
beyond which additional components add little additional variance. This is subjective but widely used.
• Cumulative PVE threshold: chooseM such that the cumulative PVE exceeds a target, e.g. 80% or
90%.
• Downstream task performance: if PCA is used as a preprocessing step (e.g. before KNN or regression),
use cross-validation on the downstream task to selectM.
Scaling before PCA
If the variables are measured on different scales (e.g. income in thousands of dollars vs. age in years),
the variable with the larger variance will automatically dominate the first PC. In this case,standardise
all variables to zero mean and unit variance before running PCA. If the variables are on the same scale
(e.g. gene expression measurements), scaling may not be necessary — but it must be a deliberate choice,
not an oversight.
11.1.5 The Biplot
Abiplotdisplays both the PC scores (observations, usually shown as points) and the loading vectors (variables,
shown as arrows) in the same two-dimensional space of the first two PCs. The direction and length of an arrow
represent how strongly the corresponding variable contributes to PC1 and PC2. Observations projected onto an
arrow’s direction have high values for that variable. The biplot is a powerful exploratory tool for simultaneously
identifying observation clusters and understanding which variables drive each principal component.
PCA Algorithm
1.Mean-centreeach variable:˜x ij =x ij−¯xj (and standardise if scales differ)
2. Compute the SVD of˜X=UDV ⊤
3. Loading vectors: columns ofV(i.e.ϕm =v m, them-th column)
4. Score matrix:Z= ˜XV(equivalently,z im = ˜x⊤
i ϕm)
5. PVE ofm-th PC:d 2
m/ ∑
ld2
l
6. ChooseMvia scree plot or cumulative PVE threshold
11.1.6 PCA for Regression: Principal Components Regression
PCR uses the firstM PC scores Z1,...,ZM as predictors in an OLS regression (replacing the originalp
predictors). Because the PCs are orthogonal, there is no multicollinearity. PCR implicitly assumes the
directions of greatest variance inXare also the most predictive forY — an assumption that is often but
not always justified. PCR is not a feature selection method: each PC is a linear combination of all original
predictors.
11.2K-Means Clustering
11.2.1 Problem Formulation
K-means clustering partitions then observations intoK non-overlapping, non-empty clustersC1,C 2,...,CK.
The goal is to minimise thewithin-cluster variation (WCV):
min
C1,...,CK
K∑
k=1
WCV(Ck)
The standard measure of within-cluster variation is squared Euclidean distance:
WCV(Ck) = 1
|Ck|
∑
i,i′∈Ck
p∑
j=1
(xij−xi′j)2
This is equivalent to minimising thetotal within-cluster sum of squares:
63

## Page 64

K∑
k=1
∑
i∈Ck
p∑
j=1
(xij−¯xkj)2 =
K∑
k=1
|Ck|
p∑
j=1
Vark(Xj)
where¯xkj is the mean of thej-th feature in clusterk(the centroid).
11.2.2 TheK-Means Algorithm
Exact minimisation of the WCV is NP-hard (requires checking allKn assignments). The standard algorithm
finds a good local minimum:
K-Means Algorithm (Lloyd’s Algorithm)
1. Randomly assign each of thenobservations to one ofKclusters (random initialisation)
2.Repeat until convergence:
(a) Assignment step: for each observationi, assign it to the clusterk whose centroid ¯xk is closest:
C(i) = arg mink∥xi−¯xk∥2
(b) Update step: recompute each centroid as the mean of all observations currently assigned to it:
¯xk = 1
|Ck|
∑
i∈Ck
xi
3. Convergence is guaranteed (WCV never increases) and typically occurs within tens of iterations
Because K-means finds a local minimum that depends on the random initialisation, it is standard practice to
run the algorithm multiple times with different random starts and keep the solution with the smallest total
WCV.
Why convergence is guaranteed
Each step of the algorithm cannot increase the total WCV. In the assignment step, each observation
moves to the nearest centroid, reducing or maintaining its individual contribution to WCV. In the
update step, the centroid is the unique minimiser of the sum of squared distances to all cluster members
(the mean minimises squared error). Since there are finitely many possible assignments and WCV is
bounded below by zero, the algorithm must converge.
11.2.3 Choosing the Number of ClustersK
Unlike supervised methods, there is no held-out label to guide model selection. Several heuristics exist:
Elbow method: plot the total within-cluster sum of squares (TWCSS) againstK. AsK increases, TWCSS
always decreases (adding more clusters can only reduce within-cluster variance). Look for an elbow — a value
ofKbeyond which the rate of decrease slows markedly. This is subjective.
Silhouette analysis: for each observationi, compute:
s(i) = b(i)−a(i)
max(a(i),b(i))
where a(i)is the mean distance fromi to all other observations in its cluster (within-cluster cohesion) andb(i)
is the mean distance fromi to all observations in the nearest other cluster (between-cluster separation). The
silhouette scores(i)∈[−1, 1]: values close to 1 indicate the observation is well-matched to its cluster, close to
0 means it is on the boundary between clusters, and negative values suggest it may be in the wrong cluster.
ChooseKto maximise the average silhouette score.
Gap statistic: compares the observed TWCSS to the expected TWCSS under a null reference distribution
(data with no cluster structure). A large gap indicates the data has more structure than expected by chance.
ChooseKwhere the gap first exceeds the gap forK+ 1minus one standard error.
K-means requires standardisation and fixedK
K-means uses Euclidean distance, so variables on larger scales dominate. Always standardise features
before clustering. Additionally, K must be specified in advance — this is a significant limitation.
Hierarchical clustering avoids this requirement.
64

## Page 65

11.3 Hierarchical Clustering
11.3.1 Motivation
Hierarchical clustering is an alternative toK-means that does not require specifyingK in advance. Instead,
it builds adendrogram— a tree-shaped diagram that shows the nested grouping of observations at every
possible level of similarity. The analyst can then choose a level of the tree (corresponding to a number of
clusters) after inspection.
The most common form isagglomerative(bottom-up) hierarchical clustering: start with each observation
as its own cluster, then successively merge the two most similar clusters until a single cluster containing all
observations remains.
11.3.2 The Agglomerative Algorithm
Agglomerative Hierarchical Clustering
1. Begin with n clusters, one per observation. Compute the n×ndissimilarity matrixDwhere
dij =d(x i,xj)
2. Find the pair of clusters( A,B )with the smallest inter-cluster dissimilarity:( A,B ) =
arg minA̸=Bd(A,B)
3. MergeAandBinto a single new clusterA∪B. Record the merge height asd(A,B)
4. Update the dissimilarity matrix: compute the dissimilarity betweenA∪Band all remaining clusters
using the chosenlinkagecriterion
5. Repeat steps 2–4 until only one cluster remains
The resulting sequence of merges defines the dendrogram. Theheightof a merge on the dendrogram corresponds
to the dissimilarity at which the two groups were joined — higher merges indicate groups that are more dissimilar.
Cutting the dendrogram horizontally at heighthyields the clusters that exist at that dissimilarity level.
11.3.3 Linkage Methods
The linkage criterion determines how the dissimilarity between two clusters (each containing multiple observa-
tions) is defined:
Linkage Definition Properties
Completed(A,B) = max i∈A,j∈Bd(xi,xj)—
maximum pairwise dissimilarity
Tends to produce compact, roughly
equal-sized clusters; sensitive to
outliers (the worst-case pair
dominates)
Singled(A,B) = min i∈A,j∈Bd(xi,xj)—
minimum pairwise dissimilarity
Can produce long, “chained” clusters
(one observation at a time joins a
growing chain); very sensitive to
bridges between clusters
Averaged(A,B) =
1
|A||B|
∑
i∈A
∑
j∈Bd(xi,xj)— mean
pairwise dissimilarity
Compromise between complete and
single; less sensitive to outliers than
complete
Ward’sMinimise the increase in total
within-cluster variance upon merging
Tends to produce compact, balanced
clusters; similar toK-means
objective; often preferred for general
use
Complete and average linkageare the most commonly used in practice.Ward’s linkageis popular when
the goal is to produce clusters similar toK-means.Single linkageis rarely preferred because of its chaining
tendency.
11.3.4 Dissimilarity Measures
The algorithm requires a dissimilarity measured(xi, xj)between observations. The choice matters and should
reflect the problem:
65

## Page 66

• Euclidean distance:d(xi, xj) =∥xi−xj∥2 =
√∑
k(xik−xjk)2 — standard default; sensitive to scale,
so standardise first.
•Manhattan distance:d(x i,xj) = ∑
k|xik−xjk|— more robust to outliers than Euclidean.
• Correlation-based distance: d(xi, xj) = 1 −corr(xi, xj)— appropriate when the overall level
(magnitude) of observations is not relevant, only their relative patterns (e.g. gene expression profiles: a
gene expressed at level 10, 20, 30 and one at level 100, 200, 300 should be considered similar).
Forcategorical variables, the most common dissimilarity is the proportion of features on which two
observations differ.
Reading a dendrogram
Two observations are similar if the height at which they first merge (their "join height") is small. Do not
compare the height of leaves directly — two leaves close to each other on the page may join at a high
height if they are assigned to different early clusters. The meaningful quantity is theheight of the
common ancestor node, not the horizontal distance between leaves.
To obtainK clusters from a dendrogram: draw a horizontal cut line at an appropriate height and count
the number of branches crossing it — this isK. Choosing the cut height is analogous to choosingK in
K-means.
11.3.5 Dendrogram and Cluster Stability
One attraction of hierarchical clustering is that the dendrogram displays the clustering at every level simultane-
ously, allowing the analyst to judge which cut is most natural. Alarge gapbetween successive merge heights
suggests a natural number of clusters (the clusters at the lower level are quite different from each other, and
the merge at the gap height is costly). Conversely, if merge heights increase gradually with no distinct gaps,
there may be no strong cluster structure in the data.
11.4 Comparing Clustering Methods
FeatureK-Means Hierarchical (Agglomerative)
Number of clusters Must specifyKInferred from dendrogram
Result type Flat partition Nested hierarchy (dendrogram)
Algorithm Iterative (Lloyd’s) Sequential merging
Objective Minimise WCV
(Euclidean)
Minimise linkage-based dissimilarity
Sensitivity Random initialisation Linkage and distance choice
Scalability Scales well to largen O(n 2)memory; slow for largen
Handles
non-spherical
clusters?
Poorly (assumes
convex clusters)
Depends on linkage (single: yes, complete: no)
11.5 Practical Considerations
Scaling/standardisation: both K-means and PCA are sensitive to variable scales. Standardise to zero mean
and unit variance unless variables are already on the same scale or there is a domain reason not to.
Reproducibility: K-means is random (initialisation) and different runs may give different solutions. Always
set a random seed and run multiple initialisations. Hierarchical clustering is deterministic given the data,
dissimilarity measure, and linkage.
Validating cluster structure: unlike supervised learning, there is no test error. Use domain knowledge,
stability analysis (repeat on bootstrap samples and measure if clusters are preserved), or internal validation
metrics (silhouette, gap statistic) to assess whether structure is meaningful or artefactual.
The curse of dimensionality in clustering: Euclidean distance becomes less meaningful in high dimensions
— all pairwise distances concentrate around the same value. In high-p settings, consider applying PCA first
(dimensionality reduction) and then clustering on the leading PCs.
66

## Page 67

Theory Questions – Chapter 11
Q1: What are principal components? Define the loading vector and the score vector for
the first PC, and explain how the second PC differs from the first.
A:Principal components are a sequence of orthogonal directions in thep-dimensional feature space
that successively capture the maximum remaining variance in the data. Thefirst loading vector
ϕ1∈Rp (with∥ϕ1∥= 1) is the direction along which the projected data has the largest variance.
Formally,ϕ1 = arg max∥ϕ∥=1Var( ˜Xϕ). Thefirst score vectorz 1∈Rn contains the projections of all
n (mean-centred) observations onto this direction:zi1 =ϕ⊤
1 ˜xi. These scores are the coordinates of the
observations in the new PC space and are used as reduced-dimension representations. Thesecond PC
is found by solving the same maximisation problem with the additional constraint thatϕ2⊥ϕ1 —
i.e. the second loading vector must be orthogonal to the first. This constraint ensures the second PC
captures variance not already captured by the first. The scoreszi2 =ϕ⊤
2 ˜xi are therefore uncorrelated
with the first scores. Subsequent PCs follow the same pattern: each is the direction of maximum
remaining variance orthogonal to all previous PCs.
Q2: Describe theK-means algorithm step by step. Why is convergence guaranteed? What
is the main limitation ofK-means, and how is it typically addressed in practice?
A:The K-means algorithm begins by randomly assigning each of then observations to one ofK clusters.
It then iterates between two steps: (1) theassignment step, where each observation is reassigned to
the cluster whose centroid (mean) is nearest in Euclidean distance, and (2) theupdate step, where
each centroid is recomputed as the mean of all observations currently in that cluster. These steps
repeat until assignments no longer change. Convergence is guaranteed because each step monotonically
decreases the total within-cluster sum of squares (TWCSS): in the assignment step, each observation
moves to a strictly nearer (or equally near) centroid; in the update step, the cluster mean is the unique
minimiser of squared Euclidean distance within the cluster. Since TWCSS is bounded below by zero
and there are finitely many distinct assignments, the algorithm must terminate. The main limitation is
that K-means finds alocal minimumof TWCSS whose quality depends on the random initialisation.
The same data with different starting assignments can yield very different clusterings. The standard
remedy is to run the algorithm many times (e.g. 10–100) with different random initialisations and retain
the solution achieving the lowest TWCSS.
Q3: Compare complete linkage and single linkage in hierarchical clustering. What kinds of
clusters does each produce, and in what situations might you prefer one over the other?
A: Complete linkagedefines the dissimilarity between two clusters as themaximumpairwise distance
between any observation in the first cluster and any observation in the second:dcomplete(A,B ) =
maxi∈A,j∈Bd(xi, xj). Because it uses the worst-case pair, complete linkage tends to producecompact,
roughly spherical clusters of similar size— two clusters are not merged until even their most
distant members are fairly close. It is, however, sensitive to outliers: a single extreme observation can
greatly inflatedcomplete(A,B )and delay a natural merge.Single linkagedefines the dissimilarity as the
minimumpairwise distance: dsingle(A,B ) = mini∈A,j∈Bd(xi, xj). Because it merges on the best-case
pair, single linkage tends to produceelongated, chained clusters: a single close observation can
"bridge" two otherwise distant groups, causing them to merge into a long chain. This chaining tendency
makes single linkage poorly suited to discovering compact, separated clusters, but it excels at detecting
clusters of unusual shapes (e.g. crescent-shaped or filamentary structures). In most practical settings
with roughly convex clusters, complete or average linkage (or Ward’s) is preferred over single linkage.
67
