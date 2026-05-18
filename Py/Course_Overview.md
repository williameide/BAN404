# BAN404 - Statistical Learning / Machine Learning for Business

## Course Overview

BAN404 is an advanced course in statistical learning and machine learning methods for business applications. The course is based on "An Introduction to Statistical Learning" (ISLP - Python edition) and covers both theoretical foundations and practical implementation using Python.

---

## Course Structure

The course is organized into **12 lectures** covering progressively advanced topics in statistical learning, from foundational regression methods to advanced ensemble methods and neural networks.

---

## Module 1: Linear Models and Regression (Lectures 2-3)

### Lecture 2: Linear Regression
- **Topics:** Simple and multiple linear regression, least squares estimation, model assessment
- **Key Concepts:** RSS, R², coefficient interpretation, hypothesis testing for coefficients
- **Code Exercise:** Exercise 3.8 from ISLP - Analyzing the Auto dataset
  - Fitting linear models (mpg vs horsepower)
  - Diagnostic plots and residual analysis
  - Confidence and prediction intervals
- **Tools:** `statsmodels`, `ISLP`, `matplotlib`

### Lecture 3: Logistic Regression
- **Topics:** Classification, logistic regression, maximum likelihood estimation
- **Key Concepts:** Odds ratios, log-odds, decision boundaries, confusion matrix
- **Code Exercise:** Exercise 4.13 (a-d) from ISLP - Weekly stock market prediction
  - Logistic regression for direction prediction
  - Training/test split methodology
  - Model evaluation metrics
- **Tools:** `sklearn.linear_model.LogisticRegression`, `statsmodels`

---

## Module 2: Classification Methods (Lecture 4)

### Lecture 4: Discriminant Analysis and KNN
- **Topics:** Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), K-Nearest Neighbors
- **Key Concepts:** Bayes classifier, discriminant functions, decision boundaries, bias-variance tradeoff in KNN
- **Code Exercise:** Exercise 4.13 (e-g) - Comparing classifiers on Weekly data
  - LDA and QDA implementation
  - KNN with different K values
  - Comparing classifier performance
- **Tools:** `sklearn.discriminant_analysis`, `sklearn.neighbors`

---

## Module 3: Resampling Methods (Lecture 5)

### Lecture 5: Cross-Validation and Bootstrap
- **Topics:** Model validation, cross-validation (LOOCV, k-fold), bootstrap
- **Key Concepts:** Overfitting detection, variance estimation, model selection
- **Code Exercise:** Exercise 5.5 - Default dataset analysis
  - Validation set approach
  - LOOCV for logistic regression
  - Bootstrap standard error estimation
- **Tools:** `sklearn.model_selection`, custom bootstrap implementations

---

## Module 4: Time Series and Regularization (Lectures 6-7)

### Lecture 6: Time Series Analysis
- **Topics:** Time series data, stock market analysis, temporal patterns
- **Data:** stocks.csv - Financial time series data
- **Key Concepts:** Trend analysis, seasonality, autocorrelation

### Lecture 7: Regularization Methods
- **Topics:** Ridge regression, Lasso regression, elastic net
- **Key Concepts:** L1/L2 penalties, shrinkage, feature selection, tuning parameter selection
- **Code Exercise:** Exercise 6.11 - Boston crime data prediction
  - Ridge regression with cross-validation
  - Lasso for variable selection
  - Comparing regularized vs. OLS models
- **Tools:** `sklearn.linear_model.Ridge`, `sklearn.linear_model.Lasso`

---

## Module 5: Non-Linear Methods (Lectures 8-9)

### Lecture 8: Polynomial and Step Functions
- **Topics:** Polynomial regression, step functions, basis functions, splines
- **Key Concepts:** Flexibility vs. interpretability, degrees of freedom, knot selection
- **Code Exercise:** Exercise 7.6 - Wage data analysis
  - Polynomial regression fitting
  - Step function implementation
  - Model comparison via cross-validation
- **Tools:** `sklearn.preprocessing.PolynomialFeatures`, `ISLP`

### Lecture 9: Generalized Additive Models (GAMs)
- **Topics:** GAMs, smooth functions, additive structure
- **Key Concepts:** Partial response functions, smooth terms, model interpretation
- **Code Exercise:** GAM example implementation
  - Fitting GAMs with multiple predictors
  - Visualizing partial effects
  - Comparing with linear models
- **Tools:** `pygam`, `statsmodels`

---

## Module 6: Tree-Based Methods (Lectures 10-11)

### Lecture 10: Decision Trees
- **Topics:** Classification and regression trees (CART), tree pruning
- **Key Concepts:** Recursive partitioning, Gini index, cross-entropy, cost-complexity pruning
- **Code Exercise:** Exercise 8.8 (a-c) - Tree analysis
  - Building classification/regression trees
  - Pruning strategies
  - Tree visualization
- **Tools:** `sklearn.tree`, `ISLP`

### Lecture 11: Ensemble Methods
- **Topics:** Bagging, Random Forests, Boosting (AdaBoost, Gradient Boosting)
- **Key Concepts:** Bootstrap aggregation, out-of-bag error, variable importance, boosting iterations
- **Code Exercise:** Exercise 8.8 continued - Ensemble comparisons
  - Random Forest implementation
  - Boosting methods
  - Feature importance analysis
- **Tools:** `sklearn.ensemble.RandomForestClassifier`, `sklearn.ensemble.GradientBoostingClassifier`

---

## Module 7: Support Vector Machines (Lecture 12)

### Lecture 12: Support Vector Machines
- **Topics:** Maximal margin classifier, support vector classifier, SVM with kernels
- **Key Concepts:** Hyperplanes, support vectors, soft margins, kernel trick (linear, polynomial, RBF)
- **Code Exercise:** Exercise 9.7 - Auto classification
  - Linear SVM implementation
  - Kernel SVM for non-linear boundaries
  - Tuning C and gamma parameters
- **Tools:** `sklearn.svm.SVC`, `sklearn.svm.SVR`

---

## Module 8: Neural Networks and Model Comparison (Lecture 13)

### Lecture 13: Neural Networks and Comprehensive Review
- **Topics:** Neural network basics, deep learning introduction, model comparison
- **Key Concepts:** Activation functions, layers, backpropagation, model selection strategies
- **Code Exercise:** Ames Housing comprehensive analysis
  - Comparing multiple ML methods
  - Neural network implementation
  - Final model selection and evaluation
- **Data:** AmesHousing.csv - Comprehensive housing dataset
- **Tools:** `sklearn.neural_network`, `tensorflow`/`keras` (potentially)

---

## Key Skills Developed

### Programming & Tools
- **Python Libraries:** numpy, pandas, matplotlib, seaborn
- **ML Frameworks:** scikit-learn, statsmodels, ISLP
- **Specialized:** pygam for GAMs, various ensemble methods

### Statistical Concepts
- Bias-variance tradeoff
- Model selection and validation
- Regularization and overfitting prevention
- Feature engineering and selection

### Practical Applications
- Regression for continuous outcomes
- Classification for categorical outcomes
- Time series analysis
- Ensemble methods for improved prediction

---

## Learning Roadmap

```
Week 1-2:  Foundation    → Linear regression, logistic regression
Week 3:    Classification → LDA, QDA, KNN
Week 4:    Validation    → Cross-validation, bootstrap
Week 5-6:  Regularization → Time series, Ridge, Lasso
Week 7-8:  Flexibility   → Polynomials, splines, GAMs
Week 9-10: Trees         → CART, Random Forests, Boosting
Week 11:   Margins       → Support Vector Machines
Week 12:   Integration   → Neural Networks, method comparison
```

---

## Assessment

The course includes:
- Compulsory assignments (hands-on coding exercises)
- Written examination testing both theory and application

---

## Prerequisites

- Basic statistics and probability
- Python programming fundamentals
- Linear algebra basics (matrix operations)

---

## Reference Materials

- **Primary Text:** "An Introduction to Statistical Learning with Applications in Python" (ISLP)
- **Course Compendium:** BAN404_Kompendium_Sander.pdf
- **Lecture PDFs and code exercises**

---

*This course provides a comprehensive foundation in modern statistical learning methods, preparing students for advanced analytics and data science roles in business contexts.*
