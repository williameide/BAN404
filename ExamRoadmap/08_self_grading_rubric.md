# BAN404 Self-Grading Rubric

> Use this after completing any mock exam or practice sub-question. Score yourself honestly. The rubric mirrors how BAN404 examiners award partial credit.

---

## Overall Scoring Philosophy

| Level | Description | Points (of 10) |
|-------|-------------|----------------|
| **0** | Blank, completely wrong method, or no meaningful attempt | 0 |
| **3** | Correct method identified but code/explanation has major errors | 3 |
| **5** | Correct method, partially correct code or explanation, key step missing | 5 |
| **7** | Correct method and code, minor conceptual gap or incomplete explanation | 7 |
| **8** | Correct code + mostly complete explanation, minor imprecision in theory | 8 |
| **10** | Correct code, correct and clear written answer, addresses all sub-parts | 10 |

---

## Rubric by Task Type

### Task Type A: "Explain what this R function does"

| Points | Criteria |
|--------|----------|
| 10 | Named the method correctly, explained each major step in plain English, stated the mathematical formula/objective (e.g., RSS + λΣβ²), mentioned key assumption (e.g., demeaning) |
| 8 | Named the method, explained most steps, omitted formula or one assumption |
| 6 | Named the method, described the loop or structure but missed the statistical interpretation |
| 4 | Said "it fits a model" or "it does regression" without naming the specific method |
| 2 | Described some lines of code in isolation without a coherent statistical picture |
| 0 | Blank or completely wrong |

**Key check:** Did you say the name of the method AND explain why it exists?

---

### Task Type B: "Use this function on data"

| Points | Criteria |
|--------|----------|
| 10 | Code runs correctly, correct output shown, result interpreted (e.g., "Optimal λ is 500") |
| 8 | Code runs, correct output, but no or minimal written interpretation |
| 6 | Correct approach but code has minor error (e.g., wrong variable name, off-by-one index) |
| 4 | Wrong function call or wrong input format, but correct understanding of what should be done |
| 2 | Code fragment only, or wrong method used entirely |
| 0 | Blank |

---

### Task Type C: "Compare two models"

| Points | Criteria |
|--------|----------|
| 10 | Both models fitted correctly, appropriate metric (MSE, accuracy) computed for both, conclusion stated with specific numbers, bias-variance or theoretical explanation included |
| 8 | Both models fitted and compared numerically, conclusion stated, no theoretical explanation |
| 6 | Both models fitted but comparison is only qualitative ("ridge is better") without numbers |
| 4 | Only one model fitted, or wrong metric used |
| 2 | Models listed but not fitted or compared |
| 0 | Blank |

---

### Task Type D: "Evaluate predictions (confusion matrix)"

| Points | Criteria |
|--------|----------|
| 10 | Probabilities predicted, threshold justified, confusion matrix shown, row proportions computed, accuracy stated, interpretation of sensitivity/specificity included |
| 8 | Threshold, confusion matrix, accuracy — all correct. Missing row proportions or brief interpretation |
| 6 | Confusion matrix shown but threshold is 0.5 for imbalanced data without justification |
| 4 | Predicted class labels but no confusion matrix or accuracy |
| 2 | Only predicted probabilities shown, no evaluation |
| 0 | Blank |

**Common mistake to avoid:** Using threshold=0.5 on a dataset with 2% positive rate and saying "accuracy is 98%." This scores 4/10 at most because the model is useless.

---

### Task Type E: "Bootstrap CI"

| Points | Criteria |
|--------|----------|
| 10 | Bootstrap loop correct, histogram shown, BOTH types of CI computed (normal + percentile), explanation of why percentile may be preferred |
| 8 | Bootstrap loop and CI correct, only one CI type or missing histogram |
| 6 | Bootstrap loop correct, CI formula wrong (e.g., used sd(y) instead of sd(boot_results)) |
| 4 | Attempted bootstrap but replace=TRUE missing or wrong statistic computed |
| 2 | CI formula only (no bootstrap) |
| 0 | Blank |

---

### Task Type F: "Descriptive statistics / EDA"

| Points | Criteria |
|--------|----------|
| 10 | Appropriate plots/tables used for each variable type (boxplot for continuous, prop.table for categorical), results summarized in words with specific numbers, promising predictors identified with reasoning |
| 8 | Correct plots, summary in words, but without specific numbers or without clear reasoning about which predictors are useful |
| 6 | Plots created but no written summary, or only one type of visualization |
| 4 | `summary(data)` shown as "EDA" without targeted analysis |
| 2 | Data loaded but no actual analysis performed |
| 0 | Blank |

---

### Task Type G: "Variable importance interpretation"

| Points | Criteria |
|--------|----------|
| 10 | `varImpPlot` shown, top 3 variables named, correct explanation of the measure (permutation-based OR Gini), connection to EDA findings stated |
| 8 | Plot shown, top variables named, interpretation given but explanation of measure incomplete |
| 6 | Plot shown, variables named, no explanation of what the measure means |
| 4 | Code shown but no interpretation |
| 2 | Only `varImpPlot(rf)` without output or any comment |
| 0 | Blank |

---

### Task Type H: "Final business answer / interpretation"

| Points | Criteria |
|--------|----------|
| 10 | Uses specific numbers from analysis, names 2-3 key predictors with their direction of effect, connects findings to business decision, mentions model with best performance, acknowledges limitations |
| 8 | Names key predictors with direction, mentions best model, no specific numbers or no business connection |
| 6 | Vague summary ("variable A is important") without numbers or business relevance |
| 4 | Lists variables without effect direction or model comparison |
| 0 | Blank |

---

## "If Stuck" Fallback Strategy Decision Tree

```
Stuck on Task 1 (code explain)?
├── Do you see sum(b^2)? → Ridge regression
├── Do you see sum(abs(b))? → Lasso regression
├── Do you see sample(..., replace=TRUE)? → Bootstrap
├── Do you see X[-i,] in a loop? → LOOCV
├── Do you see abs(x-x0) + order()? → KNN
├── Do you see smooth.spline in a loop with y-f2? → Backfitting / GAM
├── Do you see exp(eta)/(1+exp(eta))? → Logistic sigmoid
└── Not sure? → Write "This implements [best guess] because [evidence from code]"
               Partial credit is better than blank. Always name something.

Stuck on Task 2 (dataset analysis)?
├── Can't start EDA? → Run: summary(train); boxplot(y~target, data=train)
├── Can't fit model? → Use simplest version: lm(y~., data=train) or glm(y~., family=binomial)
├── Model won't run? → Check: str(data) — are factors correct? Any NAs? 
├── RF won't run? → Make sure target is a factor: as.factor(target)
├── Boosting won't run? → Make sure target is 0/1 numeric: as.numeric(as.character(target))
├── Can't interpret? → Use the template: "Variable X (coef=Y, OR=Z) increases/decreases..."
└── Running out of time? → Submit what you have, add brief written answer for unfinished parts
                           "If I had more time I would [describe approach]. Expected result: [describe]"
```

---

## Time Allocation Guide (6-hour school exam)

| Phase | Time | Action |
|-------|------|--------|
| Setup (0:00–0:10) | 10 min | Read entire exam, load template, install packages |
| Task 1 (0:10–2:30) | 140 min | ~25 min per sub-question (5 sub-questions × 10p) |
| Break + review (2:30–2:45) | 15 min | Check Task 1 answers, note what to return to |
| Task 2 (2:45–5:15) | 150 min | ~25 min per sub-question (but 2a may need 40 min) |
| Final review (5:15–6:00) | 45 min | Fill in missing written answers, check code runs |

**Priority rule:** Get 6/10 on 8 questions rather than 10/10 on 4 questions.

---

## Quick Score Calculator

After each mock exam, compute your score:

```
Task 1: 
  1a: __ / 10
  1b: __ / 10
  1c: __ / 10
  1d: __ / 10
  1e: __ / 10
  Task 1 total: __ / 50

Task 2:
  2a: __ / 10
  2b: __ / 10
  2c: __ / 10
  2d: __ / 10
  2e: __ / 10
  Task 2 total: __ / 50

EXAM TOTAL: __ / 100
```

**Grade benchmarks (BAN404 Norwegian grading):**

| Score | Grade |
|-------|-------|
| 90–100 | A |
| 77–89 | B |
| 65–76 | C |
| 53–64 | D |
| 41–52 | E |
| < 41 | F |
