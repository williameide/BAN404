# BAN404 ExamRoadmap — Master Index

> **This is your navigation hub. Everything you need for the exam is here. Start here, always.**

---

## The System at a Glance

| File | What it does | When to use |
|------|-------------|-------------|
| **You are here** `00_master_index.md` | Navigation hub, learning order, priorities | Start every study session |
| `01_curriculum_exam_map.md` | Topic × year frequency table, likelihood scores | First read to understand what matters |
| `02_gap_analysis.md` | Honest audit of existing materials | Once, for orientation |
| `03_r_from_zero_for_ban404.qmd` | R language basics + code translator | First week; reference when confused |
| `snippets/ban404_core_snippets.R` | All key R snippets in one file | During exam — copy paste |
| `04_task1_mastery.qmd` | Method recognition + explain + extend (Task 1) | Core study — main review file |
| `05_task2_playbooks.qmd` | Full EDA→model→evaluate→interpret pipelines | Core study — Task 2 preparation |
| `templates/task2_exam_template.qmd` | Blank exam template to fill during exam | During exam — open immediately |
| `06_drill_book.qmd` | Timed practice drills by topic | Active study — use daily |
| `07_mock_exams.qmd` | 3 full mock exams with solutions | Exam simulation — final week |
| `08_self_grading_rubric.md` | Answer quality rubric + fallback strategy | After every practice attempt |
| `09_exam_day_runbook.md` | Pre-exam → during exam → per-task workflows | Exam morning and during exam |

**Existing materials to use alongside:**
| File | What it is | Priority |
|------|-----------|----------|
| `Task1_comp.qmd` | 25 Task 1 code patterns with "The Tell" recognition cues | HIGH — read this |
| `Practical_understanding_comp.qmd` | When to use which method | HIGH — read this |
| `Previous Exam/R_exam_compendium_2026_retake.qmd` | Full worked solutions 2025, 2024, 2023 | HIGH — study solved examples |
| `Course_approval/BAN404_compendium_text.txt` | Full course theory | MEDIUM — theory reference |
| `Previous Exam/BAN404_exam_202X.md` | All 5 exam transcripts | Use for practice |

---

## A→Z Learning Order (follow this sequence)

### Week 1 — Foundation

| Day | Task | Files |
|-----|------|-------|
| Day 1 | Read exam format overview | `01_curriculum_exam_map.md` |
| Day 1 | Read R basics (Parts 1-2) | `03_r_from_zero_for_ban404.qmd` |
| Day 2 | Read "The Tell" for all 10 methods | `04_task1_mastery.qmd` (skim) |
| Day 2 | Read Task 2 pipeline overview | `05_task2_playbooks.qmd` (Playbook A header) |
| Day 3 | Work through 2025 exam in detail | `Previous Exam/R_exam_compendium_2026_retake.qmd` |
| Day 4 | Work through 2024 exam in detail | `Previous Exam/R_exam_compendium_2026_retake.qmd` |
| Day 5 | Work through 2023 exam in detail | Churn.csv + compendium |
| Day 6-7 | Re-read all of `04_task1_mastery.qmd` and `05_task2_playbooks.qmd` | — |

### Week 2 — Drills

| Day | Task | Files |
|-----|------|-------|
| Day 8 | Topic drills: Bootstrap + LOOCV | `06_drill_book.qmd` Topics 1-2 |
| Day 9 | Topic drills: Ridge + GAM | `06_drill_book.qmd` Topics 3-4 |
| Day 10 | Topic drills: Logistic + RF + Boosting | `06_drill_book.qmd` Topics 5-6 |
| Day 11 | Mixed drills | `06_drill_book.qmd` Mixed section |
| Day 12 | Mock Exam 1 (timed, 6h) | `07_mock_exams.qmd` Exam 1 |
| Day 13 | Grade Mock 1, review weaknesses | `08_self_grading_rubric.md` |
| Day 14 | Review weak topics from Mock 1 | `04_task1_mastery.qmd` or `05_task2_playbooks.qmd` |

### Week 3 — Exam Simulation

| Day | Task | Files |
|-----|------|-------|
| Day 15 | Mock Exam 2 (timed, 6h) | `07_mock_exams.qmd` Exam 2 |
| Day 16 | Grade + review | `08_self_grading_rubric.md` |
| Day 17 | Mock Exam 3 (timed, 8h — home exam format) | `07_mock_exams.qmd` Exam 3 |
| Day 18 | Grade + final review | — |
| Day 19-20 | Final read: `04_task1_mastery.qmd` + `snippets/ban404_core_snippets.R` | — |
| Day 21 | Read `09_exam_day_runbook.md` the morning before | — |

---

## Priority by Topic (revised from exam frequency analysis)

### 🔴 Do not skip — these appear every year

1. **Logistic regression** — fit, interpret, threshold, confusion matrix
2. **Random Forest** — fit, variable importance, evaluate
3. **Bootstrap** — SE, histogram, CI (normal + percentile)
4. **LOOCV** — inner loop structure, lambda/K tuning
5. **GAM** — detect non-linearity, fit with `s()`, compare MSE/R²

### 🟠 Very likely — prepare fully

6. **Ridge regression** — objective function, demeaning, LOOCV tuning
7. **Boosting (gbm)** — bernoulli/gaussian, n.trees, shrinkage, evaluate
8. **Confusion matrix + threshold** — class imbalance logic, row proportions
9. **Train/test split + data cleaning** — factor encoding, leakage removal
10. **EDA workflow** — boxplot, prop.table, cor(), summary()

### 🟡 Likely — prepare a solid answer

11. **KNN local regression** — distance, `order()`, LOOCV for K
12. **Backfitting** — explain in plain English (often theory-only)
13. **Lasso (glmnet)** — abs penalty, variable selection, cv.glmnet
14. **Regression/classification tree** — fit, prune, interpret, plot
15. **Variable importance** — explain MeanDecreaseAccuracy and Gini

### 🟢 Know for theory — skip code depth

16. SVM, PCA/PCR, LDA/QDA, K-means, polynomial regression
17. These appear in the course text but **never appeared on actual exams** (2021–2025)

---

## Quick Reference: Most Important Code Lines

```r
# Bootstrap
sample(1:n, size=n, replace=TRUE)

# LOOCV  
X[-i, ]    # Remove row i

# Ridge penalty
la * sum(b1^2)

# Lasso penalty
la * sum(abs(b1))

# GAM smooth
gam(y ~ s(x1) + x2)

# Logistic
glm(y ~ ., family=binomial())
predict(..., type="response")   # → probabilities

# RF
randomForest(as.factor(y) ~ ., mtry=..., ntree=500)

# Boosting
gbm(y ~ ., distribution="bernoulli", n.trees=1000, shrinkage=0.01)

# R²
1 - MSE_model / var(y)

# Confusion matrix
table(Actual=test$y, Predicted=pred)
prop.table(cm, margin=1)
```

---

## Pre-Exam Checklist (15 minutes before exam starts)

- [ ] Packages installed: `randomForest`, `gbm`, `gam`, `tree`, `glmnet`, `insuranceData`, `ISLR2`
- [ ] `snippets/ban404_core_snippets.R` open in a separate editor tab
- [ ] `templates/task2_exam_template.qmd` copied and renamed for today
- [ ] `09_exam_day_runbook.md` open
- [ ] `04_task1_mastery.qmd` open on the "Recognition Cues" section
- [ ] Calculator ready (for mental R² checks)
- [ ] Timer set

---

## Exam-Day File Access Order

1. **Open first:** `templates/task2_exam_template.qmd` → rename to `[date]_exam_answers.qmd`
2. **Read exam questions:** Identify Task 1 method from code → go to `04_task1_mastery.qmd`
3. **During Task 2:** Follow `05_task2_playbooks.qmd` step by step
4. **If stuck:** Check `snippets/ban404_core_snippets.R` and `08_self_grading_rubric.md` fallback tree
5. **Final check:** `09_exam_day_runbook.md` post-answer sanity checklist
