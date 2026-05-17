# BAN404 Exam Lookup (Printable)

## Main files to print / keep ready

- `Kompendium_NEW.pdf` — theory wording
- `cheatsheet_practical.qmd` — practical workflow and EDA
- `task1_cheatsheet.qmd` — Task 1 method recognition
- `task2_cheatsheet.qmd` — Task 2 guided structure
- `Previous_exams_solved.qmd` — reusable worked examples

## Previous exams at a glance

| Year | Main Task 1 style | Main Task 2 style |
|---|---|---|
| 2025 | Ridge, LOOCV, bootstrap, GAM | insurance classification |
| 2024 | KNN / local methods and related interpretation | airline-style classification |
| 2023 | core supervised-learning methods | churn-style classification |
| 2022 | theory / earlier-format material | package-dataset style |
| 2021 | theory / earlier-format material | package-dataset style |

## Fast reminders

- Read data before changing it
- Convert real categories to factors
- Watch for leakage variables
- Do EDA before modeling
- Use boxplots for numeric vs class
- Use stacked proportions for category vs class
- Use correlation heatmap for multicollinearity
- If OLS looks unstable because predictors are highly correlated:
  - ridge = safer for prediction
  - lasso = useful for variable selection
- Report what the code shows, not what you hoped to find

## During the exam

1. Identify the task type
2. Start from the closest runnable example
3. Change variable names carefully
4. Run the code
5. Read the output
6. Then write the explanation
