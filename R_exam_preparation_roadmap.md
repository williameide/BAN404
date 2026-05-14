# BAN404 R Retake Roadmap (Next Days)

## Goal
Build reliable execution speed for the 6-hour R exam while using your stronger interpretation skills.

## Day-by-day plan

### Day 1: Setup + core pipelines
- Install/check all exam packages.
- Rehearse 2 full pipelines:
  - regression pipeline (OLS + ridge/lasso + GAM + tree/RF)
  - classification pipeline (logistic + threshold + tree/RF/boosting)
- Build one clean Quarto exam template and verify PDF knit.

### Day 2: Latest exam #1 (2025) deep practice
- Solve 2025 Task 1 fully timed (3 hours).
- Solve 2025 Task 2 fully timed (3 hours).
- Post-practice: write “mistake log” and improve snippets.

### Day 3: Latest exam #2 (2024) deep practice
- Solve methodological Task 1 quickly and cleanly.
- Solve airline Task 2 with clear split between:
  - explanation question (“why dissatisfied”)
  - operational prediction question.
- Focus on writing short, high-quality interpretations.

### Day 4: weak-point reinforcement
- Drill only weak points from mistake log:
  - threshold tuning,
  - CV loops,
  - GAM specification,
  - RF/boosting comparison text.
- Run one 3-hour mixed mini-mock.

### Day 5: speed + communication
- Solve selected subproblems from 2021/2022/2023 patterns.
- Practice “exam-style output”: concise justification, no unnecessary output.
- Finalize snippets and compendium bookmarks.

### Day before exam (dataset release day)
- Load released dataset and do 90-minute reconnaissance:
  - variable types, missingness, target distribution,
  - quick EDA,
  - likely leakage variables,
  - candidate formulas.
- Prepare lightweight script blocks (regression + classification).
- Do not overfit to one model; prepare decision branches.

## Exam execution strategy

1. Read all tasks first and allocate time by points.
2. Solve high-certainty tasks first (secure points).
3. Always produce a valid model and explanation before optimizing.
4. Keep one function block for repeated metrics and confusion tables.
5. Reserve final 15 minutes strictly for render and upload.

## Daily mandatory drills (short)

- 15 min: confusion matrix + threshold interpretation.
- 15 min: LOOCV/k-fold coding from memory.
- 15 min: one GAM or tree model + interpretation sentence.
- 15 min: one short Quarto writeup paragraph.

## Mindset rules

- “Good and complete” beats “perfect but unfinished.”
- Explain your choices whenever there are multiple valid options.
- If a model fails, document fallback and continue.
