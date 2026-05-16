# BAN404 Gap Analysis — What Is Missing for a Self-Contained R Exam Roadmap

> **Purpose:** Identify exactly what the repository already covers, what is weak or missing, and what downstream agents need to build. This is an honest quality audit.

---

## 1. What Existing Materials Cover WELL

| Material | Strength |
|----------|----------|
| `Task1_comp.qmd` | Excellent. 25 code patterns with "The Tell" recognition cues, plug-and-play written answers, follow-up questions. Covers all high-probability Task 1 methods. |
| `Practical_understanding_comp.qmd` | Good. Explains when to use `loo(la,X,y)` vs `loo(X,y)`, decision pipeline logic, train/test vs CV choice. |
| `Previous Exam/R_exam_compendium_2026_retake.qmd` | Good for 2025 and 2024. Has full code + written answers for every sub-question of both exams. 2023 likely there too. |
| `Previous Exam/BAN404_exam_2021.md` – `2025.md` | All 5 exams fully transcribed. Ready to use. |
| `Course_approval/BAN404_compendium_text.txt` | Covers theory (Chapters 2-9, 12). Dense but complete. |
| `BAN404_Kompendium_Sander.pdf` | Full theory reference (allowed on exam). |

---

## 2. What Is MISSING or WEAK

### 2a. Critical gaps (blocks exam success)

| Gap | Why it matters | Priority |
|-----|----------------|----------|
| **Zero-to-R foundation** | Student has no prior coding knowledge. Cannot use any existing material without first understanding: variable assignment, indexing, data frames, factors, loops, functions. Without this, Task 1 code-explain is guesswork. | **P1** |
| **2021 & 2022 full worked solutions** | The compendium only has 2025, 2024, 2023 solutions. 2021 (bootstrap fundamentals) and 2022 (GAM + bagging) are missing worked R code. Both have testable patterns. | **P1** |
| **Task 2 standardized playbook** | No file shows the full EDA → model → evaluate → interpret pipeline as a reusable template. Each exam's Task 2 is similar, but the student has no drill-ready template to use under time pressure. | **P1** |

### 2b. Important gaps (reduces exam score)

| Gap | Why it matters | Priority |
|-----|----------------|----------|
| **Confusion matrix + threshold section** | Threshold selection appears in EVERY exam Task 2. The existing compendium mentions it but has no standalone drill or explanation of *why* to use a non-standard threshold. | **P2** |
| **R² computation and interpretation** | Appeared in 2025 T1e. Not covered independently. Formula is `1 - MSE_model / Var(y)`. | **P2** |
| **Variable importance explanation** | Appears in 2022, 2023, 2024 Task 2. Code exists but no standalone explanation of what the measure means or how to interpret the plot. | **P2** |
| **GAM from scratch (identify nonlinearity → fit → interpret)** | The compendium shows backfitting code but the practical workflow of "scan plots → choose s() terms → fit gam → compare MSE" is not spelled out step-by-step for a beginner. | **P2** |
| **Written interpretation templates** | Written explanations exist in the compendium for some methods, but there is no fill-in-the-blank template for: coefficient interpretation, confusion matrix commentary, variable importance commentary, or final "business answer". | **P2** |

### 2c. Nice-to-have gaps (polish)

| Gap | Why it matters | Priority |
|-----|----------------|----------|
| **Mock exams** | No full practice exams under timed conditions. Without drills, the student cannot test time management (6 hours, ~10 sub-questions). | **P3** |
| **Self-grading rubric** | No way to assess quality of answers. Student doesn't know if their written explanation is worth 8/10 or 3/10. | **P3** |
| **Exam-day runbook** | No "what to do in the first 10 minutes" guide. Under stress, students lose time on setup, not knowing where to start. | **P3** |
| **Debug checklist** | R errors under exam pressure (wrong column name, factor vs numeric, missing package) are common. No structured troubleshooting guide. | **P3** |

---

## 3. Prioritized Missing-Pieces Backlog for Downstream Agents

### Agent 2 must build:
- [ ] `03_r_from_zero_for_ban404.qmd` — R basics for BAN404 (zero-knowledge start, BAN404-scoped only)
- [ ] `snippets/ban404_core_snippets.R` — Copy-paste code blocks for all key methods

### Agent 3 must build:
- [ ] `04_task1_mastery.qmd` — Method-by-method Task 1 drill with recognition → explain → modify pipeline
- [ ] Must incorporate 2021 and 2022 Task 1 worked solutions (bootstrap, CV, GAM, logistic)

### Agent 4 must build:
- [ ] `05_task2_playbooks.qmd` — Full Task 2 playbooks (regression + classification versions)
- [ ] `templates/task2_exam_template.qmd` — Blank-but-structured exam template to use during exam
- [ ] Include: factor handling, EDA choice logic, model selection logic, threshold explanation, written answer templates

### Agent 5 must build:
- [ ] `06_drill_book.qmd` — Timed drills by topic + mixed drills
- [ ] `07_mock_exams.qmd` — 3 full mock exams in 2024/2025 format
- [ ] `08_self_grading_rubric.md` — Answer quality rubric + fallback strategy

### Agent 6 must build:
- [ ] `00_master_index.md` — Navigation hub with learning order and priorities
- [ ] `09_exam_day_runbook.md` — Pre-exam → during exam → per-task workflows

---

## 4. Coverage Summary

```
Method              | Task1_comp | Compendium_2026 | Practical_comp | MISSING
--------------------|------------|-----------------|----------------|--------
Ridge               | ✅ Full    | ✅ Full (2025)  | ✅ Partial     | Nothing
Bootstrap           | ✅ Full    | ✅ Full (2025)  | —              | 2021 wk sol
LOOCV               | ✅ Full    | ✅ Full (2025)  | ✅ Full        | Nothing
GAM/backfitting     | ✅ Full    | ✅ Partial      | —              | Step-by-step wf
KNN                 | ✅ Full    | ✅ Full (2024)  | —              | Nothing
Logistic Regr.      | ✅ Full    | ✅ Full         | —              | Written templates
Random Forest       | ✅ Code    | ✅ Full         | —              | VarImp explain
Boosting            | ✅ Code    | ✅ Full (2025)  | —              | Nothing
Trees               | ✅ Code    | ✅ Partial      | —              | Full wk solution
Lasso               | ✅ Code    | ✅ Partial      | —              | LOOCV+glmnet full
Confusion matrix    | ✅ Code    | ✅ Partial      | —              | Threshold logic
OLS                 | ✅ Full    | ✅ Full         | —              | Nothing
R basics            | ❌ None   | ❌ None        | ❌ None        | ENTIRE SECTION
R² formula          | ❌ None   | ✅ 1 line       | —              | Standalone
Written templates   | ✅ Partial | ✅ Partial      | —              | Fill-in-blank
Drills/mock exams   | ❌ None   | ❌ None        | ❌ None        | ENTIRE SECTION
Exam-day guide      | ❌ None   | ❌ None        | ❌ None        | ENTIRE SECTION
```
