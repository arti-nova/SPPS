# Student Performance Prediction System

## Objective
Predict student final grade (G3) using attendance, assignment scores, internal 
assessments, and demographic/behavioral indicators, using the UCI Student 
Performance dataset.

## Dataset
- Source: UCI Machine Learning Repository — Student Performance
- File used: `student-mat.csv` (395 students, 33 features) — *[update this line 
  if you ended up switching to student-por.csv or the merged 649-row version]*
- No missing values, no duplicates (confirmed in `data_understanding.ipynb`)

## Key EDA Findings
- G3 distribution shows a spike of 38 students with G3 = 0 — likely 
  dropouts/non-completions rather than genuine low scores. Flagged with a 
  `dropped_out` binary column rather than dropped, since removing them would 
  discard real signal about at-risk students.
- `failures` is the strongest behavioral predictor (r = -0.36), with a clean 
  monotonic drop in mean G3 across failure counts (11.25 → 5.69).
- `absences` shows a surprisingly weak raw correlation (r = 0.03), likely 
  suppressed by an outlier at 75 absences.
- G1 and G2 (prior period grades) correlate very strongly with G3 (0.80, 0.90). 
  Since the project objective explicitly includes "internal assessments," these 
  are kept as legitimate features in one model variant, and excluded in a 
  second variant to isolate what non-grade features alone can predict.

## Methodology
1. **Data cleaning**: verified no nulls/duplicates; flagged G3=0 cases
2. **Feature engineering**: binary encoding for yes/no columns, one-hot 
   encoding for multi-category columns (Mjob, Fjob, reason, guardian)
3. **Two feature variants**: with G1/G2 (upper-bound sanity check) and 
   without G1/G2 (the harder, more realistic version matching the stated 
   objective)
4. **Models tested**: Linear Regression, Ridge, Lasso, Random Forest 
   (default and GridSearchCV-tuned)
5. **Evaluation**: 5-fold cross-validation (MAE, R²), plus held-out 
   test-set diagnostics (predicted-vs-actual, residuals)

## Results

| Model                               |   MAE (mean) |   MAE (std) |   R² (mean) |   R² (std) |
|:------------------------------------|-------------:|------------:|------------:|-----------:|
| Random Forest (with G1/G2, tuned)   |        0.634 |       0.065 |       0.964 |      0.009 |
| Random Forest (with G1/G2, default) |        0.641 |       0.069 |       0.963 |      0.01  |
| Linear Regression (with G1/G2)      |        0.937 |       0.062 |       0.929 |      0.012 |
| Random Forest (no G1/G2, tuned)     |        2.128 |       0.223 |       0.63  |      0.049 |
| Random Forest (no G1/G2, default)   |        2.137 |       0.231 |       0.624 |      0.048 |
| Lasso (no G1/G2)                    |        2.299 |       0.213 |       0.609 |      0.047 |
| Ridge (no G1/G2)                    |        2.305 |       0.194 |       0.596 |      0.064 |
| Linear Regression (no G1/G2)        |        2.315 |       0.208 |       0.593 |      0.068 |

## Diagnostics
- Predicted-vs-actual and residual plots: *(insert images, saved from your 
  notebook — see code below to export them)*
- Model error on true dropout cases (G3=0) vs non-dropout cases: 
  *(paste your printed mean-error comparison)*

## Conclusion
*(1-2 sentences: which model you'd recommend, and the honest limitation — 
e.g., "the model performs well when G1/G2 are included but its practical 
value is limited since those aren't available before the course is graded; 
the no-grades model achieves R²=X, showing attendance and behavioral 
features alone have moderate but real predictive power.")*