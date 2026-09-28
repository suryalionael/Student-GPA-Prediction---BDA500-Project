# BDA500 Project 1 — Predicting Student Math Scores

Group regression project for **BDA500**. Predict a student's **math score** (continuous, 0–100) from their
reading and writing scores plus five sociodemographic factors, so schools can flag students at risk of
underperforming in mathematics and target early intervention.

## Team

| Member | Sections owned |
|---|---|
| **Lionael Surya** | 0 Setup, 1 Problem Framing, 2 Load & Clean, 3 EDA (distributions, correlations, cluster structure) |
| **Margil Parekh** | 5 Preprocessing pipeline, 6.1 Config counts, 6.2 Ridge results, comparison table + RMSE bar chart |
| **Chehoon Shin** | 6.3 Model selection, 6.4 Final test evaluation, 6.5 Residual analysis, report writing |

Every section header and code cell in the notebook carries a `# Contributor:` comment.

## Dataset

**Students Performance in Exams** — Kaggle, user `spscientist`. CC0 public domain.

- 1,000 rows, 8 columns
- **Unit of analysis:** one student
- **Target:** `math score` (continuous)
- **Numeric features:** `reading score`, `writing score`
- **Categorical features:** `gender`, `race/ethnicity`, `parental level of education`, `lunch`,
  `test preparation course`
- **Identifier / date / leakage columns:** none

No missing values and no duplicate rows. See Section 2.1–2.2 of the notebook.

### Data dictionary

| Column | Type | Role | Notes |
|---|---|---|---|
| `math score` | int | **Target** | Continuous 0–100 |
| `reading score` | int | Numeric feature | Literacy score, assumed known before math assessment |
| `writing score` | int | Numeric feature | Literacy score, assumed known before math assessment |
| `gender` | str | Categorical, binary | Sensitive attribute — monitored for bias |
| `race/ethnicity` | str | Categorical, nominal, 5 levels | Anonymised groups A–E; sensitive attribute |
| `parental level of education` | str | Categorical, ordered, 6 levels | Natural order exists, one-hot chosen |
| `lunch` | str | Categorical, binary | Free/reduced lunch proxies household income |
| `test preparation course` | str | Categorical, binary | Intervention a school can act on |

## Method

- **Split:** train/test at 10%, 25%, and 30% test size
- **Random state:** 0 and 42
- **Cross-validation:** 5-fold on the training set only
- **Preprocessing:** `ColumnTransformer` — `SimpleImputer` for numerics, `OneHotEncoder` for categoricals,
  `StandardScaler` for numerics. Fit on train only, transform test without refitting, to prevent leakage.
- **Models:** Linear Regression, Ridge (`alpha=1.0`), Polynomial Regression (`degree=3`) via `Pipeline`
- **Metrics:** MAE, RMSE, R² — mean and standard deviation across the 5 CV folds
- **Configurations:** 3 test sizes × 2 random states = 6 per model, 18 total

Random seed pinned to 42 at the top of Section 0.

## Results

See `BDA500_Group##_Model_Comparison.xlsx`, worksheet `Regression_Results`, and Section 6.3 of the
notebook for the comparison table and RMSE bar chart.

## Project layout

```
.
├── README.md
├── requirements.txt
├── StudentsPerformance.csv
├── BDA500_Project1_ML(2).pdf            # assignment spec
├── BDA500_Project1_sample-Results.xlsx  # Excel template
├── BDA500_ML_Project1_Questions.ipynb   # main deliverable notebook
├── notebooks/
│   └── 01_eda.ipynb
├── results/
│   ├── BDA500_Group##_Model_Comparison.xlsx
│   └── figures/
└── BDA500_Group##_Report.pdf
```

## Reproduce

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute BDA500_ML_Project1_Questions.ipynb --output executed.ipynb
```

Notebook resolves `StudentsPerformance.csv` and `results/figures/` relative to the repo root, so run
it from this directory.

## Due dates

- **Submission:** Wednesday, October 14, 2026, 11:59 PM ET
- **In-person demo:** Thursday, October 15, 2026
