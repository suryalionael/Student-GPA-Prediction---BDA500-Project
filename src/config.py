from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "StudentsPerformance.csv"
FIG_DIR = REPO_ROOT / "results" / "figures"
RESULTS_CSV = REPO_ROOT / "results" / "Regression_Results.csv"
RESULTS_XLSX = REPO_ROOT / "results" / "BDA500_Group##_Model_Comparison.xlsx"

FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "math score"
NUM_FEATURES = ["reading score", "writing score"]
CAT_FEATURES = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]
# Ordered feature: the six education levels have a natural school-progress order.
ORDINAL_FEATURES = ["parental level of education"]
NOMINAL_FEATURES = [c for c in CAT_FEATURES if c not in ORDINAL_FEATURES]

ID_COLS, DATE_COLS, LEAKAGE_COLS = [], [], []

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TEST_SIZES = [0.10, 0.25, 0.30]
RANDOM_STATES = [0, 42]
CV_FOLDS = 5
RIDGE_ALPHA = 1.0
POLY_DEGREE = 3
