import itertools

import pandas as pd
from sklearn.model_selection import KFold, cross_validate, train_test_split

from config import CV_FOLDS, RANDOM_STATES, TARGET, TEST_SIZES
from preprocessing import load_data


def run_config(model_fn, model_name, best_params, feature_set="all_features"):
    """Cross-validated metrics on train only, for every (test_size, random_state).

    Preprocessing is fit inside the Pipeline on each train fold; test data never
    touches the fit, which prevents leakage. Returns one row per configuration.
    """
    df = load_data()
    X, y = df.drop(columns=TARGET), df[TARGET]
    rows = []
    for test_size, state in itertools.product(TEST_SIZES, RANDOM_STATES):
        X_tr, _, y_tr, _ = train_test_split(
            X, y, test_size=test_size, random_state=state)
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=state)
        scoring = {"MAE": "neg_mean_absolute_error",
                   "RMSE": "neg_root_mean_squared_error",
                   "R2": "r2"}
        scores = cross_validate(model_fn(), X_tr, y_tr, cv=cv, scoring=scoring)
        rows.append({
            "Model": model_name,
            "Feature_Set": feature_set,
            "CV_or_Split": f"{CV_FOLDS}-fold; test={test_size}; state={state}",
            "Best_Params": best_params,
            "MAE": -scores["test_MAE"].mean(),
            "RMSE": -scores["test_RMSE"].mean(),
            "R2": scores["test_R2"].mean(),
        })
    return pd.DataFrame(rows)


def config_count():
    return len(TEST_SIZES) * len(RANDOM_STATES) * 1  # one CV setting; degree/alpha fixed