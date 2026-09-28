import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder,
                                   PolynomialFeatures, StandardScaler)

from config import DATA_PATH, NOMINAL_FEATURES, NUM_FEATURES, ORDINAL_FEATURES


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df.drop_duplicates().reset_index(drop=True)


def build_preprocessor(with_poly=False, scale=True):
    """ColumnTransformer shared by all three models so their 6 configs stay comparable.

    with_poly=True expands only the numeric block to degree 3; categorical blocks
    are untouched to keep the feature count sane.
    """
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    if with_poly:
        num_steps.append(("poly", PolynomialFeatures(3, include_bias=False)))

    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), NUM_FEATURES),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1), ORDINAL_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
             NOMINAL_FEATURES),
        ],
        remainder="drop",
    )
