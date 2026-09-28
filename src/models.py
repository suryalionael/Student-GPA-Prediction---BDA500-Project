from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from config import POLY_DEGREE, RIDGE_ALPHA
from preprocessing import build_preprocessor


def build_linear():
    return Pipeline([("prep", build_preprocessor()),
                     ("model", LinearRegression())])


def build_ridge(alpha=RIDGE_ALPHA):
    return Pipeline([("prep", build_preprocessor()),
                     ("model", Ridge(alpha=alpha))])


def build_polynomial(degree=POLY_DEGREE, alpha=RIDGE_ALPHA):
    # Degree-3 numeric terms need L2 regularization to stay stable at 26 features / 1000 rows.
    return Pipeline([("prep", build_preprocessor(with_poly=True)),
                     ("model", Ridge(alpha=alpha))])