"""Model construction, temporal splitting and the serialisable ChurnModel bundle."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config as C
from .feature_extraction import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, assert_no_leakage, snapshots_in_range

try:  # XGBoost is the primary candidate; scikit-learn's HistGradientBoosting is the documented fallback
    from xgboost import XGBClassifier
    HAVE_XGBOOST = True
except Exception:  # pragma: no cover - depends on the environment
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_XGBOOST = False

DERIVED_FEATURES = ["overdue_ratio"]


# --------------------------------------------------------------------------- #
# temporal split
# --------------------------------------------------------------------------- #
def partition_snapshots(name: str) -> list[str]:
    lo, hi = C.SPLITS[name]
    return snapshots_in_range(lo, hi)


def validate_temporal_split(train: pd.DataFrame | None, val: pd.DataFrame | None, test: pd.DataFrame | None,
                            outcome_days: int = C.OUTCOME_DAYS) -> None:
    """Raise unless partitions are chronological AND separated by at least one outcome window.

    Train labels look `outcome_days` ahead; if the validation cutoff were earlier than
    (last train cutoff + outcome_days) the training labels would contain information from the validation
    period. The same rule applies between validation and test.
    """
    parts = [("train", train), ("validation", val), ("test", test)]
    parts = [(n, d) for n, d in parts if d is not None and len(d)]
    for (n1, d1), (n2, d2) in zip(parts, parts[1:]):
        last, first = d1["snapshot_ts"].max(), d2["snapshot_ts"].min()
        if first < last + pd.Timedelta(days=outcome_days):
            raise ValueError(f"temporal leakage: {n2} starts {first:%Y-%m-%d} but {n1} outcome windows run until "
                             f"{last + pd.Timedelta(days=outcome_days):%Y-%m-%d}")
    seen: dict = {}      # no snapshot may appear in two partitions
    for n, d in parts:
        for s in d["snapshot_ts"].unique():
            if s in seen:
                raise ValueError(f"snapshot {s} appears in both {seen[s]} and {n}")
            seen[s] = n


# --------------------------------------------------------------------------- #
# preprocessing / models
# --------------------------------------------------------------------------- #
def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """overdue_ratio = days since last purchase / mean purchase interval (>1: later than this customer's own rhythm).

    NaN when the interval is undefined (fewer than two purchases) or zero - never fabricated.
    """
    X = X.copy()
    interval = X["mean_purchase_interval_days"].where(X["mean_purchase_interval_days"] > 0)
    X["overdue_ratio"] = X["days_since_last_purchase"] / interval
    return X


def _preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]   # indicator: missingness is information
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC_FEATURES + DERIVED_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005, sparse_output=False), CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)


def _derive_step() -> FunctionTransformer:
    return FunctionTransformer(add_derived_features, validate=False).set_output(transform="pandas")


def build_pipeline(kind: str, seed: int, **params) -> Pipeline:
    assert_no_leakage(FEATURES)
    if kind == "logreg":
        model = LogisticRegression(max_iter=3000, random_state=seed, **params)
        pre = _preprocessor(scale=True)
    elif kind == "random_forest":
        model = RandomForestClassifier(n_estimators=params.pop("n_estimators", 300), n_jobs=-1, random_state=seed, **params)
        pre = _preprocessor(scale=False)
    elif kind == "xgboost":
        if HAVE_XGBOOST:
            model = XGBClassifier(tree_method="hist", random_state=seed, n_jobs=4, eval_metric="aucpr", **params)
        else:   # documented fallback
            model = HistGradientBoostingClassifier(random_state=seed, **{k: v for k, v in params.items() if k in ("learning_rate", "max_depth")})
        pre = _preprocessor(scale=False)
    else:
        raise ValueError(kind)
    return Pipeline([("derive", _derive_step()), ("prep", pre), ("clf", model)])


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Only the declared feature columns reach a model (identifiers/labels are dropped here, once)."""
    return df[FEATURES]


# --------------------------------------------------------------------------- #
# baselines (score = higher means more likely to churn)
# --------------------------------------------------------------------------- #
def baseline_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        # a model that knows nothing: every customer gets the same score
        "baseline_constant": np.zeros(len(df)),
        # the obvious business rule: the longer since the last purchase, the likelier to lapse
        "baseline_recency_rule": df["days_since_last_purchase"].fillna(df["days_since_last_purchase"].max()).to_numpy(float),
        # a binary rule: nothing bought in the last 30 days
        "baseline_no_order_30d_rule": (df["orders_30d"] == 0).to_numpy(float),
    }


# --------------------------------------------------------------------------- #
# serialisable bundle
# --------------------------------------------------------------------------- #
class ChurnModel:
    """Fitted pipeline + isotonic calibrator + business threshold + metadata, saved as ONE joblib file.

    The threshold lives on the RAW probability scale and was chosen on validation data. The isotonic
    calibrator (also fitted on validation) only produces display probabilities; whether it helps is checked
    out-of-sample on the test partition (evaluate_model.py), not assumed.
    """

    def __init__(self, name: str, pipeline: Pipeline, calibrator: IsotonicRegression | None, threshold: float, metadata: dict):
        self.name, self.pipeline, self.calibrator, self.threshold, self.metadata = name, pipeline, calibrator, threshold, metadata

    def raw_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(select_columns(df))[:, 1]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated probability if a calibrator is attached, else the raw probability."""
        p = self.raw_proba(df)
        return self.calibrator.predict(p) if self.calibrator is not None else p

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Contact/no-contact decision: raw probability >= validation-chosen threshold."""
        return (self.raw_proba(df) >= self.threshold).astype(int)
